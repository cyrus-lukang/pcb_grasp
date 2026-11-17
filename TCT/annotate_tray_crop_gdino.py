"""两阶段 Grounding DINO:全图检 tray → 裁剪托盘 → 再检 pcb / white_frame。

输出类别默认: 0=pcb, 1=white_frame(不写 tray;加 --keep-tray 则 tray=2)

用法:
  # 单 PCB（轮廓匹配常用）: 有 ≥1 个 pcb 就写标签
  python annotate_tray_crop_gdino.py \\
    --images /path/to/images --out /path/to/out --viz --skip-white --max-pcb 1

  # 满盘九格旧逻辑
  python annotate_tray_crop_gdino.py \\
    --images ... --out ... --expect-total 9 --viz

  # 复用已有托盘框,跳过第一阶段
  python annotate_tray_crop_gdino.py \\
    --images ... --tray-labels some/labels --viz
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

from annotate import ClassConfig, GroundingDINO, MODELS

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLS_PCB, CLS_WHITE, CLS_TRAY = 0, 1, 2
NAMES = ["pcb", "white_frame", "tray"]
COLORS = [(0, 200, 0), (255, 180, 0), (0, 160, 255)]

TRAY_CC = ClassConfig(
    name="tray",
    prompts=["black tray", "grid tray", "compartment tray"],
    threshold=0.30,
)
PCB_CC = ClassConfig(
    name="pcb",
    # 多短语提高召回;阈值略降。过大框仍由 filter_small / NMS 滤掉
    prompts=[
        "green circuit board",
        "PCB",
        "circuit board",
        "green square circuit board",
    ],
    threshold=0.20,
)
WHITE_CC = ClassConfig(
    name="white_frame",
    prompts=[
        "white square frame on black",
        "white rectangular outline",
        "white square outline",
    ],
    threshold=0.20,
)


def load_yolo_trays(path: Path, w: int, h: int, tray_cls: int = 0):
    """从已有 labels 读托盘框(默认旧标注 tray=0)。"""
    if not path.exists():
        return []
    trays = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        if int(float(p[0])) != tray_cls:
            continue
        cx, cy, bw, bh = map(float, p[1:5])
        trays.append([
            (cx - bw / 2) * w, (cy - bh / 2) * h,
            (cx + bw / 2) * w, (cy + bh / 2) * h,
        ])
    trays.sort(key=lambda t: (t[2] - t[0]) * (t[3] - t[1]), reverse=True)
    return trays


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = ((a[2] - a[0]) * (a[3] - a[1]) +
          (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def nms(boxes, scores, thr=0.4):
    if not boxes:
        return []
    import numpy as np
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    idx = scores.argsort()[::-1]
    keep = []
    while len(idx):
        i = int(idx[0])
        keep.append(i)
        if len(idx) == 1:
            break
        rest = idx[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        ua = ((boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) +
              (boxes[rest, 2] - boxes[rest, 0]) *
              (boxes[rest, 3] - boxes[rest, 1]) - inter)
        idx = rest[(inter / np.maximum(ua, 1e-6)) < thr]
    return keep


def pick_tray_dets(dets, img_area, min_frac=0.04):
    cands = []
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        a = (x2 - x1) * (y2 - y1)
        if a >= min_frac * img_area:
            cands.append((d.bbox, d.score, a))
    cands.sort(key=lambda x: x[2], reverse=True)
    kept = []
    for box, score, _ in cands:
        if all(iou(box, k[0]) < 0.5 for k in kept):
            kept.append((box, score))
    return kept[:1]  # 本场景通常一盘


def crop_with_pad(image: Image.Image, box, pad_frac=0.02, pad_px=8):
    W, H = image.size
    x1, y1, x2, y2 = box
    pw = max(pad_px, (x2 - x1) * pad_frac)
    ph = max(pad_px, (y2 - y1) * pad_frac)
    cx1 = max(0, int(x1 - pw))
    cy1 = max(0, int(y1 - ph))
    cx2 = min(W, int(x2 + pw))
    cy2 = min(H, int(y2 + ph))
    crop = image.crop((cx1, cy1, cx2, cy2))
    return crop, (cx1, cy1, cx2, cy2)


def map_box_to_full(box, origin):
    ox, oy = origin[0], origin[1]
    return [box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy]


def filter_small(boxes_scores, tray_box, max_tray_frac=0.25, max_img_frac=0.06,
                 img_area=1.0, max_tray_iou=0.35):
    """丢掉接近整盘/整图的大框。"""
    ta = (tray_box[2] - tray_box[0]) * (tray_box[3] - tray_box[1])
    out = []
    for box, score in boxes_scores:
        a = (box[2] - box[0]) * (box[3] - box[1])
        if a > max_img_frac * img_area or a > max_tray_frac * ta:
            continue
        if iou(box, tray_box) >= max_tray_iou:
            continue
        out.append((box, score))
    return out


def green_frac(rgb: np.ndarray, box, h_lo=25, s_min=20, v_min=20) -> float:
    """框内 HSV 绿色像素占比。真 PCB 通常 >=0.1,白框误检接近 0。"""
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    hsv = cv2.cvtColor(rgb[y1:y2, x1:x2], cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, (h_lo, s_min, v_min), (100, 255, 255))
    return float((m > 0).mean())


def filter_pcb_by_green(rgb, boxes_scores, min_green=0.04):
    """强制绿度过滤:无绿则视为白框/空槽误检。"""
    kept = []
    for box, score in boxes_scores:
        gf = green_frac(rgb, box)
        if gf < min_green:
            continue
        kept.append((box, score))
    return kept


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def containment_frac(inner, outer) -> float:
    """inner 被 outer 覆盖的面积占比。"""
    xx1 = max(inner[0], outer[0])
    yy1 = max(inner[1], outer[1])
    xx2 = min(inner[2], outer[2])
    yy2 = min(inner[3], outer[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    a = _area(inner)
    return inter / a if a > 1e-6 else 0.0


def drop_outer_boxes(items, contain_thr=0.7):
    """大框套小框时去掉大框。

    items: list of (cid, box, score)
    同类/跨类均适用:小框被大框覆盖比例 >= contain_thr 则丢大框。
    """
    n = len(items)
    drop = set()
    for i in range(n):
        if i in drop:
            continue
        _, bi, _ = items[i]
        ai = _area(bi)
        for j in range(n):
            if i == j or j in drop:
                continue
            _, bj, _ = items[j]
            aj = _area(bj)
            if ai <= aj:
                continue
            # i 更大:若 j 大部分在 i 内,丢 i
            if containment_frac(bj, bi) >= contain_thr:
                drop.add(i)
                break
    return [items[k] for k in range(n) if k not in drop]


def to_yolo(cid, box, w, h, score=None):
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    s = f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
    if score is not None:
        s += f" {score:.4f}"
    return s


def _font(size=14):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser(description="gdino 两阶段:tray裁剪后再检pcb/white")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gdino-base", choices=list(MODELS))
    ap.add_argument("--tray-labels", default=None,
                    help="已有托盘 YOLO labels;提供则跳过第一阶段检 tray")
    ap.add_argument("--input-tray-cls", type=int, default=0,
                    help="tray-labels 中托盘类别 id")
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--keep-tray", action="store_true")
    ap.add_argument("--pad-frac", type=float, default=0.03)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--min-green", type=float, default=0.04,
                    help="pcb 框内绿色像素占比下限,低于则丢弃(滤白框误检)")
    ap.add_argument("--contain-thr", type=float, default=0.7,
                    help="小框被大框覆盖比例阈值,达到则去掉大框")
    ap.add_argument(
        "--expect-total",
        type=int,
        default=0,
        help="pcb+white 必须等于该数才写标签; 0=不强制(默认,适合单PCB)。满盘九格用 9",
    )
    ap.add_argument(
        "--min-pcb",
        type=int,
        default=1,
        help="至少检出多少个 pcb 才写标签(默认 1)",
    )
    ap.add_argument(
        "--max-pcb",
        type=int,
        default=0,
        help="最多保留几个 pcb(按分数); 0=不限制。单目标建议 1",
    )
    ap.add_argument(
        "--skip-white",
        action="store_true",
        help="不检测 white_frame(单 PCB 场景推荐)",
    )
    ap.add_argument(
        "--allow-no-tray",
        action="store_true",
        help="检不到 tray 时用整图当裁剪区继续检 pcb",
    )
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    images = Path(args.images)
    out = Path(args.out)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    if args.skip_white:
        out_names = ["pcb"] + (["tray"] if args.keep_tray else [])
    else:
        out_names = NAMES if args.keep_tray else ["pcb", "white_frame"]
    (out / "classes.txt").write_text("\n".join(
        (["pcb", "white_frame", "tray"] if args.keep_tray and not args.skip_white
         else out_names)
    ) + "\n")
    if args.viz:
        (out / "viz").mkdir(parents=True, exist_ok=True)
        (out / "crops").mkdir(parents=True, exist_ok=True)

    img_paths = sorted(p for p in images.iterdir() if p.suffix.lower() in IMG_EXTS)
    if args.limit > 0:
        img_paths = img_paths[: args.limit]

    backend, model_id = MODELS[args.model]
    if backend != "grounding_dino":
        raise SystemExit("[error] 本脚本仅支持 grounding_dino 后端")
    det = GroundingDINO(model_id)
    font = _font()
    tray_lab = Path(args.tray_labels) if args.tray_labels else None

    tot = {"pcb": 0, "white_frame": 0, "tray": 0}
    incomplete = []
    print(
        f"[info] {len(img_paths)} 张  两阶段gdino  "
        f"tray_labels={'yes' if tray_lab else 'detect'}  "
        f"expect_total={args.expect_total} min_pcb={args.min_pcb} "
        f"max_pcb={args.max_pcb or 'inf'} skip_white={args.skip_white}"
    )

    for idx, ip in enumerate(img_paths, 1):
        image = Image.open(ip).convert("RGB")
        rgb = np.array(image)
        W, H = image.size
        img_area = float(W * H)

        # ---- stage1: tray ----
        used_full = False
        if tray_lab is not None:
            trays = load_yolo_trays(
                tray_lab / (ip.stem + ".txt"), W, H, args.input_tray_cls)
            tray_list = [(t, 1.0) for t in trays[:1]]
        else:
            tray_dets = det.detect_class(image, TRAY_CC, args.text_threshold)
            tray_list = pick_tray_dets(tray_dets, img_area)

        if not tray_list:
            if not args.allow_no_tray:
                lab = out / "labels" / (ip.stem + ".txt")
                if lab.exists():
                    lab.unlink()
                print(f"[{idx}/{len(img_paths)}] {ip.name}: NO TRAY (skip label)")
                incomplete.append((ip.name, 0, 0))
                continue
            tray_box, tray_score = [0.0, 0.0, float(W), float(H)], 0.0
            used_full = True
        else:
            tray_box, tray_score = tray_list[0]

        crop, origin = crop_with_pad(image, tray_box, pad_frac=args.pad_frac)
        if args.viz:
            crop.save(out / "crops" / ip.name)

        # ---- stage2: pcb (+ optional white) on crop ----
        pcb_dets = det.detect_class(crop, PCB_CC, args.text_threshold)
        white_dets = (
            [] if args.skip_white
            else det.detect_class(crop, WHITE_CC, args.text_threshold)
        )

        pcbs = [(map_box_to_full(d.bbox, origin), d.score) for d in pcb_dets]
        whites = [(map_box_to_full(d.bbox, origin), d.score) for d in white_dets]
        if not used_full:
            pcbs = filter_small(pcbs, tray_box, img_area=img_area)
            whites = filter_small(whites, tray_box, img_area=img_area)
        n_before_green = len(pcbs)
        pcbs = filter_pcb_by_green(rgb, pcbs, min_green=args.min_green)
        n_drop_green = n_before_green - len(pcbs)

        # 同类 NMS
        if pcbs:
            keep = nms([b for b, _ in pcbs], [s for _, s in pcbs], 0.4)
            pcbs = [pcbs[i] for i in keep]
        if whites:
            keep = nms([b for b, _ in whites], [s for _, s in whites], 0.4)
            whites = [whites[i] for i in keep]

        # 与 tray 中心约束:框中心须在托盘内(整图模式跳过)
        def in_tray(box, inset=0.04):
            if used_full:
                return True
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            tw = tray_box[2] - tray_box[0]
            th = tray_box[3] - tray_box[1]
            return (tray_box[0] + inset * tw <= cx <= tray_box[2] - inset * tw and
                    tray_box[1] + inset * th <= cy <= tray_box[3] - inset * th)

        pcbs = [(b, s) for b, s in pcbs if in_tray(b)]
        whites = [(b, s) for b, s in whites if in_tray(b)]

        # 合并后去套娃:大框套小框(同类/跨类)一律丢大框
        items = ([(CLS_PCB, b, s) for b, s in pcbs] +
                 [(CLS_WHITE, b, s) for b, s in whites])
        n_before_nest = len(items)
        items = drop_outer_boxes(items, contain_thr=args.contain_thr)
        n_drop_nest = n_before_nest - len(items)
        final_pcb = [(b, s) for c, b, s in items if c == CLS_PCB]
        final_white = [(b, s) for c, b, s in items if c == CLS_WHITE]

        # 单/多目标:按分数截断 pcb 数量
        if args.max_pcb > 0 and len(final_pcb) > args.max_pcb:
            final_pcb = sorted(final_pcb, key=lambda x: x[1], reverse=True)[
                : args.max_pcb
            ]

        lines = []
        kept_viz = []
        if args.keep_tray and not used_full:
            lines.append(to_yolo(CLS_TRAY, tray_box, W, H, tray_score))
            kept_viz.append((CLS_TRAY, tray_box, tray_score))
        for b, s in final_pcb:
            lines.append(to_yolo(CLS_PCB, b, W, H, s))
            kept_viz.append((CLS_PCB, b, s))
        for b, s in final_white:
            lines.append(to_yolo(CLS_WHITE, b, W, H, s))
            kept_viz.append((CLS_WHITE, b, s))

        n_pcb, n_white = len(final_pcb), len(final_white)
        extras = []
        if used_full:
            extras.append("full_image")
        if n_drop_green:
            extras.append(f"drop_green={n_drop_green}")
        if n_drop_nest:
            extras.append(f"drop_outer={n_drop_nest}")
        extra = (" " + " ".join(extras)) if extras else ""
        lab = out / "labels" / (ip.stem + ".txt")

        ok = n_pcb >= args.min_pcb
        if args.expect_total > 0:
            ok = ok and (n_pcb + n_white == args.expect_total)

        if not ok:
            if lab.exists():
                lab.unlink()
            incomplete.append((ip.name, n_pcb, n_white))
            reason = (
                f"expect_total={args.expect_total}"
                if args.expect_total > 0
                else f"min_pcb={args.min_pcb}"
            )
            print(f"[{idx}/{len(img_paths)}] {ip.name}: "
                  f"pcb={n_pcb} white={n_white} SKIP ({reason}){extra}")
        else:
            if args.keep_tray and not used_full:
                tot["tray"] += 1
            tot["pcb"] += n_pcb
            tot["white_frame"] += n_white
            lab.write_text("\n".join(lines) + ("\n" if lines else ""))
            print(f"[{idx}/{len(img_paths)}] {ip.name}: "
                  f"pcb={n_pcb} white={n_white} crop={crop.size}{extra}")

        if args.viz and ok:
            viz = image.copy()
            d = ImageDraw.Draw(viz)
            d.rectangle(tray_box, outline=(0, 160, 255), width=2)
            for cid, box, score in kept_viz:
                col = COLORS[cid]
                d.rectangle(box, outline=col, width=3)
                d.text((box[0] + 1, max(0, box[1] - 14)),
                       f"{NAMES[cid]} {score:.2f}", fill=col, font=font)
            viz.save(out / "viz" / ip.name)

    print(f"[done] {tot} -> {out}")
    print(f"[done] skipped (no label): {len(incomplete)} / {len(img_paths)}")
    for n, a, b in incomplete:
        print(f"  {n}\tpcb={a}\twhite={b}\tsum={a + b}")
    skip_txt = "\n".join(
        f"{n}\tpcb={a}\twhite={b}\tsum={a + b}" for n, a, b in incomplete
    ) + ("\n" if incomplete else "")
    (out / "skipped.txt").write_text(skip_txt)
    # 兼容旧文件名
    (out / "skipped_ne9.txt").write_text(skip_txt)
    (out / "incomplete_lt9.txt").write_text(skip_txt)


if __name__ == "__main__":
    main()
