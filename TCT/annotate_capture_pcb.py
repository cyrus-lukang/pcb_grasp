"""capture/out{1,2,3,4}: GDINO 检 PCB → SAM2 分割拟合四角点。

参考 annotate_yuantu_wrist 的目录/输出结构;检测改用 Grounding DINO
(腕部图有绿地板/夹爪,纯 CV 绿连通域误检多,红板也需开放词检测)。

流水线:
  1) GDINO 分检 green / red PCB
  2) HSV 颜色占比 + 暗度 + 面积/长宽比过滤(抑地板与夹爪)
  3) NMS / 去套娃,写为同一类 pcb
  4) SAM2 box(+中心点) → 掩膜后处理 → 贴轮廓四边形角点

输出(每个 root):
  <det-subdir>/labels/*.txt       YOLO 检测框 0=pcb
  <out-subdir>/labels/*.txt       n,x1,y1,...,x4,y4,0  (像素, TL→TR→BR→BL)
  classes.txt / viz/ / overlay/   (--viz)

用法:
  .venv/bin/python annotate_capture_pcb.py \
    --roots /media/nas_users/wangzixiang/data1/capture/out{1,2,3,4} \
    --viz

  # 仅分割(已有检测框):
  .venv/bin/python annotate_capture_pcb.py \
    --roots ... --skip-det --viz
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from annotate import ClassConfig, GroundingDINO, MODELS
from annotate_green_red_pcb_gdino import (
    cross_class_nms,
    drop_outer_boxes,
    green_frac,
    nms,
    red_frac,
)
from annotate_yuantu_wrist import _area, to_yolo
from annotate_tray_divider import TRAY_CC, pick_tray
from seg_filter_imu_sam2 import (
    CORNER_COLORS,
    Box,
    Sam2Segmenter,
    draw_mask_overlay,
    load_yolo,
)
from seg_filter_pcb_sam2 import (
    CORNER_START_SEQ,
    analyze_pcb,
    expand_mask_with_metal,
    refine_poor_masks,
    strip_white_support,
)
import seg_filter_pcb_sam2 as pcb_sam2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLS_PCB = 0
NAMES = ["pcb"]
COLORS = [(0, 200, 0)]

GREEN_CC = ClassConfig(
    name="green_pcb",
    prompts=["green circuit board", "green PCB", "PCB"],
    threshold=0.22,
)
RED_CC = ClassConfig(
    name="red_pcb",
    prompts=["red circuit board", "red PCB"],
    threshold=0.22,
)

# 内部 tag,仅用于过滤与 viz 着色
TAG_G, TAG_R = 0, 1

PCB_GEOM = dict(max_aspect=2.5, min_area_ratio=0.5, min_rect_fill=0.55,
                min_blob_ratio=0.20)


def dark_frac(rgb: np.ndarray, box, v_thr: int = 70) -> float:
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return 1.0
    hsv = cv2.cvtColor(rgb[y1:y2, x1:x2], cv2.COLOR_RGB2HSV)
    return float((hsv[:, :, 2] < v_thr).mean())


def aspect(box) -> float:
    w = max(1.0, box[2] - box[0])
    h = max(1.0, box[3] - box[1])
    return max(w / h, h / w)


def filter_dets(rgb, dets, color: str, img_area: float,
                min_color: float, max_dark: float,
                min_area_frac: float, max_area_frac: float,
                max_ar: float):
    kept = []
    for d in dets:
        box, score = d.bbox, d.score
        a = _area(box)
        if a < min_area_frac * img_area or a > max_area_frac * img_area:
            continue
        if aspect(box) > max_ar:
            continue
        gf, rf = green_frac(rgb, box), red_frac(rgb, box)
        if color == "green":
            if gf < min_color or rf > gf:
                continue
        else:
            if rf < min_color or gf > rf:
                continue
        if dark_frac(rgb, box) > max_dark:
            continue
        kept.append((box, score))
    return kept


def _font(size=14):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def collect_images(roots: list[Path], per_root_limit: int = 0):
    pairs = []
    for root in roots:
        imgs = sorted(p for p in root.iterdir() if p.suffix.lower() in IMG_EXTS)
        if per_root_limit > 0:
            imgs = imgs[:per_root_limit]
        for p in imgs:
            pairs.append((root, p))
    return pairs


def _box_center(b: Box) -> tuple[float, float]:
    return ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0)


def draw_det_viz(image: Image.Image, items, font, out_path: Path):
    viz = image.copy()
    d = ImageDraw.Draw(viz)
    for tag, box, score in items:
        col = (0, 200, 0) if tag == TAG_G else (220, 40, 40)
        name = "pcb(g)" if tag == TAG_G else "pcb(r)"
        d.rectangle(box, outline=col, width=3)
        d.text((box[0] + 1, max(0, box[1] - 14)),
               f"{name} {score:.2f}", fill=col, font=font)
    viz.save(out_path)


def draw_corner_viz(image: Image.Image, boxes: list[Box], results, font,
                    out_path: Path):
    im = image.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    for i, (b, r) in enumerate(zip(boxes, results), 1):
        color = COLORS[0] if r.keep else (255, 0, 0)
        d.rectangle([b.x1, b.y1, b.x2, b.y2], outline=color, width=2)
        if r.keep and r.corners:
            pts = [(p[0], p[1]) for p in r.corners]
            d.line(pts + [pts[0]], fill=(0, 160, 255), width=2)
            for ci, (x, y) in enumerate(pts):
                cc = CORNER_COLORS[ci % 4]
                d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=cc,
                          outline=(0, 0, 0))
                d.text((x + 5, y - 6), str(ci + 1), fill=cc, font=font)
        tag = f"#{i} pcb" if r.keep else f"#{i} pcb {r.reason}"
        l, t, rr, bb = d.textbbox((0, 0), tag, font=font)
        ty = max(0, b.y1 - (bb - t) - 4)
        d.rectangle([b.x1, ty, b.x1 + (rr - l) + 4, ty + (bb - t) + 4],
                    fill=color)
        d.text((b.x1 + 2, ty + 2), tag, fill=(255, 255, 255), font=font)
    im.save(out_path)


def sam2_corners(
    image: Image.Image,
    rgb: np.ndarray,
    boxes: list[Box],
    seg: Sam2Segmenter,
    args,
) -> tuple[list, list[np.ndarray]]:
    """SAM2 分割 + 四角点拟合。返回 (SegResult 列表, 后处理 masks)。"""
    use_center = not args.no_center_prompt
    do_refine = not args.no_center_refine
    do_strip = not args.no_strip_white
    do_black = not args.no_strip_black

    centers = [_box_center(b) for b in boxes] if use_center else None
    masks = seg.masks_for_boxes(image, boxes, points=centers)
    if do_refine and boxes:
        masks, _ = refine_poor_masks(
            seg, image, boxes, masks,
            score_thresh=args.center_score_thresh)

    if do_strip:
        masks = [
            strip_white_support(m, rgb) if b.cls == CLS_PCB else m
            for b, m in zip(boxes, masks)]

    if do_black:
        masks = [
            expand_mask_with_metal(m, rgb, box=b) if b.cls == CLS_PCB else m
            for b, m in zip(boxes, masks)]

    vals = [int(m.sum()) for m in masks if int(m.sum()) > 0]
    med_area = float(np.median(vals)) if vals else 0.0

    geom = dict(PCB_GEOM)
    if args.min_rect_fill is not None:
        geom["min_rect_fill"] = args.min_rect_fill
    if args.seg_max_aspect is not None:
        geom["max_aspect"] = args.seg_max_aspect

    results = []
    for b, m in zip(boxes, masks):
        res = analyze_pcb(
            m, geom["min_blob_ratio"], geom["max_aspect"],
            med_area, geom["min_area_ratio"], geom["min_rect_fill"],
            rgb=None, box=b, cls=CLS_PCB, order_by_metal=False)
        results.append(res)
    return results, masks


def corner_lines(results) -> list[str]:
    lines = []
    obj_i = 0
    for r in results:
        if not r.keep or not r.corners:
            continue
        obj_i += 1
        parts = [str(obj_i)]
        for x, y in r.corners:
            parts += [f"{float(x):.4f}", f"{float(y):.4f}"]
        parts.append(str(CLS_PCB))
        lines.append(",".join(parts))
    return lines


def main():
    ap = argparse.ArgumentParser(
        description="capture out* 腕部 PCB GDINO 检测 + SAM2 四角点")
    ap.add_argument("--roots", nargs="+", required=True,
                    help="图像目录,如 capture/out1 out2 out3 out4")
    ap.add_argument("--out-subdir", default="labels_pcb_corners",
                    help="角点输出子目录(相对每个 root)")
    ap.add_argument("--det-subdir", default="labels_pcb",
                    help="检测框 YOLO 输出子目录(相对每个 root)")
    ap.add_argument("--out", default=None,
                    help="若指定则统一输出到该目录(角点)")
    ap.add_argument("--model", default="gdino-base", choices=list(MODELS))
    ap.add_argument("--profile", choices=("wrist", "head"), default="wrist",
                    help="检测阈值预设: wrist=腕部相机(默认); head=头部相机/data_head")
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--skip-det", action="store_true",
                    help="跳过检测,从 --det-subdir/labels 读框做 SAM2")
    ap.add_argument("--skip-seg", action="store_true",
                    help="仅检测,不写角点")
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--min-green", type=float, default=None)
    ap.add_argument("--min-red", type=float, default=None)
    ap.add_argument("--max-dark", type=float, default=None,
                    help="框内过暗则视为托盘空槽/夹爪误检")
    ap.add_argument("--min-area", type=float, default=None)
    ap.add_argument("--max-area", type=float, default=None)
    ap.add_argument("--max-aspect", type=float, default=None,
                    help="检测框长宽比上限")
    ap.add_argument("--contain-thr", type=float, default=0.7)
    ap.add_argument("--cross-nms", type=float, default=0.45)
    ap.add_argument("--inside-tray", action="store_true",
                    help="只保留中心落在托盘框内的 PCB(抑电缆/地板误检)")
    ap.add_argument("--limit", type=int, default=0,
                    help="每个 root 只处理前 N 张,0=全部")
    # SAM2
    ap.add_argument("--sam2-model", default="facebook/sam2.1-hiera-large")
    ap.add_argument("--save-mask", action="store_true")
    ap.add_argument("--no-center-prompt", action="store_true")
    ap.add_argument("--no-center-refine", action="store_true")
    ap.add_argument("--center-score-thresh", type=float, default=0.88)
    ap.add_argument("--no-strip-white", action="store_true")
    ap.add_argument("--no-strip-black", action="store_true")
    ap.add_argument("--min-rect-fill", type=float, default=None)
    ap.add_argument("--seg-max-aspect", type=float, default=None,
                    help="SAM2 拟合宽高比上限(默认 2.5)")
    ap.add_argument("--corner-start", choices=("tl", "tr", "br", "bl"),
                    default="tl")
    pre = ap.parse_args()

    PROFILE_DEFAULTS = {
        "wrist": dict(min_green=0.1, min_red=0.12, max_dark=0.72,
                      min_area=0.02, max_area=0.22, max_aspect=2.1,
                      inside_tray=False),
        "head": dict(min_green=0.1, min_red=0.08, max_dark=0.78,
                     min_area=0.012, max_area=0.16, max_aspect=2.3,
                     inside_tray=True),
    }
    prof = PROFILE_DEFAULTS[pre.profile]
    args = pre
    for k in ("min_green", "min_red", "max_dark", "min_area", "max_area",
              "max_aspect"):
        if getattr(args, k) is None:
            setattr(args, k, prof[k])
    if not args.inside_tray and prof["inside_tray"]:
        args.inside_tray = prof["inside_tray"]

    if args.skip_det and args.skip_seg:
        raise SystemExit("[error] --skip-det 与 --skip-seg 不能同时开")

    pcb_sam2.CORNER_START = args.corner_start

    roots = [Path(r) for r in args.roots]
    for r in roots:
        if not r.is_dir():
            raise SystemExit(f"[error] 目录不存在: {r}")

    pairs = collect_images(roots, per_root_limit=args.limit)
    if not pairs:
        raise SystemExit("[error] 未找到图片")

    unified = Path(args.out) if args.out else None
    root_corner_outs: dict[Path, Path] = {}
    root_det_outs: dict[Path, Path] = {}
    if unified:
        (unified / "labels").mkdir(parents=True, exist_ok=True)
        (unified / "classes.txt").write_text("pcb\n")
        if not args.skip_det:
            (unified / "labels_det").mkdir(parents=True, exist_ok=True)
        if args.viz:
            (unified / "viz").mkdir(parents=True, exist_ok=True)
            if not args.skip_seg:
                (unified / "overlay").mkdir(parents=True, exist_ok=True)
            if not args.skip_det:
                (unified / "viz_det").mkdir(parents=True, exist_ok=True)
    else:
        for r in roots:
            corner_o = r / args.out_subdir
            (corner_o / "labels").mkdir(parents=True, exist_ok=True)
            (corner_o / "classes.txt").write_text("pcb\n")
            if args.viz:
                (corner_o / "viz").mkdir(parents=True, exist_ok=True)
                if not args.skip_seg:
                    (corner_o / "overlay").mkdir(parents=True, exist_ok=True)
            root_corner_outs[r] = corner_o
            if not args.skip_det:
                det_o = r / args.det_subdir
                (det_o / "labels").mkdir(parents=True, exist_ok=True)
                (det_o / "classes.txt").write_text("pcb\n")
                if args.viz:
                    (det_o / "viz").mkdir(parents=True, exist_ok=True)
                root_det_outs[r] = det_o

    det = None
    if not args.skip_det:
        backend, model_id = MODELS[args.model]
        if backend != "grounding_dino":
            raise SystemExit("[error] 仅支持 grounding_dino")
        det = GroundingDINO(model_id)

    seg = None
    if not args.skip_seg:
        seg = Sam2Segmenter(args.sam2_model)

    font = _font()
    tot_pcb = tot_corners = 0
    empty = []
    geom_seq = CORNER_START_SEQ[args.corner_start]
    print(f"[info] {len(pairs)} 张  profile={args.profile}  "
          f"det={'off' if args.skip_det else args.model}  "
          f"seg={'off' if args.skip_seg else args.sam2_model}  "
          f"corner_start={args.corner_start}({geom_seq})  "
          f"roots={[r.name for r in roots]}")

    for idx, (root, ip) in enumerate(pairs, 1):
        image = Image.open(ip).convert("RGB")
        rgb = np.array(image)
        W, H = image.size
        stem = ip.stem if not unified else f"{root.name}_{ip.stem}"

        det_items = []
        boxes: list[Box] = []

        if not args.skip_det:
            img_area = float(W * H)
            g_dets = det.detect_class(image, GREEN_CC, args.text_threshold)
            r_dets = det.detect_class(image, RED_CC, args.text_threshold)
            greens = filter_dets(
                rgb, g_dets, "green", img_area,
                min_color=args.min_green, max_dark=args.max_dark,
                min_area_frac=args.min_area, max_area_frac=args.max_area,
                max_ar=args.max_aspect)
            reds = filter_dets(
                rgb, r_dets, "red", img_area,
                min_color=args.min_red, max_dark=args.max_dark,
                min_area_frac=args.min_area, max_area_frac=args.max_area,
                max_ar=args.max_aspect)
            if greens:
                keep = nms([b for b, _ in greens], [s for _, s in greens], 0.4)
                greens = [greens[i] for i in keep]
            if reds:
                keep = nms([b for b, _ in reds], [s for _, s in reds], 0.4)
                reds = [reds[i] for i in keep]
            det_items = ([(TAG_G, b, s) for b, s in greens] +
                         [(TAG_R, b, s) for b, s in reds])
            det_items = drop_outer_boxes(det_items, contain_thr=args.contain_thr)
            if args.cross_nms < 1.0:
                det_items = cross_class_nms(det_items, thr=args.cross_nms)
            if args.inside_tray:
                tray_dets = det.detect_class(image, TRAY_CC, 0.20)
                tray = pick_tray(tray_dets, W, H)
                pad = 24.0
                tx1, ty1, tx2, ty2 = tray
                kept_in = []
                for tag, b, s in det_items:
                    cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                    if (tx1 - pad <= cx <= tx2 + pad
                            and ty1 - pad <= cy <= ty2 + pad):
                        kept_in.append((tag, b, s))
                det_items = kept_in
            boxes = [Box(b[0], b[1], b[2], b[3], cls=CLS_PCB)
                     for _, b, _ in det_items]

            det_out = (unified / "labels_det") if unified else (
                root_det_outs[root] / "labels")
            det_out.mkdir(parents=True, exist_ok=True)
            lines = [to_yolo(CLS_PCB, b, W, H, s) for _, b, s in det_items]
            (det_out / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""))
            if args.viz:
                viz_name = ip.name if not unified else f"{root.name}_{ip.name}"
                det_viz_dir = (unified / "viz_det") if unified else (
                    root_det_outs[root] / "viz")
                draw_det_viz(image, det_items, font, det_viz_dir / viz_name)
        else:
            lbl = root / args.det_subdir / "labels" / f"{ip.stem}.txt"
            if not lbl.is_file():
                lbl = root / args.det_subdir / f"{ip.stem}.txt"
            boxes = load_yolo(lbl, W, H, cls_ids={CLS_PCB})

        n_g = sum(1 for t, _, _ in det_items if t == TAG_G)
        n_r = sum(1 for t, _, _ in det_items if t == TAG_R)
        tot_pcb += len(boxes)

        corner_results = []
        fit_masks = []
        if not args.skip_seg and seg is not None:
            corner_results, fit_masks = sam2_corners(
                image, rgb, boxes, seg, args)
            n_keep = sum(1 for r in corner_results if r.keep)
            tot_corners += n_keep
            corner_out = unified if unified else root_corner_outs[root]
            clines = corner_lines(corner_results)
            (corner_out / "labels" / f"{stem}.txt").write_text(
                "\n".join(clines) + ("\n" if clines else ""))
            if args.viz:
                viz_name = ip.name if not unified else f"{root.name}_{ip.name}"
                draw_corner_viz(image, boxes, corner_results, font,
                                corner_out / "viz" / viz_name)
                draw_mask_overlay(image, fit_masks, corner_results,
                                  viz_name, corner_out)
            if args.save_mask:
                full_mask = np.zeros((H, W), np.uint8)
                for m, r in zip(fit_masks, corner_results):
                    if r.keep:
                        full_mask = np.maximum(full_mask, m.astype(np.uint8) * 255)
                masks_dir = corner_out / "masks"
                masks_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(masks_dir / f"{stem}.png"), full_mask)

        if not boxes:
            empty.append(f"{root.name}/{ip.name}")

        if args.skip_seg:
            print(f"[{idx}/{len(pairs)}] {root.name}/{ip.name}: "
                  f"pcb={n_g + n_r}(g={n_g}+r={n_r})")
        else:
            n_keep = sum(1 for r in corner_results if r.keep) if corner_results else 0
            print(f"[{idx}/{len(pairs)}] {root.name}/{ip.name}: "
                  f"det={len(boxes)}(g={n_g}+r={n_r}) -> corners={n_keep}")

    print(f"[done] det_boxes={tot_pcb}  corners={tot_corners}")
    print(f"[done] empty: {len(empty)} / {len(pairs)}")
    empty_path = (unified / "empty.txt") if unified else (
        roots[0].parent / "empty_pcb.txt")
    empty_path.write_text("\n".join(empty) + ("\n" if empty else ""))
    for n in empty[:40]:
        print(f"  {n}")
    if len(empty) > 40:
        print(f"  ... +{len(empty) - 40} more")
    if not args.skip_seg:
        out_hint = unified if unified else f"<root>/{args.out_subdir}"
        print(f"[done] 角点 -> {out_hint}/labels")


if __name__ == "__main__":
    main()
