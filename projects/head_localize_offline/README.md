# 头部离线定位

手眼：`20261123_154818_head_chest_leftwrist/head_fisheye_right_s36_handeye_from_stereo.yaml`  
模型：`yolopose_3cls_20261105163435_041ee5/best.pt`

## 相机对应

| 落盘 | portal | 用途 |
|------|--------|------|
| `raw_left.jpg` | cam7 = head_fisheye_right | 手眼 + 检测 + 定位 |
| `raw_right.jpg` | cam6 | 仅双目右图算 FFS 深度 |

## 跑一次

```bash
cd /home/ubuntu/workspace/pcb_grasp
.venv/bin/python projects/head_localize_offline/scripts/run_localize.py \
  --capture output/live_vis/batch_029_20261124_120504/head \
  --out projects/head_localize_offline/output/batch029_head \
  --use-saved-depth
```

输出：`pred.json` / `pose.jpg` / `raw_right_detect.jpg`
