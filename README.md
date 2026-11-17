# PCB 粗定位 + 精定位抓取

独立项目：`/home/ubuntu/workspace/pcb_grasp`  
自带 `.venv`、权重、标定拷贝与脚本，**不要**借用 `Perceptual_Processing_Platform` 或其它仓库的 Python 环境。

一套 live 流程：头粗定后按模型 cls 选抓取方式——  
- **小板**：悬停 → 腕精 XY(+rz) → 示教高度 → 收夹爪  
- **大板**：双臂到 `mid(P0,P3)` 同高水平夹线 → 对挤（不收夹爪）  

类别 id：`config/grasp_config.yaml` → `classes`（预留 `pcb_large: 2`）。

---

## 环境

```bash
cd /home/ubuntu/workspace/pcb_grasp
./.venv/bin/python -c "import ultralytics,torch,cv2,pyrealsense2; print('ok')"
```

入口脚本一律走本仓库 `.venv`：

```bash
./run_live.sh ...
./run_grasp.sh ...
./check_head_depth.sh ...
./run_ui.sh
```

---

## 实时开跑（现场）

```bash
cd /home/ubuntu/workspace/pcb_grasp

# dry-run：采真图、算位姿，不下发真机
./run_live.sh --vis output/live_vis

# 真机（小板单臂 / 大板自动双臂对挤）
./run_live.sh --real --vis output/live_vis

# 模型尚未含大板时，强制走大板几何（可用 dry-run 验算）
./run_live.sh --force-class large --vis output/live_vis
```

配置：`config/grasp_config.yaml`（`large_board.*` 控制外开距离、对挤行程、力阈）。

### 网页控制台（全流程）

```bash
./run_ui.sh
# http://127.0.0.1:8770  → 选阶段/类别/真机 → 运行一轮
```

---

## 头部深度检查

```bash
./check_head_depth.sh --left /path/cam0_left.jpg --right /path/cam1_right.jpg
# 或本仓库 data/demo 样例
./check_head_depth.sh --demo
```

---

## 真机（TJ_demo）

- `pcb_grasp/tj_motion.py` → TJ_demo Concise SDK  
- 笛卡尔阻抗 + TipFT 碰触急停  
- 速度：`motion.velocity_scale` / `cart_vel_mm_s`（默认很慢）
