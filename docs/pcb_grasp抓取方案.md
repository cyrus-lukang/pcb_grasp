# 感知定位与抓取整体方案

> 覆盖对象：**托盘** · **PCB 大板** · **PCB 小板** · **空槽 (nopcb)**  
> 共享底层：S36 头双目 portal（`10.3.8.7:8787`）· TJ 双臂 `192.168.1.190` · 笛卡尔阻抗 + TipFT  
> 更新日期：2026-11-12（小板悬停/抓取姿态按实测修订）

---

## 0. 总览

同一套产线感知与运控栈上，按目标类型走不同识别/定位/夹取分支：

| 目标 | 识别 | 定位 | 夹取 | 工程入口 |
|------|------|------|------|----------|
| **托盘** | 头分割 YOLO-Seg（tray 单类） | 双目深度 + CAD SE(3) → 基座 XYZθ | 双臂 tip 合拢，力阈接触停 | `wzx/dual_arm_center` + `tray_perception_web` |
| **PCB 大板** | 头 YOLO-Pose 3cls=`pcb_large(2)` | 头粗定角点 → 基座 mid(P0,P3)/法向 tip | 双臂平行直线插入，固定间距 + 阻抗停 | `pcb_grasp` UI `8770` |
| **PCB 小板** | 头 YOLO-Pose 3cls=`pcb_small(0)` | 头粗 + 腕精（Pose + shape） | 单臂 A 悬停→降 Z→收夹爪 | `pcb_grasp` UI `8770` |
| **空槽 nopcb** | 头 YOLO-Pose 3cls=`nopcb(1)` | 参与 board ID，**不抓** | — | 同 pcb_grasp 头识别 |

```
              头摄 portal S36 (cam0/1)
                       │
         ┌─────────────┴─────────────┐
         │                           │
   料盘分割 YOLO-Seg            PCB YOLO-Pose 3cls
   CAD SE(3) 6D                 小板 / 空槽 / 大板
         │                           │
  dual_arm_center              pcb_grasp pipeline
  tip 合拢 + 力停               小板收爪 / 大板对夹
         │                           │
         └──────────┬────────────────┘
                    │
              TJ A+B + 阻抗/TipFT
```

---

## 1. 共享感知与运控底座

### 1.1 头摄

| 项 | 说明 |
|----|------|
| 数据源 | rim watch portal `10.3.8.7:8787`（与 `capture_images.py` 同源） |
| 头双目 | 约 1920×1536 鱼眼；标定后立体深度（FFS / TensorRT） |
| 手眼 | `T_base_cam`（头左鱼眼 → Link_Base） |

### 1.2 机械臂

| 项 | 说明 |
|----|------|
| 控制器 | TJ / Marvin，`192.168.1.190` |
| 阻抗 | `cartesian_impedance`（底层） |
| 力保护 | TipFT；应用层另有合拢/对挤力阈 |

### 1.3 工程分工

| 仓库 | 职责 |
|------|------|
| `wzx/dual_arm_center` | 料盘双臂 tip 几何、合拢、控制页 `8765` |
| `wzx/tray_perception_web_20261102` | 料盘视觉 + 控制一体页 `8766`（委托 dual_arm_center UI） |
| `pcb_grasp` | PCB 大/小/空槽识别定位抓取控制台 `8770` |

---

## 2. 托盘：识别 · 定位 · 双臂夹取

### 2.1 入口

| 场景 | 命令 | 地址 |
|------|------|------|
| 纯控制（无视觉） | `cd ~/workspace/wzx/dual_arm_center && ./ui.sh` | http://127.0.0.1:8765 |
| **生产（视觉+控制）** | `cd ~/workspace/wzx/tray_perception_web_20261102 && bash start.sh` | 默认 **8766** |

操作流：**计算 →（选手选盘）→ 运动 → 抓取**；无弹窗确认，靠「先算后动」+ 急停。

### 2.2 识别

| 项 | 内容 |
|----|------|
| 模型 | 料盘实例分割 `tray_480`（YOLO-Seg / TensorRT engine） |
| 输入 | 头校正图，imgsz=480，conf≈0.5 |
| 后处理 | 形态学闭运算 + 最大外轮廓 |
| 类别 | **单类 tray**（不区分 PCB 大小板） |

### 2.3 定位（6D）

1. S36 双目 → FFS/TRT 深度  
2. 分割 mask + 深度 + CAD（`600×500` 料盘 PLY）→ SE(3)  
3. 方法可选：轮廓增强稳定性校验（生产默认）/ G6D / 混合  
4. 输出到 **Link_Base**：
   - **XYZ**：CAD 原点（mm）  
   - **θ / yaw**：CAD +Y 夹持轴相对基座 +Y 的平面角（与 dual_arm_center yaw 一致）

### 2.4 夹取几何与动作

输入：`XyzYaw = (x, y, z, yaw)`，xyz 为工具 tip。

| 规则 | 定义 |
|------|------|
| tip 间距 | `distance_mm`（默认 700）水平对置 |
| 姿态 | 工具竖直；左法兰 RPY≈`(-180,0,yaw-90)`，右≈`(180,0,yaw+90)` |
| 高度 | `Z_target ≈ Z_tray + tool_z_mm`（`tool_z_mm` 默认 270） |
| 接近 | 双臂 PTP 到目标（左右短时错峰下发） |
| 抓取 | 沿连线向中心合拢，总行程 `grasp_close_mm`（默认 100），每臂一半 |
| 力停 | `grasp_force_n`（默认 5N）：沿靠近方向阻力超阈连续帧 → 该臂软停，对侧可补走 |

主配置：`dual_arm_center/config/default.json`  
（`distance_mm` / `tool_y_mm` / `tool_z_mm` / `grasp_close_mm` / `grasp_force_n` / `velocity_scale` …）

### 2.5 安全要点

- 视觉不自动下发运动；需人工点「运动」「抓取」  
- IK 预检失败则禁用运动  
- 急停：双臂抱闸，需另行 RESET  

---

## 3. PCB 大板：识别 · 定位 · 双臂对夹

工程：`pcb_grasp`（`./run_ui.sh` → http://127.0.0.1:8770）

### 3.1 识别

- 头 YOLO-Pose 3cls，`pcb_large = 2`  
- 权重：`yolopose_3cls_…_041ee5`（小板+空槽+大板）  
- `large_board.enable=true` 时有大板则优先抓大板  

### 3.2 定位

- 头多帧 + 立体/托盘平面 → 四角基座坐标 P0…P3  
- **mid(P0,P3) = 基座 XY 中点**（不是像素中点）→ tipL  
- tipR = tipL + span × 板内法向（span 优先标定 `tip_span_m`）  
- 姿态：两臂同一 `rz = baseYAngleToGripperRz(P0→P3)+90°`（平行），沿法向直线插入 03 / 对边  

### 3.3 夹取

1. UI「识别抓取点」：叠加 tipL/tipR，打印 P0/P3/tip XY（不动臂）  
2. 「执行抓取」：大板默认确认 tip → `move_both(approach)` → `squeeze`  
3. **不收夹爪**；固定 tip 高度 `grasp_z_m`、固定间距 `tip_span_m`（标定后）  
4. 阻抗 + TipFT：`squeeze_force_n` 遇阻停；单侧触停另一侧补走  

| 配置键 | 状态 | 含义 |
|--------|------|------|
| `large_board.grasp_z_m` | **待标定** | 固定 tip 高度 |
| `large_board.tip_span_m` | **待标定** | 两 tip 固定间距（防挤板） |
| `squeeze_close_m` | 0 | 标定 tip_span 后不再内挤 |
| `squeeze_force_n` | 5.0 | 遇阻停力阈 |
| `use_impedance` | true | 笛卡尔阻抗 |

---

## 4. PCB 小板：识别 · 定位 · 单臂收夹

> 细节页：[PCB 小板视觉定位与抓取方案](https://airimbot.feishu.cn/wiki/ZGraw4rF0iah7Vk7ytBccEY3njg) · 本地 `docs/PCB小板视觉定位与抓取方案.md`

### 4.1 识别

- 头 YOLO-Pose，`pcb_small = 0`  
- Board ID：基座 XY 距原点排序；默认抓 `target_board_id`（跳过 nopcb）  
- 腕：Gemini305 portal + YOLO-Pose `3a699e` + 外轮廓 shape_match  

### 4.2 定位与姿态分工

1. **头粗定**：双目 FFS → 基座位姿；多帧中值；**不**把板边 rz 用于悬停  
2. **接近（camera-down）**：腕相机光轴竖直向下，光心约板上方 `hover_height_m=0.30 m`，**keep 当前 rz**  
3. **腕精定**：仍 camera-down；Pose + shape_match（P1→P0）→ 板心 XY / rz；与粗 ID 匹配；rz unwrap  
4. **抓取姿态**：法兰 XY = 板心，`rpy=(-90°,0,rz)`（末端竖直向下），悬停高度翻姿态后再降 Z  

### 4.3 夹取

- 抓前张开 → 板心 + (-90,0,rz) → 降到 `grasp_z_m + grasp_z_lift_m`（示教 tip ≈ 0.70548 m，**+10 mm**）→ 知行收夹 **0.4**  
- 单臂 A；`stage=coarse` 只做到悬停；复位位 XYZ 当前 + 固定 rpy=(90°,0°,0°)  

---

## 5. 空槽 (nopcb)：识别 · 定位 · 不抓

| 项 | 说明 |
|----|------|
| 识别 | 头 YOLO-Pose `nopcb = 1` |
| 定位 | 与小板同一套头定位，进入 board ID 列表 |
| 抓取 | **明确不抓**；选板逻辑跳过 nopcb |
| 用途 | 空槽负样本/占位，避免误抓空位；支撑「有板才抓」 |

---

## 6. 对比一览

| 维度 | 托盘 | 大板 | 小板 | 空槽 |
|------|------|------|------|------|
| 头模 | Seg tray | Pose 3cls=2 | Pose 3cls=0 | Pose 3cls=1 |
| 6D/位姿 | CAD SE(3) XYZθ | 四角 → tipL/R | 头粗+腕精 | 仅定位 |
| 臂数 | 双臂 | 双臂 | 单臂 A | — |
| tip 规则 | 中心±半距，yaw 定向 | mid(P0,P3)+法向 span | cam-down 悬停→板心(-90,0,rz)→tip Z+lift | — |
| 终端动作 | 合拢 tip | 平行插入对夹 | 知行收夹 0.4 | 无 |
| 力控 | `grasp_force_n` 软停 | `squeeze_force_n` 阻抗停 | 运动中 TipFT | — |
| UI | 8765 / **8766** | **8770** | **8770** | 8770 识别 |

---

## 7. 现场操作速查

### 托盘

```bash
cd ~/workspace/wzx/tray_perception_web_20261102 && bash start.sh
# 浏览器 → 计算 → 运动 → 抓取
```

或纯控制：

```bash
cd ~/workspace/wzx/dual_arm_center && ./ui.sh   # :8765
```

### PCB（大/小/空槽）

```bash
cd ~/workspace/pcb_grasp && ./run_ui.sh          # :8770
# 连相机 → 识别抓取点 →（连臂）→ 执行抓取
```

---

## 8. 标定与待办

### 已就绪

- 头双目 portal、手眼、托盘 CAD/分割链路  
- 料盘 tip 间距/高度/合拢力（`default.json`）  
- PCB 头 3cls、小板示教 Z、腕精链路  
- 大板几何 + UI 确认 + 阻抗代码  

### 待标定（大板）

- `large_board.grasp_z_m`：固定 tip 高度  
- `large_board.tip_span_m`：固定两 tip 间距（防挤板）  
- 视情况微调 `squeeze_force_n`  

---

## 9. 代码索引

| 模块 | 路径 |
|------|------|
| 料盘 tip 几何 | `wzx/dual_arm_center/include/center_pose.hpp` |
| 料盘合拢/力停 | `wzx/dual_arm_center/src/move_main.cpp` |
| 料盘 UI | `wzx/dual_arm_center/ui/` · `./ui.sh` → `:8765` |
| 料盘视觉+UI | `wzx/tray_perception_web_20261102/` → `:8766` |
| PCB UI | `pcb_grasp/run_ui.sh` · `ui/server.py` → `:8770` |
| PCB 主流程 | `pcb_grasp/pcb_grasp/pipeline.py` |
| PCB 大板几何 | `pcb_grasp/pcb_grasp/large_board.py` |
| PCB 配置 | `pcb_grasp/config/grasp_config.yaml` |

---

## 10. 一句话摘要

**托盘**用头分割 + CAD 6D，双臂按 yaw 对置 tip 合拢并力停；**PCB**用头 Pose 三分类——空槽只识别不抓；小板为 camera-down 粗悬停 → 腕精 → 板心 (-90,0,rz) → tip Z+lift → 知行收夹；大板按基座 mid(P0,P3) 双臂平行插入，标定高度/间距后阻抗遇阻停。共享头摄 portal 与运控栈，业务分属 `dual_arm_center` / `tray_perception_web` 与 `pcb_grasp`。
