# PCB 小板定位与抓取方案（现场实测版）

> 日期：2026-11-12  
> 工程：`pcb_grasp` · UI `http://127.0.0.1:8770` · 配置 `config/grasp_config.yaml`  
> 运控：RIM `move_to_pose` · 左臂 A · 知行夹爪 RS485  
> 说明：按当前真机抓取测试定稿；与早期「工具朝下悬停 / 法兰示教 −227 mm」方案区分开。

---

## 1. 方案目标

完成 PCB 小板：选板 → 悬停观测 → 板心与朝向精修 → 末端竖直对齐 → 下降收夹。

两阶段视觉 + 两套末端姿态：

| 阶段 | 末端姿态 | 作用 |
|------|----------|------|
| 粗定位悬停 | **腕相机光轴竖直向下**（camera-down），**不转板边 rz** | 把头定目标送入腕相机视野 |
| 腕部精定位 | 仍 camera-down | 关键点 + 外轮廓匹配 → 基座板心 XY、rz |
| 抓取 | **末端竖直向下** `rpy=(-90°, 0, rz)`，XY=板心 | 同高翻姿态并对齐板心，再降 tip Z 收夹 |

夹取高度 = 现场示教 tip Z（`arm_*_linkTCP`）+ 防压板上抬。换板型 / 夹爪安装 / 承载高度后，重核模板、手眼、示教高度与夹爪开度。

---

## 2. 系统组成

| 模块 | 当前选型 |
|------|----------|
| 头相机 | S36 头双目，portal / frame_bundle `10.3.8.7` |
| 头深度 | Fast-FoundationStereo（FFS） |
| 头模型 | YOLO-Pose 3cls `041ee5`（小板=0 / nopcb=1 / 大板=2） |
| 腕相机 | 左腕 Gemini305（portal cam13 + depth12） |
| 腕模型 | YOLO-Pose `3a699e`，imgsz=480 |
| 精定位 | 外轮廓 Chamfer（`shape_match`，模板 `d435_pcb.npz`） |
| 手眼 | 头 `T_base_cam`（20261109）；腕 `T_tool_cam`（20261110 左 305） |
| 机械臂 | TJ / Marvin `192.168.1.190`，运控后端 RIM |
| TCP | tip = `linkTCP`，`tcp_offset=[0,0,0]` |
| 夹爪 | 知行 RS485，`grasp_position=0.4`，`release_position=1.0` |

---

## 3. 整体流程

```
头双目采集 → YOLO-Pose 粗定（5 帧中值 + board ID）
        ↓
粗定位悬停：camera-down，相机光心 ≈ 板上方 0.30 m，keep 当前 rz
        ↓
静止 → 腕 5 帧精定（Pose + shape_match，匹配粗 ID）
        ↓
张开夹爪 → 板心 XY + 末端 (-90°, 0, rz)（同高翻姿态）
        ↓
降到示教 tip Z + 10 mm → 静止 → 收夹 0.4 → 停止
```

UI 操作：

- `识别抓取点`：plan_only，只标目标、不动臂  
- `执行抓取`：全流程  
- `stage=coarse`：只做到 camera-down 悬停  

---

## 4. 定位与抓取分步

### 4.1 静止与采图

到位稳定后再采图。

| 参数 | 值 |
|------|-----|
| `settle_s` | 1.0 s |
| `still_pos_tol_m` | 1.5 mm |
| `still_rot_tol_rad` | 0.015 rad |
| `still_hold_s` | 0.15 s |
| `still_timeout_s` | 3.0 s |
| `n_frames` | 5 |
| `outlier_mad_k` | 3.0 |

### 4.2 头部粗定位

1. 校正图 + YOLO-Pose 3cls；抓取只选小板（跳过 nopcb）。  
2. FFS 深度 + `T_base_cam` → 基座中心 / 四角 / rz。  
3. 5 帧 MAD 剔野后取中值。  
4. 多板按基座 XY 距原点赋 ID（1=最近）；优先 `target_board_id`。  

**悬停阶段不下发板边 rz**，避免拧 yaw 把目标转出腕视野。

### 4.3 粗定位悬停（camera-down + keep rz）

**不是 tip 竖直朝下。** 305 光轴相对工具侧偏：相机朝下时 tip 近水平；抓取再翻末端竖直。

| 项 | 规则 |
|----|------|
| 姿态 | 腕相机光轴竖直向下（`hover_camera_down=true`） |
| 高度 | 相机光心 ≈ 板上方 `hover_height_m=0.30 m` |
| rz | 锁定接近前当前法兰 rz（`keep_rz`） |
| 夹爪 | 粗定位段不动 |
| 到位 | \|err_xy\| ≤ 80 mm，否则中止 |

实现：`flange_above_camera_down` → `move_line`（完整 rpy，不强制 tip-down）。

### 4.4 腕部精定位

悬停静止后：

1. 腕 YOLO-Pose → 框 + 关键点。  
2. 外轮廓匹配：关键点 **P1→P0** 定初向，在 `±angle_range` 内微调中心与角；`score_thresh=5.2`。  
3. 像素中心 + 深度 → 腕相机 → `T_tool_cam` → 采图 EE → 基座 `ObjectPose`。  
4. 与粗板按 XY 匹配 ID（`board_match_max_dist_m=0.12`）；匹配不到则回退腕侧实际检出板。  
5. rz 相对粗定位 `nearest_rz` unwrap，避免 ±180° 跳变。  

### 4.5 抓取位姿生成

转换：

**匹配中心像素 + 深度 → 腕相机三维 → 工具 → 基座板心 (fine.x, fine.y) + rz**

| 量 | 规则 |
|----|------|
| XY | **tip XY = 板心**（`tcp_offset=0`）。禁止「camera-down 法兰 + (fine−coarse)」增量——手眼偏置会偏心 |
| rpy | **(-90°, 0, rz)**；rz = 腕精 + `tool_rz_offset_rad=90°`，并对粗定位 unwrap |
| Z | 先同高完成板心 + 末端朝下；再降到 `grasp_z_m + grasp_z_lift_m` |

| 配置 | 值 |
|------|-----|
| `grasp_z_m` | 0.70548 m（tip 贴板示教） |
| `grasp_z_lift_m` | 0.01 m（+10 mm 防压板） |
| 实际下降 Z | ≈ 0.71548 m |

复位位（UI「记为复位位」）：只记当前 XYZ，姿态固定 **rpy=(90°, 0°, 0°)**。

### 4.6 抓取执行

1. 张开夹爪（`release=1.0`）。  
2. `FINE_ORIENT`：同高 → 板心 XY + `(-90°, 0, rz)`。  
3. `GRASP_Z`：保持姿态，降到 tip Z + lift。  
4. 静止后收夹 `0.4`（知行：0=全闭，1=全开）。  
5. 停止。本阶段不做提起 / 搬运 / 放置。  

夹爪：速度 50、抓取力矩 30、松开力矩 60、com_ch=2。粗接近失败或腕精失败则中止，不盲目下降。

---

## 5. 关键约定

| 项 | 约定 |
|----|------|
| 关键点方向 | **P1 → P0** |
| rz | `baseYAngleToGripperRz` + `tool_rz_offset_rad = π/2` |
| 粗悬停 rz | 不应用板边角；抓取段才下发腕精 rz |
| 旋转顺序 | R = Rz @ Ry @ Rx（与 rimbot 一致） |

---

## 6. 与早期方案差异（测试结论）

| 早期方案 | 当前实测定稿 |
|----------|--------------|
| 悬停「工具朝下」+ Y/Z 额外让位 | **camera-down 悬停**，无额外 Y/Z 让位 |
| 法兰示教 Z = −227.26 mm | tip 示教 **0.70548 m**，再 **+10 mm** |
| 悬停按法兰 Z≈+450 mm | 按 **相机光心上方 0.30 m** |
| 腕 D405 | **Gemini305** portal |
| 精定位后只调 XY/rz 再降 | 先 **板心 + (-90,0,rz)**，再降 Z |
| 粗定位转板边 yaw | **keep_rz**，不转板边 |
| 接近阶段收/合夹爪 | 接近不动爪；抓前张开，到位后 0.4 收夹 |
| n_frames=10 / settle=0.6 s | **5 帧 / settle 1.0 s** |

---

## 7. 验收关注点

- 悬停后腕图是否完整看到目标板（camera-down / keep_rz）。  
- 板心对齐后 tip 相对板的偏心（mm）。  
- 粗 / 精 rz 差与 unwrap 后是否仍有 ±180° 歧义。  
- 收夹是否压板、打滑、夹空。  
- 多板位 / 多角度 / 多光照重复成功率。  

平移、旋转验收阈值由板尺寸、允许夹持区与夹爪容差确定；粗精差值只作修正量观察。

---

## 8. 后续

1. 固定当前配置为基线，记多轮抓取数据。  
2. 扩展提起、搬运、放置并验证持物稳定。  
3. 大板双臂对夹见总方案；本页仅小板单臂。  

代码入口：`pcb_grasp/pipeline.py`（`run_live`）· `geometry.py` · `shape_refine.py` · `rim_motion.py`  
配置：`config/grasp_config.yaml`
