# PCB 小板视觉定位与抓取方案

> 更新日期：2026-11-12（按现场抓取测试修订）  
> 工程：`pcb_grasp` · UI `:8770` · 配置 `config/grasp_config.yaml`  
> 运控：RIM `move_to_pose` · 左臂 A · 知行夹爪 RS485

## 方案目标

针对 PCB 小板抓取，采用「头部双目粗定位 + 腕部相机精定位」两阶段方案，完成选板、悬停观测、板心与朝向修正、末端竖直朝下对齐、下降与收夹。

核心分工：

| 阶段 | 姿态 | 作用 |
|------|------|------|
| 粗定位悬停 | **腕相机光轴竖直向下**（camera-down），**不转板边 rz** | 把头定的板送入腕相机视野，保证精定位成像 |
| 腕部精定位 | 仍保持 camera-down | 关键点定初向 + 外轮廓匹配 → 基座板心 XY、rz |
| 抓取 | **末端竖直向下** `rpy = (-90°, 0, rz)`，XY 对准板心 | 在悬停高度翻姿态并对齐板心，再降到示教 tip Z 收夹 |

夹取高度用现场示教 tip Z（`arm_*_linkTCP`），再加防压板上抬量。更换板型、夹爪安装或承载高度后，需重核模板、手眼、示教高度与夹爪开度。

## 系统组成

<sheet sheet-id="gXq4Jy" token="EJHhsIU2ph7h7Mt5jOgcO1mjnLc"></sheet>

补充（相对旧表）：

| 项 | 当前 |
|----|------|
| 腕相机 | 左腕 Gemini305（portal cam13 + depth12），非旧 D405 |
| 头模 | YOLO-Pose 3cls `041ee5`（小板=0 / nopcb=1 / 大板=2） |
| 腕模 | YOLO-Pose `3a699e` imgsz=480 |
| 运控 | RIM；夹爪走知行 RS485（`grasp_position=0.4`） |
| TCP | tip = `linkTCP`，`tcp_offset=0` |

## 整体流程

```
头双目采集 → YOLO-Pose 粗定（多帧中值 + board ID）
        ↓
粗定位悬停：camera-down，相机光心 ≈ 板上方 hover_height，keep 当前 rz
        ↓
静止 → 腕多帧精定（Pose + shape_match，匹配粗 ID）
        ↓
张开夹爪 → 板心 XY + 末端 (-90°,0,rz)（同高翻姿态）
        ↓
降到示教 tip Z + lift → 静止 → 收夹 0.4 → 停止
```

UI：`识别抓取点`（plan_only，不动臂）→ `执行抓取`；`stage=coarse` 只做到悬停。

### 3.1 静止检测与图像采集

机械臂到位并稳定后再采图，避免运动中的图像与位姿不一致。

当前配置（`motion` / `detect`）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `settle_s` | **1.0 s** | 到位后再识别 / 收夹 |
| `still_pos_tol_m` | 1.5 mm | 位置静止判据 |
| `still_rot_tol_rad` | 0.015 rad | 旋转静止判据 |
| `still_hold_s` | 0.15 s | 连续静止保持 |
| `still_timeout_s` | 3.0 s | 等待超时 |
| `n_frames` | **5** | 头 / 腕多帧 |
| `outlier_mad_k` | 3.0 | MAD 剔野 |

以上容差只判断臂是否停稳，不代表视觉定位精度。

### 3.2 头部粗定位

1. 头鱼眼校正图 + YOLO-Pose 3cls，保留小板 / 空槽 / 大板；抓取只选小板（或按 `force_class`）。
2. FFS 立体深度 + `T_base_cam` → 基座四角 / 中心 / rz。
3. 连续 **5** 帧，MAD 剔野后取中值。
4. 多板按基座 XY 距原点赋 board ID（1=最近）；优先抓 `target_board_id`。
5. 空槽 `nopcb` 参与 ID、**不抓**。

粗定位输出悬停用的板心 XY 与高度参考；**悬停阶段不把板边 rz 下发到法兰**（避免为对齐板边拧 yaw、挤出腕相机视野）。

### 3.3 粗定位悬停（camera-down，keep rz）

**不是 tip 竖直朝下。** 305 光轴相对工具有侧向偏置：相机朝下时 tip 近水平；抓取时再翻到末端竖直。

实现：`flange_above_camera_down` + `move_line`（`keep_orientation=False`，完整 rpy）。

| 项 | 当前 |
|----|------|
| 姿态 | 腕相机光轴竖直向下（`hover_camera_down=true`） |
| 高度 | 相机光心 ≈ 板上方 `hover_height_m = 0.30 m` |
| rz | **锁定接近前当前法兰 rz**（`keep_rz`）；不按板边转 yaw |
| 夹爪 | 粗定位段**不动作**（开/合只在最终抓取段） |
| 到位判据 | 悬停 XY 误差 \|err_xy\| ≤ 80 mm，否则中止 |

旧方案「工具朝下 + Y/Z 额外让位、法兰 Z = 粗 Z+450 mm、示教法兰 −227 mm」已废弃：TCP 已改为 tip，悬停按相机光心高度描述。

### 3.4 腕部精定位

悬停静止后，腕 RGB-D 多帧：

1. 腕 YOLO-Pose → 框 + 关键点。
2. 外轮廓 Chamfer 匹配（`shape_match`，模板 `d435_pcb.npz`）：关键点 **P1→P0** 定初向，在 `±angle_range` 内微调中心与角。
3. 像素中心 + 深度 → 腕相机 → `T_tool_cam` → 采图时 EE → 基座 `ObjectPose`。
4. 与粗定位板按 XY 距离匹配 ID（`board_match_max_dist_m=0.12`）；匹配不到目标 ID 时回退到腕侧实际认出的板。
5. rz 相对粗定位做 `nearest_rz` 展开，避免 ±180° 跳变。

<sheet sheet-id="T7eZh6" token="EJHhsIU2ph7h7Mt5jOgcO1mjnLc"></sheet>

`score_thresh=5.2` 为现场经验阈值；分数越小轮廓越贴。弱光 / 反光时结合可视化与多帧稳定性判断。

### 3.5 坐标转换与抓取位姿

转换链路：

**匹配中心像素 + 深度 → 腕相机三维 → 工具 → 基座板心 (fine.x, fine.y) + rz。**

抓取法兰指令（实测定稿）：

| 量 | 规则 |
|----|------|
| XY | **tip XY = 板心**（`fine.x/y`，`tcp_offset=0`）。不要用「camera-down 法兰 + (fine−coarse)」做增量——手眼偏置会让夹爪偏心 |
| rpy | **(-90°, 0, rz)**：末端竖直向下；rz 来自腕精 + `tool_rz_offset_rad=90°`，并相对粗定位 unwrap |
| Z | 先在悬停高度完成「板心 XY + 末端朝下」；再降到 `grasp_z_m + grasp_z_lift_m` |

当前示教与抬升：

| 配置 | 值 | 说明 |
|------|-----|------|
| `grasp_z_m` | **0.70548 m** | tip（linkTCP）贴板示教高度 |
| `grasp_z_lift_m` | **0.01 m（+10 mm）** | 防压板上抬后再抓 |
| 实际下降 Z | ≈ **0.71548 m** | taught + lift |

复位位（UI「记为复位位」）：只记当前 XYZ，姿态固定 **rpy = (90°, 0°, 0°)**。

### 3.6 抓取执行

1. 腕精完成后，若 `do_grasp`：**先张开**夹爪（`release_position=1.0`）。
2. `FINE_ORIENT`：同高移到板心 XY，姿态切到 `(-90°, 0, rz)`（`move_tool_down_hover` / tip-down 路径）。
3. `GRASP_Z`：保持该姿态，降到示教 tip Z + lift。
4. 静止后收夹：`grasp_position=0.4`（知行：0=全闭，1=全开；0.4 略留缝夹板）。
5. 停止；本阶段不做提起 / 搬运 / 放置。

夹爪参数：速度 50、抓取力矩 30、松开力矩 60、com_ch=2。接口参数不等于物理开口毫米或实测夹持力。

粗定位接近失败或腕精失败时中止，不盲目下降。

## 关键配置

<sheet sheet-id="WDaHEz" token="EJHhsIU2ph7h7Mt5jOgcO1mjnLc"></sheet>

角度约定（与代码一致）：

- 头 / 腕关键点向量：**P1 → P0** 定板边方向。
- rz：`baseYAngleToGripperRz` + `tool_rz_offset_rad = π/2`。
- 粗悬停 **不应用** 板边 rz；抓取段才下发腕精 rz。

手眼：

- 头：`T_base_cam`（20261109 头左鱼眼 → Link_Base）。
- 腕：左腕 `T_tool_cam`（20261110 Gemini305）。

## 相对旧方案的测试结论

| 旧描述（2026-11-02 / 11-09 文档） | 当前实测结论 |
|----------------------------------|--------------|
| 悬停「工具朝下」+ Y+50 / Z+100 让位 | 改为 **camera-down 悬停**；无额外 Y/Z 让位偏置 |
| 法兰示教 Z = −227.26 mm | TCP 改为 tip；示教 **tip Z = 0.70548 m**，再 **+10 mm** 抓 |
| 悬停高度按法兰 Z=+450 mm | 按 **相机光心上方 0.30 m** |
| 腕 D405 | **Gemini305** portal |
| 精定位后仅调 XY/rz 再降 | 先 **板心 + (-90,0,rz)**，再降 Z |
| 粗定位转板边 yaw | **粗定位 keep_rz**，不转板边 |
| 夹爪全程或接近时闭合 | 接近不动爪；抓取前 **张开**，到位后 **0.4 收夹** |
| n_frames=10 / settle=0.6 s | **n_frames=5 / settle=1.0 s** |

## 验证结果与评价指标

早期报告 001～008 验证了「腕精 → 降 Z → 收夹」链路。2026-11 现场迭代后，需按**新姿态分工**重新记轮次：camera-down 悬停 → 腕精 → (-90,0,rz) 板心对齐 → tip Z+lift → 知行 0.4。

建议指标：

<sheet sheet-id="wQUsFR" token="EJHhsIU2ph7h7Mt5jOgcO1mjnLc"></sheet>

补充关注：

- 悬停后腕图是否完整看到目标板（camera-down / keep_rz 是否有效）。
- 板心对齐后 tip 相对板边的偏心（mm）。
- 粗 / 精 rz 差与 unwrap 后是否仍存在 ±180° 歧义。
- 收夹后是否压板、打滑、夹空。

平移 / 旋转验收阈值由板尺寸、允许夹持区与夹爪容差确定。粗精差值只作修正量观察；绝对精度需独立参考测量。

## 后续实施安排

1. 以当前 `grasp_config.yaml` 为基线，固定手眼、模板方向、示教 tip Z 与 lift。
2. 多板位 / 多角度 / 多光照重复抓取，统计偏心、rz 误差与失败模式。
3. 再扩展提起、搬运、放置，并验证持物稳定性。
4. 大板双臂对夹另见总方案；本页仅小板单臂。

资料依据：

- 《PCB 粗定位 + 精定位抓取实验报告（2026-11-02）》  
- 工程实现：`pcb_grasp/pipeline.py`（`run_live`）· `geometry.py` · `shape_refine.py` · `rim_motion.py`  
- 配置：`config/grasp_config.yaml`
