# 双目鱼眼手部骨骼 Pipeline v2 迭代实施方案

文档状态：`ACTIVE`

首次编写：2026-08-14

适用范围：本仓库本地开发、H20 兼容 worker、Trace API 与 React 检查前端
基线代码：`5eacb7a` 及其 H20 实跑 `h20-stage-video-5eacb7a`
当前候选代码：`8e49061`；H20 package smoke `v2-8e49061-h20-smoke1`；
120-pair canonical run `v2-8e49061-h20-120`

本文是后续算法迭代的实施依据。代码实现若因真实数据、依赖能力或性能约束而改变本文
方案，必须在同一个提交中更新“决策与偏离记录”和文末变更日志；不得只修改代码、不
更新方案。

## 1. 目标与边界

目标是在不破坏现有同步、标定、可追溯和 `fhp21/v1` 输出契约的前提下，将当前兼容性
baseline 迭代为：

```text
原始鱼眼图上的手候选
  → hand-centred virtual perspective crop
  → crop 域 RTMPose 21 点与不确定度
  → 鲁棒跨视角关联与 metric 3D fusion
  → 稳定的 sequence track
  → coarse-to-fine frame-wise MANO
  → track 级 Temporal MANO
  → Raw / Frame-wise MANO / Temporal MANO 三层不可覆盖输出
```

本轮不以“换更大的 2D 模型”作为第一手段。先固定现有 RTMDet 与 RTMPose-m 权重，隔离
验证鱼眼输入适配、几何融合和优化器本身的收益。只有冻结评测证明 2D 模型仍是主要瓶颈
时，才另立模型替换决策。

以下事项继续保持不变：

- 左右视频必须按硬件时间戳配对，不按帧号直接 zip；
- 全帧 undistortion/rectification 是标定 QA 和可视化支路，不冒充模型输入；
- Raw 3D 是不可变观测，MANO 和时序结果不得覆盖 Raw；
- 无可靠 metric 证据的 Raw 点保持 invalid，不能由 prior 伪装成测量；
- `fhp21/v1`、米制、坐标系、mapping、模型和标定 provenance 必须保留；
- MANO 私有文件仍由用户提供并校验，不下载、不提交、不打包；
- 不通过放宽 MANO 20 mm 门禁来掩盖上游或拟合问题。

## 2. 当前真实基线

### 2.1 已实跑 H20 基线执行链

提交 `5eacb7a` 在 H20 封存 run 中的真实链路是：

```text
native distorted fisheye frame
  → RTMDet（每视角按 0.30 过滤后提前截断 top-2）
  → RTMPose top-down（仍在原始鱼眼 frame+bbox 上推理）
  → 将 2D 点映射到全帧 rectified pixel
  → median rectified-y cost 的最多 2×2 关联
  → 每关节 OpenCV 线性 triangulation + 硬 gate
  → wrist，否则所有有效点质心的 3D tracker
  → 每帧 MANO：45D 零姿态、零 beta、40 次 Adam、等权 3D L2
  → MANO 失败时 Raw 3D；成功时 MANO 3D
  → timestamp-aware XYZ EMA
  → fhp21.jsonl 与检查视频
```

`cv2.fisheye` undistortion 和 stereo rectification 图目前只用于 DEBUG/QA。它们已经保存到
trace，但**没有进入 detector 或 RTMPose 的实际输入链**。当前 Temporal 阶段是
`causal_time_ema_v1`，不是 Temporal MANO Optimization。

### 2.2 `5eacb7a` 历史 baseline 的 120-pair 证据

以下数字只描述当前 session 的前 120/176 个同步 pair，不应外推为总体精度：

- 左目 116/120 帧、右目 117/120 帧检测到两只手；
- 113/120 帧得到两个跨视角 match，7 帧只有一个 match；
- 输出 233 个 Raw hand-frame，共 4,634/4,893 个有效 3D joint；
- 只有 73/233 个 Raw skeleton 具有完整 21 点；
- 有效骨边 median 33.76 mm、P95 105.61 mm、max 214.81 mm，同一 track 存在明显跳变；
- tracker 产生 `track-0000`、`track-0001` 和 4 帧短轨 `track-0002`；
- 233 个 hand-frame 均执行 MANO，466 个左右手假设均非运行错误，但全部被 20 mm 门禁
  拒绝；RMSE min/median/max 为 24.59/38.89/85.87 mm；
- 同一代表帧将迭代从 40 提高至 200/500 后，最佳 RMSE 从 29.33 mm 降到
  9.55/7.78 mm，证明当前 fitter 明显欠收敛；
- 最终 233/233 temporal 输入都是 `RAW_FUSION`，因此当前 stable 输出只是 Raw XYZ EMA。

详细证据和限制见 [当前 Pipeline 问题分析](current-pipeline-problem-analysis.md)。没有目标域
GT 时，低重投影误差、MANO 自拟合 RMSE 和“视频看起来更平滑”都不能替代准确率。

### 2.3 基线测试、本地迭代测试与缺口

`5eacb7a` 基线的 worker 契约测试已经覆盖：0/1/2 candidate、unmatched、最大匹配数优先、逐点
三角化失败原因、追踪 gap reset、MANO 文件校验与 FHP21 映射、左右手选择与 beta 冻结、
Temporal 时间戳语义、失败回退、Trace parent、全轨迹 overlay video 和大文件流式导入。

候选提交 `8e49061` 的独立测试覆盖：KB4 virtual crop/ray round-trip/valid mask/左右相机 rig
transform、virtual crop pose adapter 的 0-hand/cardinality/fail-closed、seed/recovery/top-4
候选分类、Huber IRLS stereo fusion/深度与掌心 gate/启发式 covariance、固定掌心 anchor/
constant-velocity/TTL，以及“仅 accepted MANO 参数可成为下一帧 warm-start”的控制器 seam。
该提交已在真实 H20/OpenMMLab/MANO 环境完成 1-pair smoke 和 120-pair canonical
run，证明主链、package validator 与产物导入可运行。这仍不等于已有 GT 的精度
验收或所有 phase gate 通过；逐切片状态见 8.3，实跑证据见 8.5–8.7。

仍无覆盖或尚未接通的关键项包括：

- crop resize/letterbox/mirror 的完整逆变换；
- SimCC covariance 提取或目标域 score calibration；
- top-4 recovery 已经真实 H20 主链运行，但仍无目标域候选/ghost/duplicate 标注验收；
- 极线一致的错手/错点配对、track-aware association 和 anatomical gate；
- covariance-aware tracking gate、完整 cost matrix、scene/calibration reset；
- PCA coarse → 45D fine 参数转换、锚点选择、双向 warm-start；
- 逐关节 MANO residual、2D 重投影 loss 和收敛曲线；
- MANO wrist/tip 分组 residual、每 hand-frame 耗时和 peak GPU memory 记录；
- MANO 参数空间的滑窗/离线时序优化。

### 2.4 `8e49061` 候选链路对比

下表保留历史 baseline、首次失败包的诊断证据和修正后 canonical run 的不同
身份。`v2-0782688-h20-120` 不是成功 baseline；其数字仅用于解释现场失败。

| 指标 | `5eacb7a` 历史 baseline | `0782688` FAILED 诊断 | `8e49061` canonical |
|---|---:|---:|---:|
| 120 帧中两个跨视图 match | 113 | 120 | 120 |
| Raw produced / hand-gate rejected | 233 / 旧链无该门禁 | 238 / 2 | 238 / 2 |
| Raw valid joints | 4,634 / 4,893 | 诊断包未登记 canonical baseline | 4,764 / 4,998 |
| track 数 | 3 | 2 | 2 |
| MANO produced | 0 / 233 | 213 / 238 | 213 / 238 |
| accepted MANO RMSE median / P95 | 无 accepted fit | 9.008 / 16.960 mm | 9.008 / 16.960 mm |
| export | 233 | 238 条位于失败包中 | 238 |
| 最终 package | `COMPLETED` | `FAILED / NOT_PRODUCED` | `COMPLETED / PRODUCED` |

## 3. 公开 seams 与实施约束

### 3.1 保持稳定的公开边界

| 边界 | 已有定义 | 本轮使用方式 |
|---|---|---|
| 模型与几何类型 | `src/fisheye_handpose/contracts.py` | 保持 `ImageView → Detection2D → PerspectiveCrop → CanonicalViewEvidence → SpatialObservation → PoseEstimate` 语义 |
| 组件接口 | `Detector`、`VirtualCropper`、`PoseEvidenceBackend`、`CrossViewAssociator`、`FusionBackend`、`KinematicRefiner`、`TemporalRefiner` | 新模块按这些职责拆分，不把全部算法继续堆入 runner |
| 进程边界 | `PipelineStageExecutor` 与 H20 JSON bridge | Python 3.11 core 不导入旧 OpenMMLab 环境；H20 worker 继续独立进程运行 |
| 输出 | `fhp21/v1` | landmark 顺序、状态分解、米制和 Raw 不可变语义不变 |
| Trace | `trace-record/v1` 的固定 stage vocabulary 与 DAG | 新事件复用现有 stage；添加字段和 blob role，不伪造已执行阶段 |
| 前端 | Trace API 的 `run_key/frame_key` 与只读 artifact API | 前端只消费 trace，不直接依赖 worker 内部 Python 类型 |

核心包已经定义了上述模型中立接口，但当前 H20 worker 为兼容 Python 3.10 和隔离依赖，
实际仍使用独立 dict/JSON DTO，并未实例化核心 Protocol。第一轮不跨环境直接 import 核心
包；worker 内部类型必须与核心语义对齐，并由 bridge/contract tests 验证 JSON 边界。

### 3.2 worker 内部计划拆分

计划从现有 `runner.py` 中逐步抽出以下职责，文件名可在实现时微调，但职责边界不得合并：

```text
crop.py             # virtual camera、ray warp、valid mask、坐标逆变换
evidence.py         # 2D evidence、visibility/covariance 与 mapping DTO
association.py      # 候选矩阵、跨视角与 track-aware assignment
fusion.py           # 鲁棒多视角优化、covariance、逐点 validity
tracking.py         # 掌心 anchor、motion state、TTL、one-to-one assignment
mano_fitting.py     # anchor、PCA coarse、45D fine、track-shared shape
temporal_mano.py    # track/window MANO 参数优化
runner.py           # 阶段调度、Trace DAG、失败策略，不承载数学实现
```

### 3.3 执行顺序调整

当前 worker 是“每帧一路运行到 export”的单遍 frame-major 调度。质量最高锚点、双向
warm-start 和离线 Temporal MANO 需要看到同一 track 的未来观测，因此计划改为三个内部
pass：

1. **Observation pass**：流式解码，完成 detection/crop/2D/association/fusion/tracking，
   将紧凑结构化观测写入 trace 和有界的 track spool；不在内存保存整段原图。
2. **Kinematic pass**：按 track 选择锚点，完成 MANO 双向拟合，写入独立
   `KINEMATIC_REFINEMENT` 记录。
3. **Temporal/export pass**：按时间顺序做 Temporal MANO、导出三层数据；需要视频时按
   presentation index 二次顺序解码并生成 overlay，避免长期序列图像常驻内存。

Trace 允许阶段记录不按 frame 交错排列，前端本来就按 `frame_id` 聚合。每个新记录仍只能
引用已经写入的 parent；锚点传播的依赖方向、时间方向和状态 predecessor 必须分别记录，
不能用 append 顺序暗示时间语义。

## 4. Phase 1：局部透视 crop 与可靠 2D evidence

### 4.1 实现内容

1. detector 继续在 native fisheye 上产生 proposal，bbox 对外仍是 native distorted pixel；
2. 根据 proposal 中心射线、扩边比例和目标 crop FOV 构造共享物理光心的 `VirtualCamera`；
3. 对每个 crop pixel：virtual pinhole unproject → 旋转到 rig/source ray → KB4 fisheye
   project，生成 remap 和 valid mask；不使用单一 homography；
4. RTMPose 消费物理、未镜像的 virtual crop；adapter 显式撤销内部 resize、letterbox 和
   可选 mirror，输出 physical crop pixel；
5. 通过 ray mapping 将 2D evidence 投回 native/rectified inspection space，但融合只消费
   带明确 `pixel_space_id/virtual_camera_id` 的 physical crop evidence；
6. 保留当前 native-fisheye pose 路径作为 `baseline_native_v1` ablation profile；新路径使用
   版本化 `crop_policy_id`，不得静默替换方法身份；
7. detector 采用 0.30 seed、0.20 recovery，先保留每视角最多 4 个 proposal。recovery
   candidate 只有通过 Phase 2 的 stereo/track gate 才能进入最终最多两只手，不能单纯
   降低全局阈值。

初始 crop 策略参数必须配置化并写入 run manifest：输出尺寸、bbox 扩边、目标角分辨率或
FOV、最小 valid fraction、边界处理和 policy version。

### 4.2 Trace 与前端证据

每个 candidate 至少保存：

- native source+bbox、virtual crop、valid mask 或其统计；
- `virtual_camera_id`、intrinsics、`T_rig_from_virtual`、crop policy；
- RTMPose 原始 crop 点、score/visibility、不确定度方法 ID；
- 映射回 native 与 rectified inspection space 的 21 点，invalid 保持 `null`；
- `model_input_space=virtual_pinhole`，防止前端误标全帧 rectification 为模型输入。

前端 `HAND_POSE_2D` 节点增加同帧 `native baseline ↔ virtual crop` 对比，以及 crop 本身的
关键点叠加。所有 candidate 与最终 track 都可见，recovery candidate 必须显示其状态和
拒绝原因。

### 4.3 Phase gate

- 合成 ray round-trip 在有效区域内满足数值容差，越界 pixel 必须落入 invalid mask；
- resize/letterbox/mirror 往返不改变 landmark index，坐标误差满足测试容差；
- 0-hand 不得退化为 whole-image pose；
- 当前 120 帧回归中双手最终关联不少于 118/120，ghost/duplicate 为 0；
- 冻结目标域 2D GT 上报告中心/边缘/遮挡/快速运动分桶，目标 NME ≤ 0.05、
  PCK@0.05 ≥ 95%；
- 必须提交 native 与 virtual crop 的同帧 ablation，未证明改善时不得删除 baseline。

## 5. Phase 2：鲁棒跨视角关联、metric 3D 与 tracking

### 5.1 2D uncertainty

优先从 RTMPose SimCC 分布宽度估计每关节 2D covariance；若锁定 API 无法稳定取得分布，
先在目标域标注集上拟合版本化的 `score → pixel sigma` calibration。未经校准的 model
score 只能保留为 score，不能声称是 confidence probability。

### 5.2 association

候选 cost 至少联合：

- calibrated ray/epipolar coplanarity residual，而非固定 rectified-y 单指标；
- 逐点 uncertainty、visibility 与有效 crop mask；
- positive depth、可行深度区间、鲁棒多点重投影；
- tracker 预测位置与 TTL；
- 可靠时才使用 anatomical handedness，绝不把画面左右或 track ID 当 handedness。

assignment 仍采用 one-to-one、先最大有效匹配数再最小总成本，并允许 unmatched。top-4
只存在于候选阶段，最终每帧最多输出两只手。

### 5.3 robust fusion

每关节以线性三角化作为初始化，再最小化 uncertainty-weighted Huber reprojection loss。
依次执行 valid-mask、visibility、ray angle、cheirality、depth、reprojection 和时间 skew
gate。根据投影 Jacobian 与 2D covariance 估计 `covariance_m2`；退化或不可修复点保持
invalid，并记录唯一主拒绝原因和全部诊断量。

骨长/掌宽约束先作为 QA 与离群 gate，不把模板先验产生的坐标写回 Raw measurement。
subject/track 统计必须在足够高质量观测后建立，不能用当前异常首帧初始化“正常骨长”。

### 5.4 tracking

- anchor 固定使用有效掌心集合 `[0, 5, 9, 13, 17]` 的鲁棒中心，不再在 wrist 与全关节
  质心之间切换语义；
- 状态包含位置、速度、时间戳、协方差和 last-seen，使用真实 `dt` 预测；
- 默认 250 ms TTL 内允许 unmatched 后恢复，距离 gate 随预测 covariance 变化；
- 两手 assignment 保持 one-to-one，场景切换、标定变化和超长 gap 必须硬 reset；
- track 决策保存 candidate cost matrix、预测、gate、matched/new/recovered/rejected 原因。

### 5.5 Phase gate

- 合成几何覆盖零视差、负深度、近似平行射线、单目、离群点和异方差噪声；
- robust solver 在注入离群点时不得使其他可靠关节失效或产生非有限值；
- Raw 的 valid 点必须有当前图像证据、support view、residual 与有限 covariance；
- 当前 120 帧初始化后只保留两条长期 track，new split/ID switch 为 0；
- 双手 pair precision/recall 在标注集上均 ≥ 99%；
- 当前序列的极端骨长必须变为“修正后的可靠测量”或明确 invalid，不得继续作为 valid
  输入 MANO；单骨 track CV 目标 < 5%；
- 有 3D GT 时报告 MPJPE 和尺度误差；无 3D GT 时只报告几何/一致性回归，不宣称精度。

## 6. Phase 3：coarse-to-fine frame-wise MANO

### 6.1 track 初始化

对每条完整 track 按以下质量量选择锚点，而不是使用视频第一帧：有效关节数、2D
covariance、ray angle、重投影 residual、骨长 QA、遮挡和运动模糊。锚点选择分数及分项
必须保存。

锚点先用 wrist/MCP 掌面建立 root translation 与 global orientation 的刚体初始化；对
left/right 及多个自然 pose seed 分别拟合。handedness 的序列共识与“该帧 MANO 是否低于
20 mm”分开估计，保留独立置信度。

### 6.2 coarse-to-fine 优化

第一阶段：

- `use_pca=True`，先评估 6/10/12 个 components；
- 使用 MANO mean pose（`flat_hand_mean=False`），但不假定视频初始姿态；
- 优化 root、translation、PCA pose、track-shared beta 和有界 log-scale；
- 使用置信度/协方差加权的 Huber 3D loss 与左右 crop 2D 重投影 loss；
- 使用 pose、shape、scale 和 joint-limit prior。

第二阶段：

- 将 coarse pose 显式转换为完整 45D axis-angle；转换必须有独立数值测试；
- `use_pca=False` 做小步精修，而不是从 45D 零姿态冷启动；
- 200–500 iteration 为上限，使用 loss/gradient/parameter-delta early stop；
- 固定 track-shared beta/scale，逐帧优化 pose、global orientation 和 translation；
- 从锚点向时间前后两个方向 warm-start，传播方向和初始化来源写入 trace。

若有界 scale 在冻结验证上吸收了标定尺度错误而非个体手型，应禁用并记录偏离；不能为
降低 RMSE 任意缩放 metric 手。

### 6.3 loss 与接受条件

总损失至少拆分并记录：

```text
L = L_2d_left + L_2d_right
  + lambda_3d * L_robust_3d
  + lambda_pose * L_pose_prior
  + lambda_shape * L_shape_prior
  + lambda_joint * L_joint_limit
  + lambda_scale * L_scale_prior
```

接受条件不只看 aggregate 3D RMSE，还要检查：有效支持数、左右 2D P95 residual、wrist/tip
分组 residual、关节角合法性、有限 mesh 和优化收敛状态。20 mm 继续作为 3D 上限，不因
产出率低而放宽。

#### 6.3.1 V7.1：实证触发的二阶段鲁棒拟合与一致门禁

`v2-8e49061-h20-120` 中 25 个 MANO 拒绝帧全部包含明显的局部解剖离群点，最大异常骨段
均为 wrist→little MCP；失败帧该骨段中位数为 155.7 mm，而 accepted 帧为 105.6 mm。
这些帧的双目重投影误差并不更差，说明它们属于“极线一致但解剖上错误”的 Raw 3D
观测。现有优化使用 Huber loss，验收却使用所有有效关节的普通 RMSE，两者目标不一致。

V7.1 在不修改 Raw observation 的前提下采用以下确定性流程：

1. 第一阶段继续使用全部有效 Raw joints 做当前 Huber fit，并保存逐关节 residual 与
   `first_pass_rmse_m`；
2. 只有普通 RMSE 未通过配置的 gate（当前 profile 为 20 mm）时，才从 residual 大于同一
   gate 的关节中按 residual 降序选择
   最多 `floor(valid_count * 0.10)` 个离群点；任何情况下都保留至少 15 个有效支持点；
3. 使用第一阶段最佳参数作为初始化，对 inlier 权重重新拟合一次；被降权关节不从 Raw、
   Trace 或最终 provenance 中删除；
4. 接受条件为 `inlier_rmse_m <= gate`、`effective_joint_count >= 15`，同时要求第二阶段
   输出对全部 Raw 有效点计算的 `full_rmse_m <= 2 * gate`（当前为 40 mm）作为灾难性错误
   上限；不满足任一条件仍为 `NOT_PRODUCED`；
5. `rmse_m`/`raw_rmse_m` 继续表示最终参数对全部 Raw 有效点的普通 RMSE，不改名也不以
   inlier 值覆盖；另保留 `first_pass_rmse_m`，并新增 `inlier_rmse_m`、`weighted_rmse_m`、
   `joint_weights[21]`、`inlier_mask[21]`、`effective_joint_count`、两阶段迭代数和 gate
   reason。方法标识固定为 `RESIDUAL_TRIM_10PCT_V1`，uncertainty status 为
   `HEURISTIC_UNCALIBRATED`。

不同 handedness/seed hypothesis 之间仍按最终 `full_rmse_m` 统一排序；inlier RMSE 只用于
各 hypothesis 自身的 pass/fail，不能与未裁剪 hypothesis 的 full RMSE 混排。鲁棒诊断只
进入 Trace 与 kinematic method provenance；严格 `fhp21/v1` 顶层结构保持不变。

该切片是针对真实失败分布的可审计鲁棒门禁，不是对错误测量的概率校准，也不替代 V3
robust association、PCA coarse-to-fine 或双向 Temporal MANO。离群点比例、15 点支持数和
40 mm 安全上限必须通过新 H20 run 验证；若未达到产出率与残差 gate，继续保留失败结果并
在本节和决策表记录改变，不允许直接放宽阈值。

### 6.4 Trace 与前端证据

每个 track/frame 至少保存：anchor score、side hypothesis、seed ID、coarse/fine status、
迭代数、early-stop reason、各 loss term、21 个逐点 2D/3D residual、pose/orient/trans、
track-shared beta/scale、mapping ID、输入 Raw observation ID 和初始化 predecessor。

前端 `MANO_FRAMEWISE` 节点显示 Raw ↔ MANO 同帧叠加、逐关节 residual 热度、收敛曲线、
接受/拒绝原因。拒绝时必须显示 `NOT_PRODUCED`，下游 fallback 仍明确标记为 Raw。

### 6.5 Phase gate

- PCA → 45D 转换前后 MANO joints/vertices 在容差内一致；
- 锚点不是首帧、遮挡后恢复、左右传播和不规则时间戳均有确定性测试；
- beta/scale 每 track 共享，pose/root/trans 每帧独立，左右手模型不会串用；
- 冻结目标域上 frame-wise MANO 产出率 ≥ 95%，median RMSE ≤ 10 mm、P95 ≤ 20 mm；
- wrist/tip residual 单独达标，不能由大量易拟合 MCP/PIP 掩盖；
- H20 实跑无 NaN、无 OOM，且必须报告每 hand-frame 耗时和 peak GPU memory。

## 7. Phase 4：Temporal MANO

### 7.1 初始模式

当前项目是离线数据处理，第一版选择显式 `OFFLINE`/双向滑窗模式，不伪装成 causal。
每条 track 的 beta/scale 固定，窗口变量是每帧 local pose、SO(3) root rotation 和 metric
translation。窗口重叠结果按中心权重融合，最后做一次边界一致性 pass。

损失包含：

- 当前帧左右 2D robust reprojection 和可靠 Raw 3D evidence；
- 与 frame-wise MANO 的软约束，而不是无条件复制；
- 按真实时间戳计算的 translation/pose velocity 与 acceleration；
- SO(3) geodesic rotation smoothness，不直接对 axis-angle 做普通欧氏差；
- joint limit、pose prior 与遮挡 gap 中随时间增长的不确定度。

遮挡期间可以输出 prediction，但必须是 `evidence_source=NONE + kind=PREDICTED`，covariance
随 gap 增长；重新获得图像证据后恢复为 refined。超 TTL、track switch、scene cut、标定变化
必须 reset。

### 7.2 输出与前端

始终并列保存：

1. immutable Raw Stereo `SpatialObservation`；
2. Frame-wise MANO `PoseEstimate`；
3. Temporal MANO `PoseEstimate`。

每个 temporal 记录保存 window ID/边界、source observation/kinematic IDs、优化变量、loss
分项、是否 prediction、gap、covariance、reset reason 和 configured/applied method。最终
overlay 同时支持 Raw、Frame-wise MANO、Temporal MANO，不能只显示一个名为 stable 的
黑盒结果。

### 7.3 Phase gate

- 对真实不规则 `dt`、短遮挡、长 gap、track reset 和窗口边界有确定性测试；
- 每个输入恰好产生一个对应时间戳输出，离线 latency 能力声明正确；
- 有 GT 时 temporal MPJPE 不得比 frame-wise MANO 恶化；
- 静止片段的 3D jitter 相对 frame-wise MANO 降低 ≥ 30%；
- 报告速度/加速度分布和可能的过平滑，不只报告视觉观感；
- 若未来增加在线模式，必须另实现 `FIXED_LAG` capability 并单独验收，不能复用
  `OFFLINE` 标签。

## 8. TDD 垂直切片与交付顺序

每个切片严格遵循：先提交失败的契约/数值/fixture 测试，再完成最小实现，再在 H20 真实
数据上形成可查看 trace。一个切片未通过 gate，不进入依赖它的下一切片。

| 切片 | 首先新增的测试 | 最小可检查交付 |
|---|---|---|
| V0 基线冻结 | 固定 120-pair 指标提取器、trace schema/parent 回归、配置 snapshot | 自动生成 baseline JSON，不靠人工抄统计 |
| V1 crop geometry | ray round-trip、valid mask、边缘 bbox、左右相机、确定性 policy ID | 单帧 crop+mask+camera JSON，可在前端查看 |
| V2 pose adapter | resize/letterbox/mirror 逆变换、0-hand、21 点 cardinality | crop 上 RTMPose 与映回 native/rectified overlay |
| V3 双手 recovery | seed/recovery/top-4、ghost、duplicate、unmatched | 当前 120 帧 candidate/selection 对比 |
| V4 robust association/fusion | cost matrix、异方差、离群/退化/cheirality/covariance | Raw 3D 与逐点诊断、旧/新 solver ablation |
| V5 tracking | palm 缺点、速度预测、TTL、交叉、reset | 两条长期 track 与零 split 报告 |
| V6 MANO anchor/coarse | 非首帧锚点、多 seed、PCA 转换、左右手 | 锚点帧 accepted MANO 与收敛曲线 |
| V7 MANO bidirectional | 前后 warm-start、共享 shape、拒绝回退 | 全 track frame-wise MANO 与残差分布 |
| V8 Temporal MANO | window、真实 dt、遮挡、prediction、reset | 三层输出与 jitter/accuracy 报告 |
| V9 E2E | worker bridge、trace validate、API Range、React old/new run | H20 完整 session、overlay video、前端逐阶段 QA |

### 8.1 V0 实现结果

V0 已实现只读公开 seam：`RunArtifactReader` 或 canonical run directory 输入，生成
`fisheye-handpose/baseline-metrics/v1` JSON。CLI 用法：

```bash
uv run fisheye-handpose trace-baseline RUN_DIR --output baseline.json
```

输出冻结 core manifest 配置及 worker manifest 中的 resolved configuration、标定和模型
provenance，并从 trace 自动统计 frame、hand、逐视角 detection、association、Raw 3D
validity/invalid reason、FHP21 骨长与 track-edge CV、track 生命周期、MANO 尝试/产出/RMSE、
Temporal 实际输入 stage/method。提取前复用 canonical reader 校验 hash chain、parent DAG 和
blob；`--skip-blob-verification` 只供已单独完成 blob 校验的大 run 加速重算。JSON 不包含生成
时间，因此同一封存 run 可逐字节复现。该切片只增加指标与 CLI，没有修改 worker 算法或
已有输出。

本地纯 Python/前端测试继续由各自 `uv run pytest`、`npm test/typecheck/build` 执行；需要
CUDA 的数值/模型测试在 H20 运行。已配置 H20 上保留本地编译的 SM90 MMCV，禁止在
`deploy/mmpose-h20` 内直接 `uv sync` 替换它；按该目录 README 使用现有 GPU Python。

### 8.2 V1/V2 本地阶段性实现结果

V1 已新增 geometry-only `VirtualPerspectiveCropper`：对原始 KB4 鱼眼像素做 ray
unproject/project，生成物理未镜像的 BGR crop、valid mask、`K_virtual`、
`R_source_from_virtual` 与以左目为 rig 的 `T_rig_from_virtual`。右目路径显式组合
`T_left_from_right @ T_right_from_virtual`，因此没有把右相机光心错误放在左目原点。
`policy_id` 只描述版本化策略，candidate-specific `crop_id` 绑定标定、相机、bbox 和虚拟
相机几何，但不绑定帧像素内容。独立测试覆盖 KB4 零/非零畸变、解析角度 literal、双向
round-trip、边界 mask、左右相机、非平凡外参和非法输入。

V2 已将 OpenMMLab runtime 拆为 `detect()` 与 `infer_pose()`，保留 `infer()` 作为
`baseline_native_v1` 兼容组合。可选 profile `virtual_perspective_kb4_v1` 保证 detector 仍消费
native fisheye，RTMPose 只消费每个 hand-centred crop 的完整物理 bbox；adapter 将 21 点映回
native，再由既有标定映射到 rectified inspection space。零 detection 不调用 pose；crop
valid fraction 不足显式 `NOT_PRODUCED`；cardinality/non-finite fail closed。Trace 对每个
candidate 保存 virtual camera、crop/native 21 点、valid mask 统计，并在 retention 命中时保存
crop 与 mask 图像。提交 `8e49061` 已用该 opt-in profile 在 H20 上完成 1-pair smoke
和 120-pair canonical run，因而“真实 OpenMMLab 无法运行 virtual crop”已不是缺口。但
仍未对同一输入运行成对的 native ↔ virtual ablation，也无 2D GT，所以 profile 继续
保持 opt-in，不得宣称 virtual 优于 native 或替换默认 baseline。V3 recovery 和 V4
fusion/hand gate 虽均已进入此真实主链，各自完整 phase gate 仍未通过。

每个算法 profile 都记录完整配置和 method/version ID。旧 profile 至少保留到相邻 phase
通过冻结 gate，以便同 run 或同输入 ablation；不得靠切换未记录的默认值比较结果。

### 8.3 V0–V7 `8e49061` 代码与 H20 状态快照

本表固定描述提交 `8e49061`，并以 `v2-8e49061-h20-smoke1` 和
`v2-8e49061-h20-120` 为真实环境证据。“已接主链”或“H20 可运行”都不代表
已通过需要 GT、ablation 或尚未实现能力的 phase gate。

| 切片 | 独立 seam | 主链接入 | 本地验证边界 | H20/真实模型状态 | 当前结论 |
|---|---|---|---|---|---|
| V0 基线冻结 | `baseline.py` 与 `trace-baseline` CLI 已实现 | 已接公开 CLI，不改变 worker 算法 | reader/path/CLI/零手帧契约与确定性输出测试 | 已对 completed 120-pair run 复算；baseline JSON SHA-256 `a9f633f58ae8db0d42c8a26b54af60b3f17f7b1e08ad768386e5af33574d84a5` | V0 代码与真实 run 复算已完成；指标本身仍受无 GT 边界限制 |
| V1 crop geometry | `crop.py` 已实现 KB4 ray crop、mask、双向 mapping 与 rig pose | 通过 V2 feature flag 间接接入；native 默认不调用 | geometry-only 数值与输入契约测试，不加载 RTMPose | smoke 与 120-pair virtual profile 都实际产生 crop/mask/virtual-camera trace | 真实主链可运行；resize 数值逆变换与 native ablation gate 未通过 |
| V2 pose adapter | `pose_adapter.py` 与 runtime 的 `detect()`/`infer_pose()` 拆分已实现 | `virtual_perspective_kb4_v1` 时接入；默认仍是 `baseline_native_v1`，H20 example 选择 opt-in profile | fake runtime/worker 验证 detector 看 native、pose 看 crop、trace/blob；React 可逐 candidate 对比证据 | 真实 OpenMMLab/H20 已完成 120-pair virtual run，但无 paired native run 和 2D GT | opt-in 主链通过运行性验证；不宣称优于 native，V2 质量 gate 未通过 |
| V3 双手 recovery | `candidates.py` 已实现 0.30 seed、0.20 recovery、top-4 分类和稳定 ID | 已接 runtime/contracts/runner；native/virtual 都只对 bounded pool 跑 pose，最终最多 2 matches | 单元和 fake worker 覆盖分类、provenance、candidate ID 贯穿与 fallback | H20 记录 2,472 raw proposals：473 seed/27 recovery/1,972 rejected，pool=500；120/120 帧产生 2 matches | 候选恢复与数量回归通过；无 GT，不宣称 pair precision/recall 或 ghost/duplicate=0 |
| V4 robust association/fusion | `fusion.py` 已实现 DLT 初始化、score/covariance 加权 Huber IRLS、逐点 gate 与 3D covariance | robust **fusion** 已接入；robust **association** 未实现，仍使用 median rectified-y；Raw hand gate 拒绝保留完整下游失败链 | 数值/契约测试与 covariance 对称/PSD 进程边界 | H20 Raw 238 produced/2 rejected，4,764/4,998 valid joints，112 个完整手；输出 covariance 为 `HEURISTIC_UNCALIBRATED` | robust fusion/hand gate 可运行；异常骨长仍明显，robust association 与 calibrated covariance gate 未通过 |
| V5 tracking | `tracking.py` 已实现固定掌心 median、constant velocity、TTL recovery 与 one-to-one assignment | 已替换 runner 主 tracker；missing 帧仍推进 active-state lineage，last-seen 不变 | 本地确定性测试覆盖 palm 缺点、交叉、gap、乱序时间和 recovery | H20 只产生 2 tracks：118/120 hand-frames，NEW=2、MATCHED=236、recovered=2 | 无第三条 split 的回归通过；无身份 GT，不宣称 ID switch=0；covariance gate/cost matrix/reset 仍缺 |
| V6 MANO anchor/coarse | `mano_anchor.py` 已实现 pose-agnostic、按 track、top-K 且时间去冗余的质量锚点选择 | **未接 runner**；PCA model 与 PCA→45D 转换仍未实现 | 仅独立 evidence/排序/fail-closed 测试；不会奖励首帧或平手 | 未 H20 实测 | anchor seam 已实现，coarse-to-fine 尚未实现，不能记作 V6 完成 |
| V7 MANO bidirectional | `mano_fitting.py` 已实现 track-local accepted-state 控制器；runtime 支持完整参数 warm-start、Huber、best-so-far、early-stop 与诊断 | 已接 runner；`flat_hand_mean=False,use_pca=False`，mean-pose cold start，只有 accepted fit 更新状态，warm 失败后可 cold recovery，上限 200 iter | controller/runtime 契约覆盖 reject/error 不污染状态；仍只有按时间向前传播 | 真实 H20/MANO：213/238=89.496%；accepted RMSE median=9.008 mm、P95=16.960 mm、max=19.997 mm | 误差分布 gate 通过，产出率 95% gate 失败；双向 pass、anchor 驱动、PCA coarse 和共享 shape 全轨复估仍未实现 |
| V7.1 robust MANO gate | 本地候选已实现 full-Huber→10% residual-trim weighted refit、15 点支持、inlier gate 与 2×full ceiling | 已接 `runtime.fit_mano(joint_weights=...)`、accepted-state controller、runner Trace 和 v3 provenance；Raw 与 `fhp21/v1` 顶层不变 | deploy 187 passed/2 environment skips；另以本机真实 Torch 验证 weighted objective 与 full/weighted RMSE 分离 | H20 1-pair smoke `COMPLETED`，2/2 MANO；该帧未触发二阶段，完整 120-pair gate 仍待运行 | smoke contract PASS；必须以新 immutable 120-pair run 决定 production gate |

V8/V9 尚未开始。当前 `runner.py` 仍是单遍 frame-major 调度，Temporal 仍为 Raw 或已接受
frame-wise MANO 关节 XYZ 上的 `causal_time_ema_v1`；没有 Temporal MANO、三 pass spool 或
离线前后向传播。V7 的本地接入只解决冷启动、欠迭代和失败状态污染的一部分问题，必须先
通过 H20 MANO 产出率/RMSE gate，才允许把它当成后续 Temporal MANO 的可靠输入。
`8e49061` 的 accepted RMSE 门禁通过，但产出率只有 89.496%，所以 V7 依然未通过。

### 8.4 已接入部分的当前语义与限制

- virtual pose 路径是显式 feature flag；detector 保持 native fisheye 输入，只有 RTMPose
  proposal 进入 virtual crop。该 profile 已完成 H20 真实运行，但未做成对 native
  ablation，所以默认 profile 不变。
- V4 的鲁棒优化只能处理已给定左右观测的数值融合。仅有两个视角时，极线一致的错手或
  错关节点配对可能同时具有很低重投影误差；该问题必须由 V3/association 的跨关节、track
  与必要时 anatomy 证据解决，不能把低 reprojection residual 当作配对正确证明。
- 当前 2D covariance 没有来自 SimCC 或目标域 calibration。固定 Hand5 配置的
  `SimCCLabel(normalize=False)` 输出未归一化峰值，合法值可超过 1，不是概率。主链完整保留
  `model_keypoint_scores`，并用版本化 `CLIP_0_1_V1` 生成兼容字段
  `keypoint_scores` 作为有界 operational quality weight；只有后者用于 threshold、association
  和单位 covariance 缩放。该缩放仍只是一种启发式权重，所以 Raw `covariance_m2` 明确标记
  `HEURISTIC_UNCALIBRATED`，不得解释为校准概率不确定度。canonical run 中以
  `view_keypoints_inferred.instances[].model_keypoint_scores` 为 raw 口径，所有 pose candidate 共有
  10,500 个 raw score，41 个超过 1，最大 1.162236；bounded 上界为
  1，逐值与 `clip(raw, 0, 1)` 的 mismatch 为 0。visibility/confidence probability 仍全部为
  null。FHP21 `raw.metrics[].left_score/right_score` 保存的是 bounded operational weight，
  不是 raw SimCC response。
- robust fusion 后要求当前帧至少 3 个有效掌心点。未通过时仍保存
  `raw_hand_gate_not_produced` 及逐点证据，但不创建假 track，也不向 MANO/Temporal/export
 传播 prior 生成的手；即使同帧另一个 hand 成功，该失败 hand 也保留独立的
  Kinematic/Temporal/Export `NOT_PRODUCED` 链。Trace 因果顺序固定为
  association → Raw observation → tracking → refinement/export，而不是把已被 tracker 消费的 Raw
  observation 错写成 tracking 的后代。
- V5 tracker 使用固定距离 gate，而非计划中的 covariance-adaptive gate；当前 trace 记录
  assignment 的 anchor/prediction/distance/recovery，但没有完整候选 cost matrix 和所有
  rejected track gate。

### 8.5 `0782688` 第一次 H20 实跑证据与现场修正

真实 120-pair run `v2-0782688-h20-120` 完成了 worker 计算，但 core 在导入时按设计
fail-closed，最终状态为 `FAILED / NOT_PRODUCED`。失败不是 CUDA、模型或几何异常，而是
旧输出契约错误地把 RTMPose 未归一化 SimCC 峰值当成 `[0,1]` 概率：9,996 个双目 joint
score 中 29 个超过 1（0.290%），最大 1.09536755。worker return code 为 0，core trace
保留 request/stdout/stderr/invalid manifest/events/summary/fhp21，并可完整校验 FAILED 状态；
逐阶段图片与视频引用的 1,914 个 blob 未被导入，这是失败包可视化仍需补齐的已知缺口。

从保留的 2,900 条事件与 238 条输出中得到的非最终诊断证据为：120/120 帧均有两个跨视图
match；Raw 238 produced、2 rejected；track 数为 2（118 与 120 条）；frame-wise MANO
213/238=89.50%，通过项 RMSE median=9.008 mm、P95=16.960 mm。双手、track、导出数量和
通过项误差均较旧基线改善，但 MANO 产出率仍未达到 95% gate，且失败集中于右手
frame 41–73。由于 package 最终失败，这些数字只能作为故障诊断，不能登记为 canonical
成功基线。

现场修正保持最终协议严格：raw `model_keypoint_scores` 原样留证；bounded
`keypoint_scores` 才参与当前启发式权重；visibility/confidence probability 继续为 null。
producer 在每条 FHP21 append 前执行同一严格 validator，使此类错误在首条记录而不是整段
处理结束后暴露。修正必须先过 1-pair package smoke，再使用新 run ID 重跑 120 pair；禁止
覆盖或续跑上述 immutable FAILED run。

修正后的 smoke 与 canonical run 分别记录在 8.6 和 8.7。它们是新的 immutable
run，不改变也不删除本节的失败证据。

### 8.6 `8e49061` 1-pair package smoke

run `v2-8e49061-h20-smoke1` 以 `COMPLETED / PRODUCED` 结束；audit PASS，worker return
code 为 0。1 个 pair 产生 2 个 match、2 个 Raw/Temporal/export 输出和 2 个新 track；
MANO 2/2 产出，accepted RMSE median 为 8.123 mm。完整 `trace-validate` 得到
39 records、27 blobs、0 error 和 0 warning。

该 smoke 还直接覆盖了 score 修正边界：raw SimCC 最大值 1.00577068 被原样保留，
bounded quality 对应为 1。overlay 为 H.264、1600×1300、`yuv420p`，可完整解码
1 帧。因而本 smoke 只证明修正后的逐条 validator、raw/bounded score 双通道、真实
OpenMMLab+MANO、artifact import 和 overlay 路径可运行；它不支持对 120 帧产出率、
长期 tracking、jitter 或精度作结论。

### 8.7 `8e49061` 120-pair canonical run

run `v2-8e49061-h20-120` 是修正后的新封存 run，不是对 8.5 FAILED run 的续跑或
覆盖。其系统与产物完整性如下。

| 证据 | 观察值 | 结论边界 |
|---|---|---|
| 代码/run | commit `8e49061`；`v2-8e49061-h20-120` | 只对应本次封存配置与输入 |
| 最终状态 | `COMPLETED / PRODUCED`；audit PASS；worker return code 0；120 pairs | 证明系统和 package 完成，不等于所有算法 gate 通过 |
| Trace | 2,911 records；1,922 unique blobs；0 error/0 warning；last hash `349da756965b585f3215938cfa45250cd90c884af0d1ae429295ed2131bac423` | hash chain、parent DAG 与所有引用 blob 验证通过 |
| 物理 blobs | 293,660,212 bytes | 内容寻址去重后的实际存储；不等于 blob reference 数 |
| baseline | JSON SHA-256 `a9f633f58ae8db0d42c8a26b54af60b3f17f7b1e08ad768386e5af33574d84a5`；configuration snapshot SHA-256 `fd3ba6d5861ee87a9603f7c5676f6b5928e3aa44fbcd65a0f84cc118e115cbac` | 同一 run 可确定性复算 |
| FHP21 | 238 records，6,550,860 bytes，SHA-256 `a435070c26efcc84cfd52842997a39bc671844e52e1f7da9f28979be730f59a2` | 达到当前 export 数量回归；2 个 hand-gate reject 不伪造输出 |
| overlay | H.264 High、`yuv420p`、1600×1300、30 fps、120 decoded frames、4 s、2,443,157 bytes，SHA-256 `fa914b4b9443ae4056e3e552a5abe2f234c1386055cb446853e123e16c865f97` | 视频结构与帧数验证通过；观感不代替 jitter 数值指标 |
| event status | WARNING=41，SKIPPED=6，FAILED=0 | warning/skipped 必须结合逐手拒绝原因解释，不证明算法质量 |
| MANO 性能 | 每 hand-frame 耗时和 peak GPU memory 未埋点 | `NOT_MEASURED`；不用端到端 wall time 除以 hand 数冒充 |
| Trace API/检查路径 | run key `36b9f22a092519c8`；health/list/detail/frames、首末帧 detail、Raw reject → MANO skipped 链和 MP4 HEAD+Range `206` 均 PASS | 功能 PASS；当前 H20 list 首次 6.6 s/重复 4.75 s、detail 4.05–4.6 s，frames/record/Range 1–10 ms；性能修正后待复测，不在本记录中宣称性能 PASS |

算法层的观察和 gate 分开记录：

| 层级 | 观察值 | gate/结论 |
|---|---|---|
| Candidate policy | 2,472 raw proposals：SEED=473、RECOVERY=27、REJECTED=1,972；bounded pool=500 | 分类与有界池可追溯；无 GT，不宣称 detector recall/FP |
| Association | 120/120 帧均产生 2 matches，共 240；pool unmatched left=18/right=2 | `≥118/120` 数量回归 PASS；无 GT，不宣称 match 正确、precision/recall 或 ghost=0 |
| Raw hand gate | 238 produced；frame 24 和 86 各 1 个 `INSUFFICIENT_PALM_SUPPORT` reject | fail-closed 语义 PASS；拒绝手不建假 track |
| Raw joints | 4,764/4,998 valid=95.318%；112 个 complete-21 hand-frame | 仅是几何有效性，不是 3D 精度 |
| Raw bones | median=33.540 mm，P95=108.005 mm，max=185.142 mm；217 个 hand-frame 存在 >50 mm 骨边，187 个存在 >100 mm 骨边 | 异常骨长目标未通过；无 GT 不报 MPJPE |
| Tracking | `track-0000` 118 hand-frames（0–119），`track-0001` 120（0–119）；NEW=2、MATCHED=236、recovered=2 | 两条长轨且无第三条 split 的数量回归 PASS；无身份 GT，不宣称 ID switch=0 |
| MANO production | 213/238=89.496%；25 no-fit | `≥95%` FAIL，不放宽 20 mm 门禁 |
| MANO accepted residual | RMSE median=9.008 mm、P95=16.960 mm、max=19.997 mm | median/P95 residual gate PASS；这是对 Raw 3D 的拟合残差，不是 GT accuracy |
| MANO failures | `track-0000`: frames 39–40；`track-0001`: 41、42、44–54、56、58–61、67–70、73 | 失败集中段需逐帧查上游 Raw/关联与初始化，不从成功分布中删除 |
| MANO attempts | 266；init source `ACCEPTED_STATE`=231、`COLD_RECOVERY`=21、`COLD_START`=14 | 只说明控制器实际路径；PCA/anchor/bidirectional 仍未实现 |
| Temporal | 238 produced；input MANO=213/Raw=25；method=`causal_time_ema_v1` | jitter gate `NOT_EVALUATED`；不是 Temporal MANO |
| Score semantics | `view_keypoints_inferred` 口径 10,500 raw；41 个 >1，max=1.162236；bounded max=1，clip relation mismatch=0；method=`CLIP_0_1_V1`/`HEURISTIC_UNCALIBRATED` | score 契约修正 PASS；FHP21 left/right score 是 bounded；visibility/confidence probability 仍全为 null |
| Export | 238/240 | `≥238/240` PASS；缺少的 2 个是显式 Raw hand-gate reject |

逐阶段可追溯性统计如下。`blob refs` 是记录引用次数，`unique blobs` 是内容寻址
去重后的物理对象数；两者不应相互替代。

| Stage | records | blob refs | unique blobs |
|---|---:|---:|---:|
| SYSTEM | 2 | 7 | 7 |
| DISCOVERY | 1 | 0 | 0 |
| CALIBRATION | 1 | 0 | 0 |
| SYNCHRONIZATION | 123 | 240 | 240 |
| DECODE | 2 | 0 | 0 |
| RECTIFICATION | 122 | 480 | 480 |
| QA | 2 | 1 | 1 |
| DETECTION | 240 | 0 | 0 |
| POSE_2D | 1,216 | 1,000 | 501 |
| CROSS_VIEW_ASSOCIATION | 240 | 0 | 0 |
| RAW_FUSION | 240 | 240 | 240 |
| KINEMATIC_REFINEMENT | 241 | 213 | 213 |
| TEMPORAL_REFINEMENT | 240 | 238 | 238 |
| EXPORT | 241 | 2 | 2 |

### 8.8 `1579ebc` V7.1 1-pair smoke

run `v71-1579ebc-h20-smoke1` 在真实 H20、OpenMMLab 和 MANO 资产上以
`COMPLETED / PRODUCED` 结束；audit PASS、worker return code 0。1 个 pair 产生 2 个 match、
2 个 Raw/Temporal/export 与 2/2 MANO。完整 `trace-validate` 为 39 records、27 blobs、
0 error/0 warning，last hash 为
`923febadf20505209dd9b8f1e25273ed6c4161de1bd9a8352fbd36d0b4a1b27e`。

两只手本帧均由首阶段 full-Huber 直接通过，没有触发 weighted refit：track-0000 的
full RMSE 为 7.771 mm、有效支持 20；track-0001 为 8.475 mm、支持 21。track-0000 的
wrist 权重为 0 是因为 Raw wrist 无效，不是 residual trim；`trimmed_joint_indices=[]`。
这条现场证据促使前端将零权重拆成 `NO RAW SUPPORT` 与显式 `TRIMMED` 两种状态，避免
把缺测关节误报为算法裁剪。overlay 为 H.264、`yuv420p`、1600×1300、30 fps，完整解码
1 帧。

H20 runtime doctor 全通过，真实 Torch 的 weighted objective 与 ordinary/weighted RMSE
分离测试通过。该 smoke 证明 v3 provenance、gate payload、strict FHP21、artifact import 和
视频链可执行；因为本帧没有触发二阶段，它不能证明 10% trim 能提升 120 帧产出率，后者
必须由新的 full run 验收。

## 9. 总体验收矩阵

| 层级 | 验收 |
|---|---|
| 数据/几何 | 同步与标定 QA 继续通过；crop/fusion 坐标空间无混用 |
| Detection | 标注集 per-hand recall ≥ 99%，0-hand FP ≤ 1% |
| 2D | NME ≤ 0.05，PCK@0.05 ≥ 95%，包含中心/边缘/遮挡分桶 |
| Association | pair precision/recall ≥ 99%，允许 unmatched，ID switch=0 |
| 当前 120 帧回归 | 双手关联 ≥ 118/120；export ≥ 238/240；ghost/duplicate=0 |
| Raw 3D | valid 点均有 covariance/support/residual；异常点 invalid；报告 GT MPJPE 或明确无 GT |
| Tracking | 初始化后两条长期 track，无 `track-0002` 类 split |
| MANO | 产出率 ≥ 95%，median RMSE ≤ 10 mm，P95 ≤ 20 mm，分组 residual 达标 |
| Temporal | MPJPE 不恶化；静止 jitter 降低 ≥ 30%；prediction/reset 语义正确 |
| 可追溯性 | Raw/MANO/Temporal 均独立；`trace-validate` 通过；所有 applied method 可追溯 |
| 前端 | 每帧/每手/每节点 before-after、失败原因、残差和三层视频均可检查；旧 run 可降级显示 |

### 9.1 `v2-8e49061-h20-120` 验收观察

本表不修改上述门禁。`NOT_EVALUATED` 不是 PASS；它表示当前 run 没有所需 GT、
ablation、性能埋点或目标 Temporal MANO 实现。

| 层级 | 当前证据 | 状态 |
|---|---|---|
| 数据/几何 | audit PASS；120 pairs；virtual crop/fusion 坐标契约与进程边界通过测试和实跑 | 运行性 PASS；无 GT 精度结论 |
| Detection | 候选分类和 pool 完整留证 | `NOT_EVALUATED`；无 per-hand presence GT，不报 recall/FP |
| 2D | raw/bounded score、crop/native/rectified 点和 mask 可查 | `NOT_EVALUATED`；无 2D GT，不报 NME/PCK |
| Association | 120/120 帧产生两个 match | 数量回归 PASS；precision/recall、ghost/duplicate 为 `NOT_EVALUATED` |
| 当前 120 帧回归 | association 120/120，export 238/240 | 两个数量 gate PASS；ghost/duplicate 未标注，整行仅 PARTIAL |
| Raw 3D | valid point 有 support/residual 与 heuristic covariance；仍有大量 >100 mm 骨边 | 异常骨长 gate FAIL；MPJPE/scale accuracy `NOT_EVALUATED` |
| Tracking | 2 条长轨，无第三条 split，2 次 recovery | 数量回归 PASS；ID switch `NOT_EVALUATED` |
| MANO | production 89.496%；accepted RMSE median/P95 9.008/16.960 mm | production gate FAIL；aggregate residual gate PASS；wrist/tip 分组与 GPU 性能 `NOT_EVALUATED` |
| Temporal | 213 个 MANO 与 25 个 Raw 输入均产生 causal XYZ EMA | `NOT_EVALUATED`；尚无 Temporal MANO、GT 对比或 jitter 指标 |
| 可追溯性 | 2,911 records/1,922 blobs，0 校验错误或 warning；Raw/MANO/Temporal 独立且 method ID 可追溯 | PASS |
| Trace API/前端检查路径 | real run 的 list/detail/frames、首末帧、Raw reject 失败链和 MP4 Range 功能通过 | 功能 PASS；H20 list/detail 性能待修正后复测 |

“当前 120 帧回归”只能防止这个已知样本退化，不能代替目标域标注集。若 238/240 与人工
presence GT 冲突，以 GT 为准并在偏离记录中修改该回归门槛，不能为了凑数制造手。

## 10. 失败与回退策略

- crop 无效：candidate 记录 `NOT_PRODUCED`，不得静默回退全图 pose；只有显式 ablation
  profile 才能运行 native baseline；
- robust fusion 不足：Raw 点 invalid，保留左右 2D evidence；
- MANO 未收敛：Frame-wise MANO `NOT_PRODUCED`，Temporal MANO 不得冒用该结果；
- Temporal MANO 未通过：保留 Raw 与 Frame-wise MANO，不用 XYZ EMA 冒充目标方法；
- overlay 失败：算法结构化输出仍可完成，但 run 标记可视化 warning；
- 任一阶段出现非有限数、坐标系不匹配、未知 parent 或 artifact hash 错误：fail closed。

## 11. 决策与偏离记录

实现中出现方案改变时，在这里追加一行，并在正文相应 Phase 同步修订。`影响`必须说明
指标、输出语义、Trace/前端兼容性以及是否需要重跑既有数据。

| 日期 | Phase/提交 | 原方案 | 实际改变 | 证据与原因 | 影响 | 批准/结论 |
|---|---|---|---|---|---|---|
| 2026-08-14 | 方案建立 | 单遍 frame-major worker | 计划改为 observation / kinematic / temporal 三 pass | 最高质量锚点、双向 warm-start 和离线 Temporal MANO 需要完整 track；整段图像不可常驻内存 | Trace 仍按 frame 聚合；overlay 计划二次解码；输出契约不变 | 待实现验证 |
| 2026-08-14 | V2 / 本地工作树 | virtual crop 通过 gate 后替换模型输入 | 先以 `virtual_perspective_kb4_v1` feature flag 接入，默认保持 native | 目前只有 fake runtime/contract 证据，尚无真实 RTMPose/H20 同帧 ablation | 旧配置与旧 run 兼容；新 profile 必须显式配置并重跑才生效 | 保持 opt-in，H20 gate 后再决定默认值 |
| 2026-08-14 | V3–V4 / 本地工作树 | recovery/robust association 后再进行 robust fusion | 先接 robust fusion，随后已补接 seed/recovery/top-4 candidate pool；association 仍沿用 median rectified-y，只把最终 cardinality 限为 2 | 候选恢复可独立降低单视图漏检，但当前两视角重投影仍无法识别极线一致的错手/错点配对 | Raw 数值 gate 与候选池均改变，所有新结果需重跑；不能将低重投影误差解释为 association 正确 | 阶段性顺序偏离；V3 candidate 已接，robust association 仍未完成 |
| 2026-08-14 | V4 / 本地工作树 | 退化 Raw 点保持 invalid，不由 prior 补写 | 增加 hand-level 最少 3 个掌心支持 gate；失败 hand 保存拒绝证据但不创建 track | 没有可靠掌心 anchor 时创建 track 会把无测量证据的对象传播到 MANO/Temporal | 相比旧 run，部分 frame/hand 将从带 track 的 Raw fallback 变为 `raw_hand_gate_not_produced` 与无 track 的 `NOT_PRODUCED`；前端需显示拒绝事件，既有数据需重跑 | 接受该 fail-closed 语义；禁止假 track |
| 2026-08-14 | V5 / 本地工作树 | tracker 包含 position/velocity/covariance 与 covariance-adaptive gate | 当前只接入固定掌心 median、constant velocity、固定距离 gate 和 TTL | 这些能力已有本地确定性测试；tracker state 尚无 covariance，trace 也没有完整 cost matrix/rejected gate | method ID 已改变但 phase gate 未通过；需 H20 重跑评估 split/ID switch | 部分实现，不删除后续 covariance/reset 工作 |
| 2026-08-14 | V6–V7 / 本地工作树 | 先完成非首帧锚点与 PCA coarse→45D fine，再做双向 warm-start | 已实现未接主链的 anchor selector；PCA 延后，先把 mean-pose/full-45D accepted-state warm-start 接入真实 runtime/runner，并将上限 40→200 | smplx 0.1.28 的 PCA components/mean 到 full-45D 转换尚未在 H20 做等价性验证；已有首帧 200/500 iter 收敛证据可直接支持先修欠迭代 | 新主链不假设首帧平手，reject/error 不污染 accepted state；但只有前向 warm-start，不能称 bidirectional 或 coarse-to-fine | 有意偏离：先交付风险较低的 frame-wise v2；H20 gate 后再决定 PCA 与三 pass 接入 |
| 2026-08-14 | V7 / 本地工作树 | accepted-state warm-start 失败后直接拒绝该帧 | 同一帧增加锁定 handedness/beta 的 cold recovery；仍只有通过 RMSE gate 的结果才更新状态 | 单一 warm basin 失败不应抹掉同帧从 mean seed 恢复的机会；控制器测试覆盖 warm reject/error、recovery 成功/失败与状态不污染 | 每帧最坏拟合次数增加；Trace `selection.attempts` 会出现 `COLD_RECOVERY`，旧 run 不变，新 run 需重跑比较产出率与耗时 | 接受；H20 gate 同时检查 RMSE、耗时和 recovery 占比 |
| 2026-08-14 | Trace / 本地工作树 | tracking event 可直接 parent association，Raw 在其后补写 | Raw observation（包括 zero-match 的 `raw:none`）先落盘，tracking parent 指向本帧 accepted/rejected/empty Raw；MANO 从当前 tracking 开始；每个 rejected hand 都补齐下游失败链 | tracker 实际已消费 Raw observation，旧 DAG 因果方向相反；mixed-valid/rejected 双手测试证明失败 hand 曾在后续节点消失；missing 帧仍会改变 active tracker 的 missed-update 状态 | 只改变新 Trace 的 parent DAG、active-state lineage 与失败证据，不改变 FHP21 成功输出；旧 run 仍可读，新结果需重跑 | 修正为与实际数据流一致的因果链 |
| 2026-08-14 | Trace UI / 本地工作树 | virtual crop 候选图像全部立即加载，run detail 每次重扫全部 blob | 候选证据改为一次只展开一个；内容寻址 artifact 使用 private immutable cache；封存 run 的完整 blob validation 使用 1 秒短缓存，artifact 访问仍实时校验，ACTIVE run 不缓存 | v2 每帧最多 8 份 crop/mask 证据，旧 eager/no-store 路径会重复下载并对数百 blob 逐次 stat，造成明显卡顿 | finalized run detail 的完整性状态最多延迟 1 秒反映未授权的磁盘篡改；直接 artifact 访问立即发现并失效缓存；算法输出与 trace schema 不变 | 接受短缓存折中；保留 fail-closed artifact 读取与测试时可注入时钟 |
| 2026-08-14 | V8 / 本地工作树 | 三 pass 离线 Temporal MANO | 尚未实现；继续使用单遍 `causal_time_ema_v1` | runner 仍对 Raw/accepted MANO 的 XYZ 做 causal EMA，没有窗口参数优化或三层独立输出 | 当前 stable 只能声明 XYZ EMA；没有 Temporal MANO 指标、视频或能力标签，既有输出契约暂不变 | 不是设计替换，只是待实现；不得把 EMA 改名为 Temporal MANO |
| 2026-08-14 | V2/V4 / H20 run `v2-0782688-h20-120` | RTMPose `keypoint_scores` 可直接作为 `[0,1]` confidence | 明确保留未归一化 `model_keypoint_scores`，另以 `CLIP_0_1_V1` 派生有界 operational quality；最终 probability 仍为空 | pinned SimCC codec `normalize=False`，真实 9,996 个 score 中 29 个 >1；直接作为 covariance 权重会制造过度自信并使严格 package validator 拒绝 | FHP21 bounded score contract 不放宽；Trace 增加 raw score 与 method/status；producer 改为逐条提前验证 | 现场证据驱动修正；clip 仅是兼容启发式，目标域 calibration/SimCC distribution covariance 仍属 V4 后续工作 |
| 2026-08-14 | V2/V4 / `8e49061` smoke | score 修正必须先通过 1-pair package smoke | 使用新 immutable run `v2-8e49061-h20-smoke1` 完整执行 real OpenMMLab/MANO、逐条 validator、artifact import 和 overlay | raw max=1.00577068 原样留证且 bounded=1；2/2 export 与 MANO；39 records/27 blobs 完整校验 | 证明 package 修正可用，不使用 1 帧统计宣称长轨或质量 gate | smoke PASS；按原计划使用新 run ID 执行 120 pairs |
| 2026-08-14 | V0–V7 / `v2-8e49061-h20-120` | 修正 package 后重跑 120 pairs，MANO production 目标 ≥95% | canonical run `COMPLETED / PRODUCED`，但 MANO production 仅 213/238=89.496%；25 个失败仍显式回退 Raw | Trace 2,911 records/1,922 blobs 完整校验；accepted RMSE median/P95 9.008/16.960 mm；2 个 Raw hand-gate reject；2 条长轨 | score/package 修正已验证；不放宽 20 mm 或 95% 门禁；Temporal 仍有 25 个 Raw fallback，新指标不可写回历史 baseline | canonical 系统 run PASS；V7 production gate FAIL；robust association、PCA/bidirectional、Temporal MANO 和 GT 精度继续待完成 |
| 2026-08-14 | V7.1 / implementation plan | 继续增加迭代、放宽 20 mm 门槛或优先实现 PCA/双向传播 | 先实现二阶段 residual-trim refit，并用与鲁棒 loss 一致的 inlier gate；保留 ordinary RMSE 和 40 mm 全局安全上限 | 25/25 拒绝帧最大异常骨段均为 wrist→little MCP；失败/通过帧最大骨长中位数 155.7/105.6 mm，covariance std 48.9/21.6 mm，而失败帧重投影误差反而更低。反事实 trim-worst-1/10% 的预计产出率为 97.90%/99.16%；增加 seed、反向初始化和单纯增加 iteration 均未解决 | Raw observation 与 20 mm inlier 门槛不变；MANO Trace/attempt schema 增加权重、mask、两类 RMSE 和 gate reason；需要新 run ID 完整重跑，旧 run 不改写 | 方案先行，按 TDD 实现 `RESIDUAL_TRIM_10PCT_V1`；真实 H20 数据决定是否保留或修订该策略 |
| 2026-08-14 | V7.1 / local implementation | 固定写死 20/40 mm，并用 inlier RMSE 同时做 gate 与跨 hypothesis 排序 | trigger/residual threshold 从现有 `max_fit_rmse_m` 派生，full ceiling 固定为其 2 倍；gate 仍用 inlier，但 handedness/seed 统一按 final full RMSE 排序 | 独立审查发现硬编码会与合法非 20 mm 配置分叉，且不同 mask 的 inlier RMSE 不可横向比较；14 点低 RMSE、左右不同 gate、weighted reject/error accepted-state 防污染均有回归 | 默认 profile 的数值仍是 20/40 mm；配置兼容性和 handedness 选择语义更明确；Trace 新增 first/full/inlier、21 点 weights/mask、支持数与阶段迭代，FHP21 v1 不变 | 属于实现阶段必要修正，已同步正文；H20 前本地 contract PASS，真实策略效果仍待 smoke/full run |
| 2026-08-14 | V7.1 / `v71-1579ebc-h20-smoke1` | 零权重 joint 在前端统一显示为 trimmed | 根据 `trimmed_joint_indices` 区分显式 residual trim；其余 mask=false/weight=0 显示 `NO RAW SUPPORT` | 首帧 track-0000 wrist 的 Raw validity 无效、weight=0、trimmed list 为空；将其显示为裁剪会错误解释算法行为 | 只改变前端诊断标签与样式；Trace、算法和旧 run 不变；新增 produced/rejected/legacy UI tests | 现场语义修正；smoke 系统/契约 PASS，二阶段效果仍需 full run |

复制模板：

```text
| YYYY-MM-DD | Phase N / <commit> | <原方案> | <实际改变> | <数据/测试证据> | <兼容性与重跑影响> | <结论> |
```

## 12. 变更日志

| 日期 | 变更 |
|---|---|
| 2026-08-14 | 建立 v2 实施方案；记录真实 baseline、公开 seams、四阶段算法、Trace/前端产物、TDD 垂直切片、验收门槛和偏离模板。 |
| 2026-08-14 | 完成本地 V0–V5 与 frame-wise MANO v2 的可运行切片：trace baseline、KB4 virtual crop/adapter、top-4 recovery、robust fusion/hand gate、稳定掌心 tracker、MANO anchor seam 与 accepted-state warm-start；React 增加逐 candidate crop/native 对比。所有 H20 能力仍标记待实跑。 |
| 2026-08-14 | 完成 V0：新增版本化 baseline 指标提取器、core/worker 配置快照和 `trace-baseline` CLI；以四次 RED→GREEN 覆盖 reader、path、CLI 与零手帧 seam，未改变算法。 |
| 2026-08-14 | 新增 V1 本地 geometry seam：KB4 virtual crop、valid mask、左右相机 rig transform 与版本化 geometry identity；仅由 V2 feature flag 路径调用，未 H20 验证。 |
| 2026-08-14 | 新增 V2 本地 opt-in 主链：拆分 native detector/crop pose adapter，保存 candidate crop trace；保留 native baseline，等待 H20 ablation 决定是否默认启用。 |
| 2026-08-14 | 完成 V3 candidate 主链接入：native/virtual profile 均保存 seed/recovery/rejected 全部决策，对最多四个候选跑 pose，并将最终双目匹配限制为两只手；robust association 仍待实现。 |
| 2026-08-14 | 部分接入 V4：robust stereo fusion、逐点启发式 covariance 和 fail-closed hand gate 已进入主链；robust association、2D calibration 与极线一致错配防护尚未实现。 |
| 2026-08-14 | 部分接入 V5：固定掌心 anchor、constant-velocity、TTL recovery 已进入主链；covariance-adaptive gate、完整 cost trace、reset 与 H20 长轨验证待完成。 |
| 2026-08-14 | 新增 V6 pose-agnostic anchor selector seam，但尚未接主链，PCA coarse 仍延后。 |
| 2026-08-14 | 接入 V7 frame-wise MANO v2：mean pose、full-45D robust fit、200 iter 上限、best-so-far/early-stop、accepted-state warm-start 与完整诊断；双向 pass 和 Temporal MANO 仍未实现。 |
| 2026-08-14 | 加固 V4/V7 边界：covariance 跨进程校验对称/PSD；mixed-valid 双手保留逐手失败链；Trace 修正为 Raw→tracking；MANO 最后 optimizer step 纳入 best-state，并在 warm reject 后执行同侧 cold recovery，失败 attempt 不再成为状态前驱。 |
| 2026-08-14 | React 增加 mixed-hand `PARTIAL`、untracked rejection 与 `hand_reason` 展示，并将 virtual crop/mask 改为单候选按需加载；Trace API 对内容寻址 artifact 启用 immutable 浏览器缓存及封存 run 的短时 validation 缓存。 |
| 2026-08-14 | H20 首次 120-pair v2 worker 计算完成但因未归一化 SimCC score 超过 1 被 package validator 正确拒绝；记录 FAILED run、诊断指标与过程 blob 保留缺口，修正为 raw model response 与 bounded quality 双通道并将校验提前到逐条写出。 |
| 2026-08-14 | `v2-8e49061-h20-smoke1` 完成 1-pair package smoke；raw >1 score 保留、bounded weight、strict validator、真实 MANO、完整 artifact import 与 1 帧 overlay 全部通过。 |
| 2026-08-14 | `v2-8e49061-h20-120` 以 `COMPLETED / PRODUCED` 封存；回写 baseline/configuration hash、候选到导出的逐阶段证据、Trace API 功能验收与 gate 结论。双手/export 数量回归通过，MANO production 只有 89.496%，异常骨长、API 性能复测、native ablation、GT 精度、PCA/bidirectional 和 Temporal MANO 仍未完成。 |
| 2026-08-14 | 在实现前冻结 V7.1 二阶段鲁棒 MANO 方案：第一阶段全有效点 Huber，失败后最多裁剪 10% 的高残差点并从最佳参数重拟合；使用 20 mm inlier RMSE、至少 15 点支持和 40 mm full-RMSE 安全上限，Raw 与失败证据保持不可变。 |
| 2026-08-14 | 完成本地 V7.1 TDD 切片：weighted runtime、accepted-state robust gate、runner Trace 与 v3 provenance 接通；修正为配置派生 gate/2×ceiling、跨 hypothesis 按 full RMSE 排序，严格 `fhp21/v1` 顶层不变。deploy 187 项通过，真实 Torch weighted objective 通过；H20 smoke/full 仍待执行。 |
| 2026-08-14 | `v71-1579ebc-h20-smoke1` 在 H20 完成 1-pair smoke：2/2 MANO、39 records/27 blobs、strict Trace 与 1 帧 H.264 overlay 全通过；本帧未触发 weighted refit。前端据此把无 Raw 支持的零权重点与 residual-trim 点分开显示。 |
