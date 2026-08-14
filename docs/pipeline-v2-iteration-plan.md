# 双目鱼眼手部骨骼 Pipeline v2 迭代实施方案

文档状态：`ACTIVE`

首次编写：2026-08-14

适用范围：本仓库本地开发、H20 兼容 worker、Trace API 与 React 检查前端
基线代码：`5eacb7a` 及其 H20 实跑 `h20-stage-video-5eacb7a`

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

### 2.2 当前 120 pair 的证据

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

当前本地工作树另有独立测试覆盖：KB4 virtual crop/ray round-trip/valid mask/左右相机 rig
transform、virtual crop pose adapter 的 0-hand/cardinality/fail-closed、seed/recovery/top-4
候选分类、Huber IRLS stereo fusion/深度与掌心 gate/启发式 covariance、固定掌心 anchor/
constant-velocity/TTL，以及“仅 accepted MANO 参数可成为下一帧 warm-start”的控制器 seam。
这些本地测试不等于真实 OpenMMLab、MANO 或 H20 数据验收；逐切片状态见 8.3。

仍无覆盖或尚未接通的关键项包括：

- crop resize/letterbox/mirror 的完整逆变换；
- SimCC covariance 提取或目标域 score calibration；
- top-4 recovery 到鲁棒跨视角关联的主链回归；
- 极线一致的错手/错点配对、track-aware association 和 anatomical gate；
- covariance-aware tracking gate、完整 cost matrix、scene/calibration reset；
- PCA coarse → 45D fine 参数转换、锚点选择、双向 warm-start；
- 逐关节 MANO residual、2D 重投影 loss 和收敛曲线；
- 新 MANO 控制器与真实 `OpenMMLabRuntime.fit_mano()` 的接口/数值集成；
- MANO 参数空间的滑窗/离线时序优化。

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
crop 与 mask 图像。该 profile 目前保持 opt-in，必须完成 H20 同帧 ablation 后才允许替换默认
baseline。V3 recovery 已接入 native 与 virtual 两条路径，V4 fusion/hand gate 也已在后续
切片接入；这些本地能力不应反向解读为 V2 已通过真实数据 gate。

每个算法 profile 都记录完整配置和 method/version ID。旧 profile 至少保留到相邻 phase
通过冻结 gate，以便同 run 或同输入 ablation；不得靠切换未记录的默认值比较结果。

### 8.3 V0–V7 当前代码状态快照

本表描述当前本地工作树，不描述已经推送或已经在 H20 跑过的版本。“已接主链”仅表示
`runner.py` 在相应 profile 下会调用该实现，不代表通过 phase gate。

| 切片 | 独立 seam | 主链接入 | 本地验证边界 | H20/真实模型状态 | 当前结论 |
|---|---|---|---|---|---|
| V0 基线冻结 | `baseline.py` 与 `trace-baseline` CLI 已实现 | 已接公开 CLI，不改变 worker 算法 | reader/path/CLI/零手帧契约与确定性输出测试 | 尚未记录由当前工作树在 H20 重算完整封存 run | 本地实现完成，真实 run 复算待执行 |
| V1 crop geometry | `crop.py` 已实现 KB4 ray crop、mask、双向 mapping 与 rig pose | 通过 V2 feature flag 间接接入；native 默认不调用 | geometry-only 数值与输入契约测试，不加载 RTMPose | 未 H20 实测 | seam 已实现，phase gate 未通过 |
| V2 pose adapter | `pose_adapter.py` 与 runtime 的 `detect()`/`infer_pose()` 拆分已实现 | `virtual_perspective_kb4_v1` 时接入；默认仍是 `baseline_native_v1`，H20 example 选择 opt-in profile | fake runtime/worker 测试验证 detector 看 native、pose 看 crop、trace/blob；React 已能逐 candidate 对比 crop/native keypoints、mask 与虚拟相机；未验证真实 OpenMMLab 内部 resize | 未运行真实 OpenMMLab/H20 ablation | opt-in 主链与前端诊断已可达，尚不能据此宣告优于 native |
| V3 双手 recovery | `candidates.py` 已实现 0.30 seed、0.20 recovery、top-4 分类和稳定 ID | 已接 runtime/contracts/runner；native/virtual 都只对 bounded pool 跑 pose，每视角最多 4 个候选，association 最终最多 2 matches | 单元和 fake worker 覆盖分类、provenance、candidate ID 贯穿、legacy fallback；duplicate/ghost 仍交给 association/track gate | 未 H20 实测 | recovery 已接主链但未通过真实双手 gate；不得把候选直接当最终手 |
| V4 robust association/fusion | `fusion.py` 已实现 DLT 初始化、score/covariance 加权 Huber IRLS、逐点 gate 与 3D covariance | robust **fusion** 已由 `geometry.triangulate_match()` 接入；robust **association** 未实现，仍使用 median rectified-y；Raw hand gate 拒绝也会生成完整下游 `NOT_PRODUCED` 链 | 数值单测与 fake worker 契约；输入 covariance 缺失时使用单位阵，输出标记 `HEURISTIC_UNCALIBRATED`；进程边界强制 covariance 对称且 PSD | 未 H20 实测 | 仅 fusion/Raw hand gate 部分完成，association gate 未通过 |
| V5 tracking | `tracking.py` 已实现固定掌心 median、constant velocity、TTL recovery 与 one-to-one assignment | 已替换 runner 主 tracker；无观测/部分缺手帧仍推进每条 active track 的 Trace 状态前驱，同时保持 last-seen 时间不变 | 本地确定性测试覆盖 palm 缺点、离群、交叉、短/长 gap、乱序时间与 observed→missing→recovered lineage | 未在 120-pair run 验证 split/ID switch | 主链已接，但 covariance-adaptive gate、完整 cost matrix 与 scene/calibration reset 尚缺 |
| V6 MANO anchor/coarse | `mano_anchor.py` 已实现 pose-agnostic、按 track、top-K 且时间去冗余的质量锚点选择 | **未接 runner**；PCA model 与 PCA→45D 转换仍未实现 | 仅独立 evidence/排序/fail-closed 测试；不会奖励首帧或平手 | 未 H20 实测 | anchor seam 已实现，coarse-to-fine 尚未实现，不能记作 V6 完成 |
| V7 MANO bidirectional | `mano_fitting.py` 已实现 track-local accepted-state 控制器；runtime 支持完整参数 warm-start、Huber、best-so-far、early-stop 与诊断 | 已接 runner；MANO model 改为 `flat_hand_mean=False,use_pca=False`，冷启动为 MANO mean pose，accepted fit 才锁 side/beta/完整参数；warm attempt 不通过时在锁定 side/beta 下尝试 cold recovery；example 上限 200 iter | fake worker/controller、单步 optimizer post-step 与 model-config 契约通过；reject/error 不污染参数或 Trace 状态前驱，gap 可重启；仍只有按时间向前传播 | 尚未在当前代码上加载真实 MANO/H20 | frame-wise v2 主链已接，双向 pass、anchor 驱动和共享 shape 全轨复估仍未实现 |

V8/V9 尚未开始。当前 `runner.py` 仍是单遍 frame-major 调度，Temporal 仍为 Raw 或已接受
frame-wise MANO 关节 XYZ 上的 `causal_time_ema_v1`；没有 Temporal MANO、三 pass spool 或
离线前后向传播。V7 的本地接入只解决冷启动、欠迭代和失败状态污染的一部分问题，必须先
通过 H20 MANO 产出率/RMSE gate，才允许把它当成后续 Temporal MANO 的可靠输入。

### 8.4 已接入部分的当前语义与限制

- virtual pose 路径是显式 feature flag；detector 保持 native fisheye 输入，只有 RTMPose
  proposal 进入 virtual crop。未做 H20 ablation 前默认 profile 不变。
- V4 的鲁棒优化只能处理已给定左右观测的数值融合。仅有两个视角时，极线一致的错手或
  错关节点配对可能同时具有很低重投影误差；该问题必须由 V3/association 的跨关节、track
  与必要时 anatomy 证据解决，不能把低 reprojection residual 当作配对正确证明。
- 当前 2D covariance 没有来自 SimCC 或目标域 calibration；单位 covariance 再按 score
  缩放只是一种启发式权重，所以 Raw `covariance_m2` 明确标记
  `HEURISTIC_UNCALIBRATED`，不得解释为校准概率不确定度。
- robust fusion 后要求当前帧至少 3 个有效掌心点。未通过时仍保存
  `raw_hand_gate_not_produced` 及逐点证据，但不创建假 track，也不向 MANO/Temporal/export
 传播 prior 生成的手；即使同帧另一个 hand 成功，该失败 hand 也保留独立的
  Kinematic/Temporal/Export `NOT_PRODUCED` 链。Trace 因果顺序固定为
  association → Raw observation → tracking → refinement/export，而不是把已被 tracker 消费的 Raw
  observation 错写成 tracking 的后代。
- V5 tracker 使用固定距离 gate，而非计划中的 covariance-adaptive gate；当前 trace 记录
  assignment 的 anchor/prediction/distance/recovery，但没有完整候选 cost matrix 和所有
  rejected track gate。

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
