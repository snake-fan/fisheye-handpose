# 双目鱼眼 Ego 手部 3D Pipeline 设计评审

评审对象：用户提供的《双目鱼眼 Ego 手部 3D 骨骼点提取 Pipeline 工程设计报告》

评审日期：2026-08-12
结论：**有条件通过，不应按原报告原样实现。**

报告提出的主线是正确的：神经网络负责 2D 语义观测，标定几何负责物理尺度，MANO
负责运动学约束，时序模块负责连续性；同时保存 Raw Stereo、逐帧 MANO 和时序 MANO
三层结果。这比直接用端到端 RGB-to-3D 模型更可解释，也适合设备质量评测。

但报告对真实 Orbbec 数据契约、鱼眼边缘视场、MMPose 可复现性、输出状态语义、模型
许可和旧版 PyTorch 安全风险处理不足。下述 P0 项在进入完整模型流水线前必须修正。

## 1. 决策摘要

### 1.1 保留

- 以硬件时间戳而不是帧号同步左右视频。
- 先完成标定解析、鱼眼几何、合成测试和真实数据几何 QA，再接入模型。
- 2D 模型输出 bbox、21 点、逐点分数，并保留原始模型证据。
- 低可信点不能无条件三角化；单目和无图像证据必须显式区分。
- 保留 Raw Stereo、MANO frame-wise、MANO temporal 三层产物，后两层不得覆盖 Raw。
- MANO shape 在同一 track 内共享，pose、global orientation 和 translation 按时间变化。
- 所有阈值和 loss 权重配置化，并保存模型、标定、映射和参数 provenance。
- 分阶段测试、可视化和验收；上一阶段未通过时不进入下一阶段。

### 1.2 修改

- 将输入从三个文件改为真实的多 part session，而不是假定 `left.mp4/right.mp4`。
- 将全帧 stereo rectification 从模型必经路径改为几何 QA/debug 工具；感知主路径使用
  保视场的鱼眼 proposal 和局部 virtual-perspective crop。
- 不使用会漂移的 `MMPoseInferencer(pose2d="hand")` 别名作为生产接口。固定代码 tag、
  config、checkpoint、权重 SHA-256 和 native joint schema。
- 将 detector 和 pose estimator 拆开，并让“没有检测框”产生 0 hand，而不是退化为整图
  pose 推断。
- 将简单 `|v_left-v_right|` 和线性 DLT 升级为带不确定度、可见性、时间偏差和鲁棒
  reprojection loss 的标定射线融合。
- 将单一 `joint_status` 拆成 validity、evidence source、estimate stage 和 estimate kind。
- 将 MANO fitting 改成分阶段优化，并使用真实时间间隔计算时序速度和加速度。
- 将“默认阈值”视为启动配置，不视为质量标准；阈值必须由目标域标注集校准。

### 1.3 拒绝

- 拒绝用视频帧号或 container FPS 代替硬件曝光时间戳。
- 拒绝未经厂家说明或经验验证，就认定 YAML 中的 `KB` 与 OpenCV 四参数 fisheye
  约定完全相同。
- 拒绝把整张鱼眼图强制 remap 到单张针孔图后裁掉边缘视场，再声称覆盖完整。
- 拒绝依赖 MMPose `hand` alias、运行时自动选模型或自动下载未知版本权重。
- 拒绝 detector 无输出时自动对全图运行 RTMPose。
- 拒绝用低分点、单目点或近乎平行的射线生成“Raw metric 3D measurement”。
- 拒绝将 MANO 或 temporal prediction 标记为当前帧双目实测。
- 拒绝将内部 NaN 直接写成非标准 JSON `NaN`；JSON 使用 `null + validity`。
- 拒绝自动下载、提交、打包或再分发 MANO 模型文件。
- 拒绝在含凭据或敏感挂载的通用服务环境中加载未知 `.pth/.pkl`。

## 2. P0 问题

### P0-1：报告的输入契约与真实数据不匹配

真实 session 包含：

```text
*_camera_left_partNNNN.mp4
*_camera_left_partNNNN_pts.csv
*_camera_right_partNNNN.mp4
*_camera_right_partNNNN_pts.csv
*_calibration_camera.yaml
```

当前样例是 1600×1300、约 30 FPS、多 part 可扩展的数据。标定中相机名为 `IR_L` 和
`IR_R`，模型字段为 `KB`；外参平移量约 119.9，但 YAML 本身未声明单位和 transform
方向。因此必须继续使用当前仓库已经建立的严格入口：

- 确定性 session discovery 和 part 顺序；
- 每个 part 完整解码，解码帧数必须等于对应 PTS 行数；
- 左右硬件时间戳单调、一对一、不可复用地配对；
- 显式声明 translation unit 和 extrinsics convention；
- 归一化为 `T_right_from_left` 和 metre；
- 对 baseline、旋转、畸变可逆性及真实 epipolar/disparity/positive-depth 做 QA。

报告里的 `left.mp4/right.mp4/calibration.yaml` 只能作为上层逻辑示意，不能成为实际
loader contract。完整实现也不能绕回按 frame index zip 两条视频。

### P0-2：全帧 rectification 不适合作为感知主路径

Ego 双鱼眼的价值之一是边缘视场。把整个 1600×1300 鱼眼画面 remap 为一张有限 FOV
的针孔图会出现：

- 边缘区域被裁掉或产生大量无效像素；
- 边缘的手被强烈拉伸，偏离 detector/RTMPose 的训练域；
- 为保留超大 FOV 而降低有效像素密度，手部细节反而下降；
- 后续 bbox 和关键点容易混用 source、rectified、crop 三种像素坐标。

修订后的原则是：

1. 全帧 `cv2.fisheye.stereoRectify` 继续用于标定验收、可视化和可选 baseline；
2. proposal backend 可以在原始鱼眼图、重叠 virtual tiles 或两者组合上找手；
3. adapter 对外统一返回 native distorted source pixel 中的 proposal；
4. 围绕 proposal 建立共享物理光心的局部 virtual pinhole camera；
5. RTMPose 只消费局部透视 crop，输出先逆 resize/letterbox/mirror，再回到 physical crop
   pixel；
6. 通过 virtual camera 的射线和 rig calibration 融合，不把任意局部 crop 点直接塞进
   全帧 rectification 的 `P1/P2`。

只有当左右观测确实处于同一对 rectified pixel spaces 时，`|vL-vR|` 才是直接有效的
误差。独立定向的 virtual crops 应使用 calibrated epipolar/ray coplanarity error。

### P0-3：MMPose alias 和 Inferencer 不是稳定生产边界

报告称 `hand` alias 使用 SSDLite MobileNetV2，但 MMPose v1.3.2 实际源码中的默认
hand detector 是 RTMDet-Nano：

```text
rtmdet_nano_320-8xb32_hand.py
rtmdet_nano_8xb32-300e_hand-267f9c8f.pth
```

这与文档 alias 表已经不一致，说明 alias 不能承担可复现性。v1.3.2 的
`Pose2DInferencer.preprocess_single` 还会清空传入的 `bboxes`；当 detector 无结果或
未启用时，它会构造 whole-image bbox。对本项目而言，这会把“0 hand”变成一次全图
RTMPose 推断，产生幻觉关键点的风险不可接受。

生产 adapter 应采用以下之一：

- 显式 detector + MMPose 低层 `init_model/inference_topdown`；或
- 已保证只包含单手的 virtual crop + 明确 whole-image pose 模式。

必须固定并记录：

```text
MMPose tag          v1.3.2
pose config         rtmpose-m_8xb256-210e_hand5-256x256.py
pose checkpoint     rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth
detector config     显式选择，不使用 alias
detector checkpoint 显式选择
source revision     commit SHA
artifact SHA-256    每个权重文件
native joint set    名称、顺序、定义和 mapping ID
```

“无检测框输入、外部 bbox 是否被尊重、0/1/2 hand、左右手和镜像恢复”必须成为契约
测试，不能只验证 import 成功。

### P0-4：输出状态不能压缩为一个枚举

原报告的 `INVALID/STEREO_MEASURED/MONO_INFERRED/PRIOR_ONLY` 混合了至少四个维度：

| 维度 | 建议值 | 含义 |
|---|---|---|
| `validity` | `VALID / INVALID` | 坐标是否可被下游消费 |
| `evidence_source` | `MULTIVIEW / MONOCULAR / NONE` | 当前图像证据来源 |
| `estimate_stage` | `RAW_FUSION / KINEMATIC_REFINEMENT / TEMPORAL_REFINEMENT` | 产生阶段 |
| `estimate_kind` | `MEASURED / REFINED / PREDICTED` | 测量、调整或预测 |

Raw fusion 的单目 landmark 在没有可信 metric-depth backend 时应是
`INVALID + MONOCULAR`，而不是伪造深度。MANO/时序阶段可以产生有效 prediction，但它
仍是 `NONE + PREDICTED`，不是 stereo measurement。

`SpatialObservation` 必须不可变；每个 `PoseEstimate` 用 source observation ID 引用其
原始证据。内部数组允许对无效坐标使用 NaN，JSON 序列化必须写 `null` 并保留显式
validity。

### P0-5：MANO 与 checkpoint 许可必须先过门

- MMPose 代码使用 Apache-2.0，但代码许可不能自动代表所有预训练权重及其训练数据
  许可。Hand5 混合使用 COCO-hand、OneHand10K、FreiHand、RHD 和 HalpeHand；每个
  backend manifest 都要保留训练数据和权重的许可 provenance。
- MANO 注册页明确声明模型仅供非商业研究使用；`smplx` 仓库许可同样限制为非商业
  科研，并禁止未经许可再分发。
- “公司内部设备评测”是否属于商业使用不能由工程代码替代法务判断。若存在商业、
  产品、客户交付或内部生产用途，必须先取得组织法务确认或商业许可。
- `MANO_LEFT.pkl/MANO_RIGHT.pkl` 必须由获许可用户手动放置，加入 gitignore，不进入
  wheel、Docker image、模型缓存包或测试 fixture，也不能出现在 CI artifact 中。
- `.pth` 和 `.pkl` 都可能通过 Python pickle 执行代码。只加载来自批准来源且完成
  SHA-256 校验的文件；未知模型文件一律拒绝。

## 3. 推荐架构

```text
Orbbec multi-part session
  -> Discovery + full decode audit
  -> Hardware timestamp pairing
  -> Explicit KB4 calibration normalization + geometry QA
  -> FrameSet(actual per-camera timestamps)
  -> HandProposalBackend
       (native fisheye / virtual tiles internally; native source pixels externally)
  -> VirtualPerspectiveCrop + valid mask + typed VirtualCamera
  -> RTMPose Hand5 NativeViewEvidence
  -> explicit LandmarkMapper -> fhp21/v1 CanonicalViewEvidence
  -> CrossViewAssociation (allows unmatched views)
  -> robust calibrated ray fusion + covariance
  -> immutable Raw SpatialObservation
  -> optional MANO KinematicRefiner -> frame-wise PoseEstimate
  -> timestamp-aware TemporalRefiner -> temporal PoseEstimate
  -> versioned artifacts + metrics + overlays
```

模块边界必须携带 pixel space、camera/crop ID、calibration ID、timestamp、coordinate
frame、unit、mapping ID 和 backend provenance，不能传递无法判断坐标空间的裸
`[N,2]` 或 `[N,3]` 数组。

### 3.1 Proposal 与 2D evidence

- 第一版可以显式选择官方 RTMDet-Nano hand detector 作为 baseline，但必须在本项目
  鱼眼中心和边缘分别评估召回率。
- 若全图 detector 在边缘失败，优先增加有重叠的 virtual perspective tiles，而不是
  盲目放宽阈值。
- proposal、track prediction 和 pose estimator 之间的 bbox 来源必须记录。
- RTMPose 原始 score 只是模型分数，不是 visibility probability，也不是像素 covariance。
  需要在目标域标注集上做可靠性校准；未校准前只保存原始 score 和启发式 covariance，
  不应命名为 probability。
- 模型内部如做翻转或把左右手镜像到统一方向，adapter 必须在输出前恢复 physical crop
  space，并记录 handedness 处理。

### 3.2 跨视角关联和三维融合

Hungarian matching 可以保留，但 cost 至少应组合：

- calibrated epipolar/ray coplanarity residual；
- 共同可靠 landmarks 数量及其不确定度；
- disparity sign、cheirality、可行深度和最小 ray angle；
- bbox/ray 的空间一致性；
- track motion prediction 和 track-level handedness hypothesis；
- unmatched penalty，允许一侧没有对应手。

Raw fuser 先做加权射线/DLT 初始化，再最小化 robust reprojection loss。每点至少输出：

- 3D 坐标和 3×3 covariance；
- support view IDs；
- 每视角 reprojection residual；
- ray angle、cheirality 和 timestamp skew；
- validity 与 evidence source。

固定 `2.5 px`、`3 px`、`1 deg` 可以作为初始配置，不能写成设备质量结论。最终阈值应
由像素噪声、baseline、工作距离和人工标注误差分层统计得出。

### 3.3 MANO fitting

报告中 `use_pca=False + Adam + ||pose||²` 对遮挡帧约束不足。建议分阶段：

1. 在高质量双目帧用 mean/PCA pose 初始化 root translation、orientation 和 pose；
2. 在同一 track 的若干高质量帧联合估计 shared beta；
3. 冻结或强约束 beta，逐帧优化 root 与 articulation；
4. 需要时再切换 full axis-angle，并加入 joint limits 或经过批准的 pose prior；
5. 最后做 robust 2D reprojection + 3D observation 的联合精修。

不能给 MANO 增加任意自由 global scale 来“吸收”标定错误。MANO 输出单位、translation
单位和 neutral hand 尺寸必须通过单元测试确认并统一到 metre。

Hand5 的 21 个标注点与 MANO joints + tip vertices 只是在宽泛解剖意义上相近。即使采用
`744/320/443/554/671` 等常见 tip vertex，也必须生成显式 mapping record；除非 operational
construction 完全相同，否则不能标记为 `EXACT`。应在高质量帧上量化 mapping 的系统偏差。

handedness 应由整条 track 的图像、几何和拟合 loss 共同决定，不能因单帧 loss 波动而
左右切换。

### 3.4 Temporal refinement

速度和加速度必须按真实时间计算：

```text
v_t = (J_t - J_(t-1)) / delta_t
a_t = 2 * (v_t - v_(t-1)) / (delta_t + delta_(t-1))
```

窗口大小同时记录 frames 和覆盖的 nanoseconds。track switch、scene cut、calibration
变化、handedness 变化和过大时间 gap 都是硬 reset 边界。预测点的 covariance 必须随
遮挡时长增长，confidence 递减。

只减少 jitter 不等于提高精度；验收还要检查快速运动幅度、相位延迟和轨迹过度收缩。

## 4. IR / 光谱域风险

报告称输入是 RGB，但真实标定把两路相机命名为 `IR_L/IR_R`。当前抽查视频经解码器可
转换为三通道 BGR，但输出通道可能来自单通道复制或解码器的颜色转换，因此不能证明
源流是标准可见光 RGB；也不能证明相机处理链、白平衡或近红外响应与 Hand5 训练域一致。
**这是数据契约中的 P0 歧义，必须按 capture/session 记录并验证，不能凭解码输出永久
假定。**

RTMPose Hand5 配置使用三通道 RGB 归一化，并由五个常规公开手部数据集训练；没有证据
表明官方权重已覆盖本设备的鱼眼边缘、IR/近 IR、头戴视角、运动模糊或手-物交互域。
风险可能首先表现为 detector 漏检，其次才是 21 点偏差。

目标域验收集至少应覆盖：

- 画面中心、中半径和最外圈；
- 左手、右手、双手交叉和相互遮挡；
- 手持物体、桌面接触、袖口、手表和肤色差异；
- 静止、快速运动、运动模糊、强光、暗光和局部过曝；
- 若存在，RGB、灰度、IR/近 IR 各种成像模式；
- 0 hand 的负样本。

建议先建立最少 200 对同步帧、跨至少 5 个 session 的分层人工标注集，并在后续生产
验收前扩充。分别统计 detector recall/false positive、2D NME/PCK、中心到边缘的误差曲线、
跨视角 association precision/ID switch、Raw 3D valid ratio 和 reprojection residual。

将单通道复制为三通道只能解决 tensor shape，不能消除光谱域差异。若目标域指标不达标，
先微调 detector，再微调 RTMPose；训练和验证必须按 session/subject 划分，避免相邻帧
泄漏。微调权重仍需新的 model manifest、数据许可记录和 artifact hash。

## 5. H20 的 uv 环境策略

H20 算力不是约束；兼容旧 OpenMMLab 栈才是主要环境风险。不要把 CUDA 模型依赖直接
塞进当前 core 默认依赖。建议建立独立的 Linux x86_64、Python 3.10 backend 环境和独立
`uv.lock`，core 与 backend 通过 typed artifacts/API 连接。

推荐兼容矩阵：

| 组件 | 固定版本 | 说明 |
|---|---:|---|
| Python | 3.10.x | OpenMMLab 旧栈兼容面更稳 |
| NumPy | 1.26.4 | 必须 `<2`，避免 xtcocotools ABI 问题 |
| PyTorch | 2.1.0+cu121 | 与官方 MMCV torch2.1.0 wheel 精确闭合的复现基线 |
| torchvision | 0.16.0+cu121 | 与 torch 2.1.0 配套 |
| MMCV | 2.1.0 | 使用官方 cu121/torch2.1/cp310 wheel |
| MMEngine | 0.10.3 | 精确锁定的 0.10.x 版本 |
| MMDetection | 3.2.0 | MMPose 1.3.2 要求 `<3.3.0` |
| MMPose | 1.3.2 | 固定 tag/commit，不追 main |
| Chumpy | 0.71 @ `2816a138…` | 固定现代化 fork SHA，替代不能构建/import 的 0.70 |
| smplx | 0.1.28 | 仅在 MANO 许可通过且文件由用户提供后启用 |

环境规则：

- PyTorch cu121 index 使用 `explicit = true`，只允许 torch/torchvision 从该 index 获取；
- MMCV 使用精确的官方 cp310 Linux wheel URL，不经 alias、flat index 或源码 fallback；
- 不允许缺 wheel 时静默从 sdist 编译 MMCV；H20 同步时使用 locked lock
  （`uv sync --locked`）并检查实际安装的是 cp310 manylinux x86_64 CUDA wheel；
- 不同时安装 `mmcv`/`mmcv-lite`，也不同时安装 `opencv-python`/
  `opencv-python-headless`；model backend 使用自己的 OpenCV 依赖，避免污染 core；
- MMPose 1.3.2 声明的 `chumpy 0.70` 无法在现代 PEP 517 下可靠构建，并且在 NumPy 1.26
  下不能 import。当前部署锁使用经审查的现代化分支 `0.71`，并锁定完整 Git SHA；
- 项目要求 `uv>=0.12.3,<0.13`；lock 后只用 `uv sync --locked`，保存 uv 版本、lockfile
  hash、Python 版本、driver、
  CUDA runtime、GPU 型号和所有 artifact hash；
- uv 不管理 NVIDIA driver。服务器 driver 必须先满足 CUDA 12.1 runtime 要求。

当前 core lock 在 Python `<3.12` 分支可选择 NumPy 2.x，与该 legacy backend 不兼容。
因此独立 backend lock 优于强行合并；若坚持单环境，必须把整个项目 NumPy 限制改为
`<2`，并在 Python 3.10 重跑 core 全套测试。

## 6. 旧 Torch / pickle 安全边界

PyTorch 官方说明旧 release 不会回移安全修复，而 PyTorch 模型本质上应视为程序。
GHSA-53q9-r3pm-6pq6 明确指出 `torch<=2.5.1` 受影响、`torch 2.6.0` 修复；因此这里固定的
torch 2.1.0 只能作为 **legacy compatibility enclave**，不能作为长期通用服务基础：

- 容器/用户无 root、无凭据、无 SSH key，只有必要输入只读挂载和输出目录写权限；
- 权重先在受控下载步骤获取，校验 URL、大小和 SHA-256；正式推理阶段关闭外网；
- 只允许 manifest 白名单中的官方 `.pth` 和获许可的官方 MANO `.pkl`；
- 未知 checkpoint、用户上传 checkpoint、未知 Python config 和自定义 op 一律拒绝；
- 不将 `torch.distributed` 端口暴露到不可信网络；
- 对输入尺寸、帧数、bbox 数量和资源预算设置上限；
- 依赖和容器镜像做持续漏洞扫描，但不得把“扫描通过”理解为旧 torch 获得安全回补；
- 长期目标是在隔离环境中将可信 detector/pose 权重导出为 ONNX/TensorRT 或更安全的
  权重格式，并通过数值等价和目标域回归测试后，迁移到受支持的现代 runtime。

如果 MMPose/mmengine 内部必须走普通 `torch.load` 才能读取官方 checkpoint，这个加载
步骤只能发生在上述隔离区。不得为了方便关闭 artifact hash 和来源检查。

参考：

- [PyTorch Security Policy](https://github.com/pytorch/pytorch/blob/main/SECURITY.md)
- [PyTorch GHSA-53q9-r3pm-6pq6](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6)
- [MMPose v1.3.2 inference source](https://github.com/open-mmlab/mmpose/blob/v1.3.2/mmpose/apis/inferencers/pose2d_inferencer.py)
- [MMPose v1.3.2 default detector source](https://github.com/open-mmlab/mmpose/blob/v1.3.2/mmpose/apis/inferencers/utils/default_det_models.py)
- [MMPose installation/version relations](https://mmpose.readthedocs.io/en/latest/installation.html)
- [uv PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [uv package and flat indexes](https://docs.astral.sh/uv/concepts/indexes/)
- [MANO registration/license notice](https://mano.is.tue.mpg.de/register.php)
- [smplx license](https://github.com/vchoutas/smplx/blob/main/LICENSE)

## 7. 分阶段验收

每一阶段都必须产生机器可读报告、可视化和失败退出码。没有目标域精度真值前，只能说
“程序可运行/几何自洽”，不能说“模型精度通过”。

### Phase 0：环境、数据与许可

验收项：

- `uv sync --locked` 在 H20 主机成功，环境无未锁定包；
- torch 识别 H20，CUDA tensor、FP32/FP16 matmul 正常；
- `mmcv.ops.nms` 在 CUDA tensor 上运行，而不只是 import `mmcv`；
- config、detector/pose checkpoint、source revision 和 SHA-256 全部进入 manifest；
- MANO 未提供时，MANO tests 明确 skip 且 pipeline 可输出 Raw 结果；
- MANO 提供时，许可确认、路径和 hash 均存在，模型文件不进入 Git/artifact；
- 真实 session discovery、完整解码、PTS 配对和 calibration audit 通过。

失败即停止：CUDA op 不可用、依赖从 sdist 意外编译、NumPy/xtcocotools ABI 错误、
artifact hash 不匹配、数据单位/外参方向未确认。

### Phase 1：proposal、virtual crop 与 2D pose

验收项：

- synthetic virtual-pixel -> source ray -> virtual-pixel round trip 达数值容差；
- valid mask、resize、letterbox、rotation 和 mirror 逆变换有单元测试；
- 0/1/2 hand 和外部 bbox 契约测试通过；0 detection 不执行全图 pose fallback；
- 在真实中心/边缘/遮挡/快速运动样本上保存 bbox 和 21 点 overlay；
- 在分层标注集上报告 detector recall/false positive 与 2D NME/PCK，而非只看视频观感；
- NativeViewEvidence 到 `fhp21/v1` 的每个 joint 都有 mapping record。

### Phase 2：关联与 Raw metric 3D

验收项：

- 合成无噪声投影/三角化接近数值精度；
- 像素噪声、掉点、错配、近平行射线、负深度和单目输入都有失败测试；
- 0/1/2 hand 跨视角匹配允许 unmatched，不强制一一配满；
- 真实数据输出 per-joint covariance、support views、reprojection residual、ray angle、
  cheirality、timestamp skew 和 validity；
- 单目且无 metric-depth backend 的 raw landmark 必须无效；
- overlay 的 3D reprojection 与两侧原始观测人工复核通过。

该阶段是第一个业务 milestone：即使 MANO 尚未可用，也能独立输出可审计的 Raw Metric 3D。

### Phase 3：MANO frame-wise

验收项：

- 官方 MANO left/right 模型 forward、autograd、单位和 21 点 mapping 测试通过；
- synthetic MANO pose 能被投影、扰动并重新拟合；
- shared beta 在 track 内恒定，gap/track switch 后不串用；
- 逐帧输出 observation loss、2D reprojection loss、prior loss 和 optimizer 状态；
- Raw SpatialObservation 字节级不被修改；MANO 结果拥有新 ID 并引用 raw ID；
- `MEASURED/REFINED/PREDICTED` 和 evidence source 标注正确。

### Phase 4：Temporal refinement

验收项：

- irregular timestamps 下的 velocity/acceleration 计算有 synthetic test；
- gap、scene cut、calibration change、track switch 会 flush/reset；
- 输入与最终输出一一对应，timestamp 有序且 pending frame 不丢失；
- 统计 jitter、acceleration、轨迹幅值变化和时延；
- 遮挡 prediction 的 covariance 随时间增长，不会无限期保持高置信。

### Phase 5：真实短片端到端

先选择覆盖 0/1/2 hand、边缘手、遮挡和快速运动的 10–30 个同步片段，而不是立即跑全库。
每个 run 输出：

```text
run_manifest.json
raw_spatial_observations.jsonl / npz
mano_framewise_pose.jsonl / npz       # MANO 可用时
mano_temporal_pose.jsonl / npz        # temporal 可用时
metrics.json
left_overlay.mp4
right_overlay.mp4
stereo_3d_visualization.mp4
logs/
```

同一 lock、config、weights、calibration 和输入重复运行时，结果应在声明的数值确定性范围
内一致。通过短片门槛后再全量运行，并按 session 汇总：

- decode/sync/calibration pass rate；
- detector/2D 指标及中心到边缘分层；
- cross-view match rate、unmatched rate 和 ID switches；
- Raw valid ratio、reprojection、ray angle 和 covariance；
- `MULTIVIEW/MONOCULAR/NONE` 比例；
- MANO residual 及其对 Raw 的调整量；
- temporal jitter、acceleration、lag 和 prediction gap 长度。

## 8. 最终结论

方案的科学分工和阶段化方法值得保留，但“OpenCV 全帧 rectification -> MMPose hand alias ->
`cv2.triangulatePoints` -> MANO”不能直接作为最终工程定义。修订后的稳定边界应是：

```text
严格数据/标定契约
  + 保视场的 virtual-camera 感知
  + 固定版本且可审计的 2D backend
  + 不确定度感知的标定融合
  + 不覆盖 Raw 的可选 MANO/Temporal refinement
```

环境跑通的定义也必须从“可以 import”升级为“CUDA op、官方权重、真实 bbox/pose、双目
融合和许可边界均完成 smoke 验收”。满足这些条件后，这套路线适合在 H20 上继续实现和
评测；在 IR/鱼眼目标域指标出来前，不应提前承诺最终精度。
