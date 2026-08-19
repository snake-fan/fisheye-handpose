# 当前双目鱼眼手部骨骼抽取 Pipeline 问题分析报告

> **历史基线报告（已被后续 v2 工作取代）**
>
> 本文只描述提交 `5eacb7a` 和 run `h20-stage-video-5eacb7a` 的当时状态，保留用于问题
> 证据和决策追溯。它不是当前 worker 能力清单；virtual-perspective crop、候选恢复、鲁棒融合、
> tracking 和 MANO 改进的当前结果见
> [`pipeline-v2-iteration-plan.md`](pipeline-v2-iteration-plan.md)。

## 1. 报告范围

本报告分析当前 `fisheye-handpose` 在真实 H20 环境中的端到端运行结果，目标是回答：

1. 哪些阶段已经具备可靠的工程基础；
2. 哪些阶段存在确定的算法问题；
3. 当前最终输出为什么没有使用 MANO；
4. 双手漏检、轨迹拆分、3D 骨长异常和抖动分别由什么原因造成；
5. 下一轮应按什么顺序修改，以及如何量化验收。

分析基线：

- 代码提交：`5eacb7a`；
- H20 run：`h20-stage-video-5eacb7a`；
- 数据项：`Orbbec_Ego_AZEL764000H_19700102_204253`；
- 处理范围：前 120 个同步双目 pair；完整 session 有 176 个 pair；
- 输出：233 个 hand-frame、2,373 条 trace record、1,196 个 blob；
- trace 校验：hash chain、record 和 blob 完整性全部通过；
- 最终视频：120 帧、H.264、1600×1300、30 FPS。

报告中的判断分为三类：

- **确定事实**：可由代码、trace 或复现实验直接证明；
- **高风险推断**：证据支持，但仍需要标注数据或消融实验确认；
- **待验证项**：当前没有足够证据，不能作为结论。

## 2. 总体结论

当前系统的主要问题不是 RTMPose 或 MANO 模型“不成熟”，而是成熟模型之外的自定义连接层仍是基础版：

```text
native fisheye 域偏移
        ↓
未经标定的 2D score 与遮挡离群
        ↓
逐关节独立线性三角化，缺少解剖与时序门禁
        ↓
Raw 3D 出现不合理骨长和深度跳变
        ↓
MANO 每帧冷启动、40 次等权 L2 优化欠收敛
        ↓
233/233 MANO 结果被 20 mm 质量门禁拒绝
        ↓
最终退化为 Raw 3D 的逐点 causal EMA
```

按对最终数据可用性的影响排序：

| 优先级 | 问题 | 影响 |
|---|---|---|
| P0 | Raw 3D 存在极端且时序不稳定的骨长，且仍被送入后续阶段 | 阻止其直接作为可靠 3D 标签使用 |
| P0 | MANO fitter 明显欠收敛，当前 MANO 产出率为 0% | 运动学约束完全没有进入最终输出 |
| P0 | 当前 Temporal 只是 21 点 XYZ EMA，不是 Temporal MANO Optimization | 只能减小表面抖动，不能恢复骨长、关节角或遮挡状态 |
| P1 | RTMDet/RTMPose 直接消费 native fisheye，而非局部透视 crop | 鱼眼边缘、遮挡和小手的 2D 精度不可控 |
| P1 | 检测阶段在双目/时序验证前只保留 top-2 | 低分但真实的第二只手无法被后续几何恢复 |
| P1 | 关联只使用 rectified y 残差，跟踪锚点语义会切换 | 存在错配风险，且已发生一次确定的 Track 拆分 |
| P1 | 没有目标域 2D/3D GT | 无法把误差准确归因于模型、标定还是几何 |
| P2 | 当前实跑只覆盖完整 session 的前 120/176 pair | 当前统计不能代表完整数据分布 |

## 3. 逐阶段分析

### 3.1 数据发现、视频解码与时间同步

#### 当前实现

- Core audit 完整解码视频并检查视频帧数与硬件 timestamp 行数一致；
- 左右流按硬件 timestamp 单调一对一匹配；
- worker 使用匹配后的 presentation-order frame index 读取对应图像；
- clock offset 以纳秒无损传入 worker，并保留 raw/corrected timestamp provenance。

#### 实跑证据

- 左视频 183 帧、右视频 197 帧，H.264，1600×1300；
- 完整 session 共得到 176 对，重叠区有效匹配 174 对；
- overlap match rate 为 99.43%；
- 全 session 绝对 skew：median 12 μs、P95 48 μs、P99 63 μs、max 177 μs；
- 本次前 120 对：median 8 μs、P95 47 μs、max 63 μs。

#### 结论与问题

**确定事实：同步不是当前 MANO 失败和 Raw 3D 异常的主要原因。** 当前 skew 相对于 30 FPS 帧周期足够小。

仍存在以下覆盖问题：

1. 本次只处理了 120/176 对，即 68.2%；
2. 当前结论来自一个 session，尚未覆盖更多被试、边缘位置、快速运动和严重遮挡；
3. clock offset 当前显式为 0，没有跨长序列漂移估计；
4. multi-part 数据还缺少真实跨 part 的端到端验收样本。
5. 解码结果是三通道且通道差异明显，但 YAML 相机名为 `IR_L/IR_R`；不能只根据 BGR shape 声称光谱语义是标准 RGB，颜色空间和传感器模式需要进入 session metadata。

#### 建议与验收

- 先跑完整 176 对，再扩展到至少 10 个不同 session；
- 保持 P99 skew ≤ 0.25 ms；
- 每个 part 单独报告 decoded count、timestamp count、drop gap 和 overlap；
- 禁止 worker 与 core 使用不同的 pair 列表。

### 3.2 鱼眼标定

#### 当前实现

- 显式读取 KB4 内参、左右外参、单位和外参方向；
- 内部统一使用米；
- 构建 OpenCV fisheye stereo rectification；
- 使用真实同步帧做经验极线 QA。

#### 实跑证据

- 双目 baseline：119.895 mm；
- 12/12 QA sample 可用，共 2,320 个 inlier；
- rectified vertical error：median 0 px、P95 0.829 px；
- expected disparity sign：100%；
- positive triangulated depth：100%。

#### 结论与问题

当前标定足以证明“这组参数能生成基本正确的双目几何”，但不能证明所有手部工作区内的 metric 3D 都正确。

风险包括：

1. 厂家 YAML 只写 `KB`，与 OpenCV KB4 的完全等价性仍主要依赖经验 QA，而不是厂家契约；
2. ORB/背景纹理极线 QA 主要证明图像几何，没有验证手所在边缘、动态区域和已知物理尺度；
3. audit calibration ID 与 worker rectification ID 使用不同的生成规则，展示时容易把“相机标定”和“输出 rectification geometry”混为一个 ID。

#### 建议与验收

- 使用标定板覆盖中心和边缘，要求手部工作区 P95 vertical error < 2 px；
- 用已知长度或已知深度目标验证 metric scale，目标相对误差 < 2%；
- 分离并稳定记录 `camera_calibration_id` 与 `rectification_id`；
- 若厂家不能确认 KB 模型，至少保留当前经验 QA 作为每次运行 hard gate。

### 3.3 Fisheye Undistortion

#### 当前实现

worker 可以生成 undistorted 图像并保存到 trace，但这些图像只用于逐阶段检查。Detector 与 RTMPose 没有消费该图像。

#### 确定问题

前端流程图容易被理解为：

```text
undistort → detector → RTMPose
```

实际执行语义是：

```text
native fisheye → detector/RTMPose
native 2D keypoints → fisheye undistortPoints → rectified points
```

当前 UI 已明确将 undistortion 标为 `DEBUG_ONLY`，但算法仍缺少为透视模型准备的 hand-centered virtual perspective crop。

#### 根因

全幅鱼眼直接展成单张大 FOV pinhole 图会压缩中心分辨率、拉伸边缘，因而没有把 full-frame remap 接入 detector。这一判断是合理的；问题在于替代方案“局部虚拟透视 crop”尚未实现。

#### 建议与验收

- 保留 full-frame undistort 作为 QA；
- 新增围绕 hand proposal 的 virtual perspective crop；
- crop 必须保存虚拟相机、源相机、像素映射和 valid mask；
- 做 `native fisheye crop` 与 `virtual perspective crop` 的同帧 2D 精度消融。

### 3.4 Stereo Rectification

#### 实跑证据

- 输出尺寸：1600×1300；
- `balance=0.8`；
- left valid 85.58%、right valid 86.51%、common valid 84.09%；
- HFOV 159.47°、VFOV 154.84°；
- rectified P1 的 fx 约 144.79 px。

#### 问题原因

如此宽的 rectified FOV 会把手压缩到很小的像素尺度，并在边缘产生明显采样变化，因此它适合：

- 几何 QA；
- 对比可视化；
- 可选 dense stereo baseline。

它不适合直接作为当前 RTMDet/RTMPose 的整幅输入。约 15.9% 的输出像素也不在双目共同有效区域内。

#### 建议

不要将 detector 简单改成消费整幅 rectified 图。正确路径是：

```text
native/多透视 tile proposal
  → hand-centered virtual perspective crop
  → RTMPose
  → crop ray 映回相机/rig 几何
```

### 3.5 Hand Detection

#### 当前实现

- 显式固定 RTMDet-Nano hand detector；
- bbox threshold 0.30；
- 按 score 排序后直接截取 top-2；
- 零 detection 时返回空，不会对整幅图幻觉式运行 pose。

#### 实跑证据

在 240 个 view-frame 中：

- 左目 116/120 帧检测到两只手，4 帧只有一只；
- 右目 117/120 帧检测到两只手，3 帧只有一只；
- 第二候选分数：min 0.3097、P10 0.3578、median 0.3905；
- 7 个单手 view-frame 都是 detector 只返回一个合格候选，而不是 association 在两个候选中丢掉一只手。

#### 结论

**当前 detector 并非只能识别一只手。** 原前端只显示第一个实例的问题已经修复。算法层面剩余问题是第二只手的召回率和候选保留策略。

#### 根因

1. RTMDet 直接消费 native fisheye，目标域与常规透视训练域不同；
2. RTMDet 的 test pipeline 会把 1600×1300 整图 keep-ratio 缩进 320×320；本 run 的手框原图 median 约 187×206 px，进入 detector 后只有约 37.5×41.3 px，P10 约 31.8×34.1 px；
3. 第二只手的 score 靠近 0.30 阈值；
4. top-2 在双目和时序验证之前执行，第三、第四个 recovery candidate 会永久丢失；
5. 没有使用上一帧 track ROI 帮助恢复短时低分手。

#### 建议与验收

- 使用双阈值：0.30 作为 seed，0.20 作为 recovery pool；
- 每视图保留 top-4，经过双目正深度、重投影、关节支持和 track prior 后再最多保留两只手；
- 不直接全局降低最终接受阈值，以免引入 ghost hand；
- 在有 0/1/2 手 GT 的目标域验证集上要求：per-hand recall ≥ 99%，0-hand false positive rate ≤ 1%；
- 当前 120 帧回归目标：双手关联至少 118/120，且 ghost/duplicate 为 0。

### 3.6 RTMPose-m Hand5 2D Keypoints

#### 当前实现

- 使用低层 `inference_topdown`，避免 inferencer alias 和 whole-image fallback；
- 每个 detector bbox 输出 21×2 关键点和 21 个 score；
- 预测发生在 native fisheye bbox crop 上；
- 关键点预测后才映射到 rectified pixel space。

#### 实跑证据

- 共 473 个 pose instance、9,933 个 2D point；
- 所有数值有限且位于图像范围内；
- score：min 0.0905、P10 0.3412、median 0.5238、P90 0.7391；
- 146 个点低于 0.20，560 个点低于 0.30；
- 233 个已选双目 match 的 median epipolar error：median 0.679 px、P95 约 1.25–1.28 px（取决于分位数定义）、max 1.686 px；
- 94.7% 的 joint pair 最终进入 Raw 3D。

逐关节统计显示问题集中在关键位置：

- wrist 在 233 个双目手观测中只有 159 次有效；
- 74 次 wrist 因 rectified vertical error > 5 px 被拒，占 31.8%，而其 score 中位数为 0.534，并非主要由低分导致；
- ring tip 只有 165 次有效；
- little tip 只有 162 次有效，其中 71 次由低 score 造成。

#### 结论与问题

这些指标证明大多数左右 2D 观测具有良好的极线一致性，但**极线一致不等于 2D 解剖位置正确**。两个视角可能对 wrist、tip 或遮挡关节产生相似的系统偏差。

确定缺失：

1. 没有目标域人工 2D GT；
2. 没有 PCK、NME、中心/边缘、遮挡和运动模糊分桶；
3. model score 被直接当阈值使用，但它不是校准后的 visibility probability 或像素 covariance；
4. 没有保留 heatmap/SimCC 分布宽度，融合阶段只能使用点估计。
5. worker 将 RTMPose 的原生 21 点顺序直接声明为 `fhp21/v1`，但没有在每次输出中持久化逐点 operational mapping；wrist 和 tip 定义仍需验证。

#### 建议与验收

- 建立小规模但精确的目标域 2D 标注集；
- 分别报告中心、鱼眼边缘、遮挡、快速运动和双手交互；
- 目标：NME ≤ 0.05，PCK@0.05 ≥ 95%；
- 将 2D score 校准为误差或 covariance；
- 实现 virtual perspective crop 后，与当前 native crop 做同帧 ablation。

### 3.7 Cross-view Hand / Keypoint Association

#### 当前实现

- 对所有满足 score threshold 的同 index joint，计算 rectified y 差；
- 使用 median y residual 作为候选成本；
- 在最多 2×2 候选内优先最大匹配数，再最小化总成本；
- 不使用 handedness、appearance、disparity sign、可行深度或 track prior。

#### 实跑证据

- 113 帧得到两个 match，7 帧得到一个 match；
- 所有左右各有两个候选的 113 帧都成功得到两个 match；
- match support：min 17、median 21；
- 两种 2×2 完整配对的原始总成本 margin：min 3.31 px、median 15.43 px。

#### 结论

当前样本没有明显的整手关联歧义证据，association 不是本 run 中七个单手帧的原因。但现有成本函数在更复杂数据上仍是高风险点。

#### 风险原因

1. 极线 y 一致只能限制对应点位于同一极线，不能保证是同一只手；
2. 错误 match 可能先被选中，之后即使 triangulation 失败也不会回溯尝试次优 assignment；
3. 没有 track motion、handedness 或 appearance 约束；
4. 对关键点只用硬阈值，没有使用不确定度。

#### 建议与验收

- association cost 增加正 disparity、正深度、可行深度范围和 robust multi-joint reprojection；
- 加入 track motion prior，允许 unmatched；
- 有可靠左右手分类后再加入 handedness，不能把 track ID 直接当作左右手；
- 将固定 pixel gate 改成 calibrated ray/coplanarity angular gate，或至少随 rectified focal length 缩放；当前 P1 fx 约 144.79 px，5 px 大约对应 1.98°；
- 在人工双手 ID GT 上统计 pair precision/recall 和 ID switch。

### 3.8 Stereo Triangulation 与 Raw Metric 3D

#### 当前实现

- 对每个关节独立调用 `cv2.triangulatePoints`；
- 当前 gate：keypoint score ≥ 0.20、epipolar error ≤ 5 px、reprojection error ≤ 3 px、ray angle ≥ 0.5°、positive depth；
- 不满足条件的关节标为 invalid；
- 没有跨关节骨长、掌宽、深度连续性或时序一致性 gate。

#### 实跑证据

- 233 个 Raw skeleton；
- 4,634/4,893 个 joint 有效，即 94.7%；
- 每骨架有效点 min 16、median 20；
- invalid 原因：LOW_KEYPOINT_SCORE 133、EPIPOLAR_ERROR 126；
- 有效点 reprojection error：median 0.329 px、max 2.504 px；
- ray angle min 9.97°；
- depth 0.138–0.482 m。

尽管被标为 `VALID` 的 4,634 个点都通过了上述像素和几何 gate，骨架比例和时序稳定性仍明显异常；另外 259 个点已经被 low-score/epipolar gate 拒绝：

- 20 条骨边共 4,091 个有效骨长；
- median 33.76 mm、P95 105.61 mm、max 214.81 mm；
- `track-0001` 的 wrist→little MCP 边中位数为 103.3 mm，但 P95 达 199.8 mm、max 214.8 mm；
- `track-0001` 的单帧手内 depth span：median 101.7 mm、P95 206.5 mm；
- 相比之下 `track-0000` 的 depth span median 69.2 mm、P95 106.6 mm；
- 233 条骨架中只有 73 条拥有完整 21 点，占 31.3%。

说明：不能把所有 >50 mm 的边直接判错，因为 FHP21 的 wrist→MCP 是掌骨射线，neutral MANO 中也可达到约 82–95 mm。真正的强证据是**同一条边在同一 track 内出现近 2 倍跳变、极端 165–215 mm 段以及整体 depth span 剧烈变化**。

最坏的 frame 46 / `track-0001` 提供了直接例子：

- wrist disparity 72.06 px，得到 z=0.240 m；
- little MCP disparity 44.39 px，得到 z=0.390 m；
- wrist→little MCP 因而达到 214.81 mm；
- 两点却仍通过现有门禁：对应 epipolar error 为 3.424/2.048 px，reprojection error 为 1.714/1.025 px，score 为 0.316/0.343。

这证明当前 reprojection gate 主要约束 rectified vertical residual，无法发现水平 disparity 差异造成的手内深度撕裂。

#### 结论

**这是当前最明确、最严重的上游算法问题。** 小 reprojection error 只说明三角化点能重新解释输入的两个 2D 点，并不证明两个 2D 点是同一解剖 landmark，也不证明深度稳定。

#### 根因

1. 每个关节独立 DLT，没有利用手掌和手指的整体结构；
2. 没有把 2D score 转成 covariance，也没有 robust weighting；
3. 没有骨长、掌宽、深度范围和相邻帧连续性检查；
4. 遮挡或误定位关节即使通过像素 gate，也会产生很大的 metric depth 偏差；
5. 当前 reprojection gate 主要复核同一组输入观测，不是独立准确性证据。
6. 代码只要求 Raw 至少有一个有效点就将骨架标为 `PRODUCED`，没有“最低有效点 + 必要掌点”门禁；
7. 3D covariance 当前全部为未估计，下游无法进行 uncertainty-weighted MANO/temporal refinement。

#### 建议与验收

- 先估计 2D uncertainty，再做带解剖、同 track 骨长和时序先验的 hand-level joint optimization；仅对同一对 2D 观测做 nonlinear reprojection 仍可能复现错误水平 disparity；
- 使用 Huber/Geman-McClure 抑制单关节离群；
- 加入 cheirality、工作距离、掌宽、骨长和同 track 骨长稳定 gate；
- 不可修复的点保持 invalid，不要用 MANO 或 temporal 结果冒充双目实测；
- 输出每点 3×3 covariance；
- Raw 至少满足冻结的有效点数、wrist/palm 支持和几何质量后才允许 `PRODUCED`；
- 使用 subject/edge-specific 稳健骨长范围，不使用统一 50 mm 阈值；同 track 单骨长度 CV 目标 < 5%，超出稳健分布 3σ 的离群必须为 0 或明确 invalid；
- 在有 3D GT 的小型验证集上报告 absolute 与 wrist-relative MPJPE。

### 3.9 Sequence Tracking

#### 当前实现

- 最多跟踪两条当前 observation；
- wrist 有效时用 wrist 作为 anchor；
- wrist 无效时改用所有有效关节的 3D centroid；
- 基于上一时刻位置做最大基数、最小距离匹配；
- `max_root_distance=0.15 m`，`max_gap=250 ms`；
- 没有速度预测。

#### 实跑证据

- `track-0000`：113 帧；
- `track-0001`：116 帧；
- `track-0002`：4 帧，出现在 frame 108、110、115、118；
- `track-0000` 的 wrist 在 33/113 帧无效，`track-0001` 在 37/116 帧无效，锚点语义切换并非偶发；
- frame 108 的 wrist 无效，anchor 从 wrist 切到全点 centroid；
- 新 anchor 到原 track 上次 anchor 的距离为 158.409 mm，刚超过 150 mm gate，因此创建了 `track-0002`。

#### 结论

Track 拆分根因已经确定：**锚点语义切换 + 无运动预测 + 硬距离门槛**，不是出现了第三只手。

#### 建议与验收

- 始终使用固定语义的 palm center，例如 MCP `[5,9,13,17]` 的鲁棒中心；wrist 只作附加特征，MCP 缺失时优先使用 motion prediction，而不是切换到另一组点的 centroid；
- 使用 constant-velocity prediction 和 250 ms TTL；
- 距离 gate 结合 observation covariance，而不是固定 150 mm；
- 轨迹缺失时降低 confidence，不立刻创建新 ID；
- 当前 120 帧回归目标：初始化后恰好两个长期 track，ID switch/new split 为 0。

### 3.10 MANO v1.2 Frame-wise Fitting

#### 当前实现

- 左右 MANO v1.2 文件通过 manifest、bytes 和 SHA-256 验证；
- `smplx.create()` 成功加载 left/right 模型；
- MANO 16 joints 加 5 个 tip vertices 显式映射到 FHP21；
- 每帧优化 45D hand pose、global orientation、translation 和首次 beta；
- Adam 固定 40 iterations，learning rate 0.03；
- 等权 3D L2 data term，加很弱的 pose/orient/beta L2；
- 只有 RMSE ≤ 20 mm 才接受结果；
- 第一个接受结果之后才会锁定 handedness 和 beta。

#### 实跑证据

- `mano_models_loaded` 成功；
- 233 个 hand-frame 全部进入 fitting；
- 因始终没有 accepted state，每帧左右模型各尝试一次，共 466 次；
- 466 次都不是运行错误，而是 RMSE 超限后 `REJECTED`；
- RMSE：min 24.59 mm、median 38.89 mm、mean 40.84 mm、max 85.87 mm；
- `mano_output_count=0`。

代表帧的只读收敛消融：

| 迭代次数 | left RMSE | right RMSE |
|---:|---:|---:|
| 40 | 29.33 mm | 37.52 mm |
| 200 | 9.55 mm | 22.81 mm |
| 500 | 7.78 mm | 10.56 mm |

#### 结论

**当前 MANO 无产出的直接原因之一是 40 次冷启动优化严重欠收敛。** 代表帧在不放宽 20 mm 门禁的情况下，通过增加迭代已经可以成功。

但不能据此认为所有帧只增加 iterations 就会解决，因为 Raw 3D 的异常骨长与最佳 MANO RMSE 存在明显相关性，Raw 离群仍是第二个主要原因。

#### 其他实现问题

1. 每帧 pose、orientation 和 beta 从零开始，translation 只用目标均值初始化；
2. 首个 accepted frame 之前没有 warm start；
3. 即使 accepted，也只继承 beta/handedness，不继承上一帧 pose/orient/trans；
4. 所有关节等权，未使用 score、covariance、visibility 或 robust loss；
5. aggregate RMSE 会被少数深度离群点主导；
6. FHP21 与 MANO wrist/tip 是 operational mapping，不应假设为完全相同的解剖点；
7. 没有 palm-first、finger-second 的 coarse-to-fine 初始化。
8. 当前只有 fit 被接受后才保存 handedness；尽管较低 RMSE side 在 `track-0000/0001/0002` 上分别 113/113、116/116、4/4 一致偏向 left/right/left，最终 handedness 仍全部为 unknown。

#### 建议与验收

- 保持 20 mm 门禁，不能直接放宽到 40 mm；
- 允许最多 200–500 次迭代，但加入 early-stop 和 best-so-far；
- 首帧先拟合掌心刚体，再拟合手指；
- 同 track 使用上一帧 pose/orient/trans warm start；
- 用高质量帧估计一次 beta 和 subject scale，之后冻结；
- 使用 confidence/covariance-weighted Huber loss；
- 保存逐关节 residual 和收敛曲线；
- 将序列 handedness 共识与 MANO 20 mm 几何接受门禁解耦，但保留独立 confidence；
- 目标：frame-wise MANO 产出率 ≥ 95%，median RMSE ≤ 10 mm，P95 ≤ 20 mm；
- 必须同时检查 wrist/tip residual，避免 aggregate RMSE 掩盖 mapping 偏差。

### 3.11 Temporal Refinement

#### 当前实现

当前唯一方法是 `causal_time_ema_v1`：

- 按真实 timestamp 计算 alpha；
- 对每个 3D XYZ 独立做 EMA；
- gap、non-monotonic timestamp 或 input stage 变化时 reset；
- 不优化 MANO pose、orientation、translation 或 beta；
- 不使用骨长、关节角、速度和加速度约束。

#### 实跑证据

- 233/233 temporal input 都是 `RAW_FUSION`；
- 230 个 hand-frame 应用了 EMA；
- 3 个 track 的首帧发生 reset；
- MANO 字段全部为 null，handedness 全部 unknown；
- 最终 233 条 export 虽属于 `TEMPORAL_REFINEMENT` stage，但其真实语义是 Raw XYZ EMA。

#### 问题原因

EMA 可以降低高频数值变化，但存在以下限制：

1. 不能修复错误骨长或错误深度；
2. 每关节独立处理，会破坏手的整体刚性和关节结构；
3. 没有显式预测状态，当前无效点不会成为“可信测量”；
4. time constant 80 ms 会引入相位延迟，但当前没有 lag 指标；
5. Track 拆分会重置 temporal state；
6. 与目标设计中的 Temporal MANO Optimization 不相符。

#### 建议与验收

- 在 frame-wise MANO 稳定后实现 fixed-lag/sliding-window MANO；
- track 内共享 beta/scale；
- 优化 root SE(3)、local pose、2D reprojection与速度/加速度；
- 旋转使用 SO(3) geodesic/log loss，不直接平均 axis-angle；
- 对预测点明确标记 `PREDICTED` 并增大 covariance；
- 同时报告 jitter、acceleration 和 lag，不能只看画面更平滑；
- 目标：在 MPJPE 不恶化的条件下，静止段 jitter 降低 ≥ 30%，端到端 lag ≤ 1 frame。

### 3.12 Stable Metric FHP21 Export、QA 与前端

#### 当前状态

- `fhp21.jsonl` 输出契约完整，invalid 点使用 null；
- raw、MANO、temporal provenance 分离；
- trace 和大文件 blob 使用 hash chain/content address；
- 前端能逐帧查看十个节点、所有 track、MANO 失败原因和 Raw-vs-Stable 视频；
- 2×2 视频每个同步 pair 恰好一帧，可用于观察抖动。

#### 当前问题

1. 最终结果目前是 smoothed Raw，而不是 Stable Temporal MANO；
2. 393 个 warning 主要包含逐帧 MANO 未产出和 export 降级，数量大但缺少面向用户的根因聚合；
3. 视频是诊断证据，不是精度指标；
4. 没有 absolute/wrist-relative MPJPE、PCK、bone-length variance、acceleration、lag 和 ID-switch 的统一 run summary；
5. 只有 73/233 条输出具有完整 21 点，其余 160 条为 16–20 点；
6. 顶层 stage 为 `TEMPORAL_REFINEMENT`，但消费者不能只看该字段判断输入来自 Raw 还是 MANO；configured kinematic method 也不等于 applied；
7. 当前没有目标域 3D GT，因此不能声明“Stable Metric 3D Hand Skeleton 已达到精度要求”。

#### 建议与验收

- summary 按根因聚合 warning，避免每帧 warning 淹没主问题；
- 同时展示 Raw、MANO frame-wise、Temporal MANO 三层指标；
- 增加 `selected_input_stage`、`kinematic_status/rejection_reason` 和 configured/applied method；MANO 必需模式下成功率不达标应标记 `DEGRADED/FAILED`；
- 输出 coverage、invalid reason、ID switch、bone CV、jitter、lag；
- 只有达到预先冻结的 accuracy gates 后，run 才允许标记为 dataset-ready；
- 前端继续保持事实语义：MANO 未产出时必须显示 `NOT_PRODUCED`，不能把 Raw EMA 标为 MANO。

## 4. 问题之间的因果关系

当前问题不是单点故障，而是误差沿 pipeline 放大：

1. native fisheye 域偏移可能增加边缘和遮挡关节的 2D 偏差；
2. detector 提前 top-2 导致低分第二只手无法被几何恢复；
3. 2D score 未校准，三角化只能做硬阈值；
4. 独立线性 DLT 能产生低 reprojection error，但仍可能得到错误深度；
5. 缺少解剖 gate，使异常骨长被当作有效 Raw measurement；
6. tracker 的 wrist/centroid 语义切换把异常 Raw 3D 放大为 ID split；
7. MANO 用等权 L2 拟合这些离群点，又只有 40 次冷启动迭代；
8. MANO 全部拒绝后，EMA 只能平滑错误 Raw，不能恢复正确运动学结构。

因此修改顺序必须是：

```text
测量与标注
  → 鱼眼局部透视适配
  → detector recovery
  → 2D uncertainty
  → robust association/triangulation
  → 稳定 tracking
  → frame-wise MANO
  → Temporal MANO
```

如果跳过 Raw 3D 修复而直接增加 MANO 自由度或放宽门槛，MANO 只会掩盖错误测量。

## 5. 推荐整改计划

### Phase A：先建立可判定的目标域评测集

- 从中心、边缘、遮挡、快速运动和双手交互中抽样；
- 至少人工标注 2D keypoints、hand presence、cross-view identity；
- 用已知尺度/深度目标或小规模 3D GT 验证 metric 3D；
- 冻结指标和数据划分后再改算法。

### Phase B：修复感知输入与双手召回

- 实现 virtual perspective crop；
- detector 使用 0.30 seed + 0.20 recovery、top-4 candidate；
- 用双目/track gate 收敛到最终最多两只手；
- 完成 native vs virtual crop ablation。

### Phase C：修复 Raw Metric 3D

- 加入不确定度与 robust nonlinear triangulation；
- 加入深度、掌宽、骨长和时序连续性 gate；
- 修复 tracker anchor，加入 motion prediction；
- 在进入 MANO 前保证异常骨长为 0 或明确 invalid。

### Phase D：重构 MANO frame-wise

- 200–500 iteration 上限 + early-stop；
- palm rigid init + pose warm-start；
- track-shared beta/scale；
- confidence-weighted robust loss；
- 输出逐关节 residual 和收敛曲线。

### Phase E：实现真正的 Temporal MANO

- fixed-lag/sliding-window；
- root、pose、translation 与观测重投影联合优化；
- velocity/acceleration 和 joint-limit prior；
- 输出 raw/frame-wise/temporal 三套不可覆盖的结果。

## 6. 建议的阶段验收门槛

| 阶段 | 建议门槛 |
|---|---|
| Sync | P99 skew ≤ 0.25 ms；无错误 index 对齐 |
| Calibration | 手部工作区 P95 vertical error < 2 px；metric scale error < 2% |
| Detection | per-hand recall ≥ 99%；0-hand FP ≤ 1% |
| 2D Pose | NME ≤ 0.05；PCK@0.05 ≥ 95%，并按中心/边缘分桶 |
| Association | 双手 pair precision/recall ≥ 99%；允许 unmatched；ID switch=0 |
| Raw 3D | edge/subject-specific 3σ 离群为 0 或 invalid；单骨 track CV < 5%；报告 MPJPE |
| Tracking | 初始化后只保留两条长期 track；本 120 帧无新 split |
| MANO | 产出率 ≥ 95%；median RMSE ≤ 10 mm；P95 ≤ 20 mm |
| Temporal | MPJPE 不恶化；静止 jitter 降低 ≥ 30%；lag ≤ 1 frame |
| Export | raw/MANO/temporal provenance 完整；invalid/null/state 一致；trace validate 通过 |

这些门槛需要在冻结的目标域验证集上计算。没有 GT 时，reprojection error 和视觉观感不能替代 2D/3D accuracy。

## 7. 当前可以信任与不能信任的内容

### 当前可以信任

- session 文件发现、硬件 timestamp 配对和完整性审计；
- 当前标定在本 session 上的基本极线一致性；
- 每个阶段的 trace provenance、状态、blob hash 和失败原因；
- detector 在当前 120 帧多数时候确实发现两只手；
- 当前 MANO 文件已经加载，且失败是质量门禁拒绝，不是未接入；
- 前端展示的 `MANO NOT_PRODUCED` 与 `RAW_FUSION → EMA` 是真实执行路径。

### 当前不能直接信任为高质量标签

- 未经目标域 GT 验证的 2D keypoints；
- 仅凭低 reprojection error 判定为正确的 Raw 3D；
- 包含不合理骨长的 metric skeleton；
- 当前 MANO 输出，因为实际产出数为 0；
- 当前 temporal 输出的运动学正确性，因为它只是逐点 EMA；
- 仅凭 overlay 视频“看起来更平滑”得出的精度结论。

## 8. 最终判断

当前工程已经具备良好的可追溯、可复现和逐阶段可视化基础，但算法精度链尚未闭合。最优先的工作不是换一个更大的手部识别模型，而是：

1. 让透视模型在合适的局部透视输入上工作；
2. 让 Raw 3D 在进入 MANO 前满足基本解剖和不确定度约束；
3. 让 MANO 从可靠初始化和 temporal state 中收敛；
4. 用目标域 GT 和冻结指标证明每次修改真实改善，而不是只让视频更平滑。

在完成 Phase A–D 之前，当前输出应继续标记为研究/诊断结果，而不是可直接用于训练的最终 3D 骨骼标签。
