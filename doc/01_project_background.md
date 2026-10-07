# WAM 终局：非显式像素世界模型与物理语言预测

**版本**：Research design draft v0.4
**日期**：2026-10-07
**参考论文**：[PhiZero: A World Model Built Around Physical Language](https://github.com/yaoyao-jpg/PhiZero)；本地 PDF：`/home/bubble/类脑计算/参考/WAM/PHIZERO.pdf`  
**项目目标**：围绕 WAM（world/action model）探索“先预测世界状态转移，再按需渲染像素”的世界模型路线，并判断是否可以进一步做到不生成未来视频，只预测可用于理解、规划和控制的物理语言/结构化状态。

## 1. 研究问题

主流视频世界模型把未来帧或视频 latent 作为主要预测目标。这样可以得到高视觉保真度，但模型是否真正学会了“物体怎样运动、接触怎样发生、动作会造成什么后果”并不容易判断，像素重建还会把大量容量用于背景、纹理和外观细节。

PhiZero 给出一个直接启发：把视频中的状态转移压缩成离散 physical language，先由 reasoner 预测离散转移，再由 diffusion decoder 渲染视频。其 tokenizer 使用时空编码器、transition-level Q-Former 和 FSQ；decoder 只接收首帧与物理语言，以鼓励瓶颈编码状态变化而不是静态外观；reasoner 则从首帧和高层动作意图自回归预测物理语言。

本项目不把 PhiZero 的设计当作最终答案，而是把它拆成可验证的研究假设：

1. **非显式世界模型**：只在内部表征空间预测未来状态，不要求生成未来像素；
2. **物理语言作为接口**：离散 token、连续 latent、对象关系图或事件序列都可以成为“物理语言”，关键是它能否支持预测、规划、干预和评价；
3. **reason-then-render 可选化**：视频渲染只是可视化、数据增强和人类检查模块，不应成为每次 rollout 的必经步骤；
4. **可干预性优先于重建分数**：好的世界模型应能回答“如果执行动作 A，目标物体的位置/接触/可达性会怎样”，而不是只在 PSNR、FVD 或视频偏好分数上表现好；
5. **跨外观和跨 embodiment 泛化**：如果状态转移表示真的与外观解耦，它应能把同一运动模式迁移到不同场景、机器人或仿真域。

## 2. 与 PhiZero 和 Next Forcing 的关系

PhiZero 的主线是：

```text
视频 V ── tokenizer ──> 离散物理语言 z
首帧 I0 + 动作意图 c ── reasoner ──> z_future
I0 + z_future ── diffusion decoder ──> 未来视频 V_future
```

本项目当前阶段保留 `z` 作为显式可检查的中间变量，但不把未来视频渲染作为 rollout 的必要步骤：

```text
观测历史 o_{≤t} ── dynamics/reasoner ──> 长时序物理语言 z_{t:t+H}
z_{t:t+H} ── token-to-observable bridge ──> 轨迹、事件和可测物理量
（可选）首帧 + z ── renderer ──> 未来视频，仅用于可视化或审计
```

Next Forcing 则是在连续视频 latent 上进行 multi-chunk prediction：主模型预测当前 chunk，辅助模块同时预测多个未来 chunk。它提供“多步监督可以缓解局部目标短视”的重要基线，但连续 latent 的 MCP 不能直接解决离散 physical token 的词表组合、窗口边界和事件顺序问题。

因此，当前工作的定位不是声称“连续 latent 一定优于离散 token”，而是研究两种表征在长时序上的不同瓶颈：

| 表征 | 现有代表 | 长序列风险 | 本项目关注点 |
|---|---|---|---|
| 连续视频 latent | Next Forcing | 局部多 chunk 预测仍可能累积漂移 | 作为 continuous-latent baseline |
| 离散 physical token | PhiZero | token block 的组合漂移、边界断裂、事件失序 | 设计全局一致的长序列预测 |

PhiZero 当前仍是数据驱动的经验性状态转移表示，不是可解释的符号物理定律；论文也指出视觉不可观测的触觉、微观粒子和长时域转移仍是限制。WAM 的研究重点因此放在“非显式但可预测、可干预、可验证”的表示，而不是宣称已经获得真正的物理定律。

## 3. 物理语言长序列的真正缺口

这里需要准确区分三个问题。

**第一，PhiZero 已经有离散 token 序列预测，但主要以固定窗口为基本单位。**
PhiZero 的 reasoner 会从当前首帧和条件出发，自回归预测一个长度为 `N` 的 physical-language token 序列；论文实现中，33 帧、9 个时间 latent 状态和每个相邻状态对的 32 个 transition symbols 组成长度为 256 的序列。它不是只预测一个 token，也不是完全没有长序列建模。当前的限制是：reasoner 的基本训练目标对应一个固定时长片段，超过这个窗口时主要采用 sliding-window rollout，把上一段的末帧作为下一段的条件。这样可以延长视频，但跨窗口的物理状态、事件阶段和误差累计缺少一个显式的持久记忆与全局约束。论文也把 hierarchical/recurrent prediction beyond fixed-duration clips 列为后续方向。

**第二，Next Forcing 解决的是连续视频 latent 的多 chunk 监督。**
它把当前 chunk 之外的 next-1、next-2、next-3 chunk 一起作为训练目标，缓解模型只看近邻时间步的 myopic supervision。这个思想可以借鉴到物理语言，但直接把 MTP/MCP 搬到离散 token 上只解决“看得更远”，不自动解决物理语言特有的组合漂移：token 表示的是局部状态转移，前一步的小误差会改变后一步的条件，离散错误经过长序列组合后可能变成不连续、重复或物理事件顺序错误。

**第三，物理语言不是完整状态，不能只依赖局部 token Markov 链。**
单个 transition token 往往只表示“发生了什么变化”，不一定包含足够的绝对位置、速度、接触阶段和历史上下文。长时序预测需要一个跨片段保留的 physical state summary，以及一个检查不同局部转移能否组成同一条全局轨迹的约束。

因此，项目的核心问题应改写为：

> **如何在离散物理语言空间中，从局部 transition token 预测扩展到具有持久状态、跨片段组合和全局物理一致性的长时序预测，而不依赖未来像素生成？**

## 4. 分层次的 novelty

### 4.1 弱主张：把 Next Forcing 搬到物理 token

在未来多个 horizon 同时预测 physical token，能够提供多步监督，是必要的训练基线，但单独不足以构成主要创新。它只能说明模型被要求预测更远的 token，不能说明这些 token 在长时间组合后仍然保持物理一致。

### 4.2 中等主张：离散物理语言比连续 latent 更容易检查

与 Next Forcing 的连续视频 latent 相比，离散 physical token 可以做 token-level 统计、转移聚类、事件 probe、序列编辑和跨外观迁移，因此具有更直接的可检查性。但应写成“更容易归因和审计”，不能未经实验直接声称离散表示具有更高的内在物理可解释性。

这里的“物理可解释性”必须由实验定义，而不是由 token 的离散形式定义。至少需要验证：

- **状态 probe**：token 或 memory 是否包含位置、速度方向、接触状态和事件阶段；
- **事件 probe**：能否区分接近、碰撞、反弹、停止、遮挡等事件及其时间顺序；
- **局部干预**：替换一个 token block 后，变化是否主要局限在对应的物理事件；
- **组合一致性**：`A` 后接 `B` 与直接观察到的 `A⊕B` 是否得到一致的可观测后果；
- **跨外观保持**：改变首帧外观和背景后，相同运动是否保留相似的 transition representation。

只有当这些 probe 和干预指标改善时，才能声称模型比连续 latent 更容易进行物理归因；token 数量更少或更离散本身不是证据。

### 4.3 强主张：全局一致的层级物理语言长序列模型

首选方法可以暂命名为 **Hierarchical Global-Consistent Physical Language Model（HG-PLM，层级全局一致物理语言模型）**。它包含三个相互配合的部分：

1. **局部转移预测**：在 token 或 token block 层面预测下一段 physical language；
2. **持久物理记忆**：每个局部转移块更新一个跨窗口的 physical state summary，记录阶段、速度趋势、接触关系和不确定性；
3. **跨片段全局约束**：要求相邻 token block 在边界状态、事件顺序、轨迹连续性和可观测物理量上能够组合成同一条长轨迹。

训练目标可以写成：

```text
L = L_local
  + λ_mtp L_multi_horizon
  + λ_mem L_state_summary
  + λ_comp L_transition_composition
  + λ_phys L_observable_physics
```

其中 `L_local` 是普通的 token 交叉熵，`L_multi_horizon` 是 Next Forcing/MTP 风格的多 horizon 预测；真正的新增部分是 `L_state_summary`、`L_transition_composition` 和 `L_observable_physics`。

物理约束不能直接对离散 token ID 做“能量守恒”。应通过一个 token-to-observable bridge，把 token 分布或 token block 映射到可测的轨迹、速度方向、接触状态、事件阶段等量，再施加可微的连续性和事件约束。只有在仿真器提供质量、速度、接触和外力时，才额外加入条件化的能量/动量残差。

### 4.4 一句话主张

建议论文主张写成：

> **现有物理语言模型主要以固定时长窗口内的局部 token 自回归为基本预测单位；我们引入持久物理记忆、多 horizon token 监督和跨片段物理一致性，使离散 physical language 能在不生成未来像素的情况下稳定预测更长时间的状态转移。**

更简短的中文版本是：

> **我们不是简单预测更多物理 token，而是让多个局部转移 token 能够组成一条全局一致的物理演化轨迹。**

## 5. 这个故事解决什么问题

它针对的是三个具体失败模式：

| 失败模式 | 现有方法的表现 | HG-PLM 的对应机制 |
|---|---|---|
| 近邻正确、长时域漂移 | 局部 CE 或单窗口预测缺少远期约束 | multi-horizon supervision + scheduled sampling |
| 滑动窗口边界不连续 | 下一窗口只依赖上一段末帧，阶段和速度可能跳变 | persistent physical memory + boundary consistency |
| token 序列物理事件失序 | 离散错误在递归组合后被放大 | transition composition + event/observable constraints |

因此“更长”不能只定义成输出 token 数更多，而应定义成：在固定算力、固定 token 预算或固定窗口扩展次数下，模型保持更低的长时域状态误差、更少的事件顺序错误和更小的窗口边界断裂。

### 5.1 建议的最小对照

```text
PhiZero fixed-window
PhiZero sliding-window
Continuous-latent Next Forcing
Discrete physical-language local AR
Discrete physical-language + MCP
HG-PLM: MCP + persistent memory + global consistency
```

主结果应比较：

- 长时域状态/轨迹误差；
- 接触、碰撞、掉落、停止等事件 F1 和事件时间误差；
- 窗口边界的状态跳变；
- token 的 codebook 使用率、重复率和语义 probe；
- 同一物理过程在不同外观下的 token 一致性；
- 预测延迟、显存和单位时间覆盖的物理时间。

这样论文的比较逻辑就不是“我们的序列比 PhiZero 长”，而是：

> **PhiZero 提供了离散物理语言和固定窗口的 reason-then-render 基线；Next Forcing 提供了连续 latent 的多步监督基线；我们研究的是如何把多步监督、持久物理记忆和全局一致性结合起来，解决离散物理语言长序列递归中的组合漂移。**

## 6. 初步 idea 候选

### Idea A：Physical-language-only world model（首选）

训练 tokenizer 将相邻 latent state 的变化编码成离散 token 或短序列；训练 dynamics model 只预测未来物理语言，不解码视频。下游用状态预测误差、接触事件、事件顺序和长时域稳定性评价。视频 decoder 只保留为离线诊断器。

**核心假设**：对于被动物理预测，预测“物体/关系/运动如何变化”比生成每一帧的纹理更重要。
**最小验证**：在 PISA/Kubric 或小型 2D/3D 物理环境中，与像素预测、连续 latent 预测、object-state 预测比较相同算力下的 multi-step state error、事件 F1 和长时域漂移。
**主要风险**：物理语言可能只记住外观变化，丢失绝对位置、遮挡下的状态和细粒度接触信息。

### Idea B：Non-explicit world model with task-sufficient latent（任务充分而非物理显式）

不强制 token 具有人工可读语义，只要求 latent 能同时支持未来状态判别、反事实动作排序和策略规划。使用 inverse dynamics、forward dynamics、contrastive predictive coding 和 policy value consistency 约束，避免把“离散 token 可读”误认为“世界模型有效”。

**核心假设**：一个对任务足够的非显式表示可能比固定的物理语言词表更容易优化，也更适合连续控制。  
**主要风险**：latent 可能退化成策略特征，跨任务复用和科学解释性不足。

### Idea C：事件图/关系图物理语言

把每个时间窗编码为对象节点、接触边、相对位姿、速度方向、容器关系和事件（接触、支撑、遮挡、破碎、流入等），再用图 transformer 预测图的增量。与 PhiZero 的一维 FSQ token 比较，检验结构化关系是否减少 token 数量并改善组合泛化。

**核心假设**：对象关系和事件比纯序列 token 更容易进行局部干预与长时域滚动。  
**主要风险**：对象检测/跟踪错误会把 perception 错误传给 dynamics；开放世界中的关系词表难以固定。

### Idea D：不渲染像素的反事实动作世界模型

给定观测历史，模型直接预测一组结构化未来后果：目标轨迹、接触是否成立、碰撞/停止事件、事件时间和不确定性。训练时用视频 tokenizer 或仿真状态产生弱标签；动作候选和规划接口保留到后续控制阶段。

**核心假设**：长时序物理预测只需要足以描述未来状态和事件的 consequence representation，不需要完整未来视频。
**最小验证**：比较 pixel rollout、连续 latent rollout、离散 physical-language rollout 和 consequence-only rollout 的状态误差、事件准确率、延迟和显存。

### Idea E：世界模型的“渲染审计器”而非“渲染核心”

让结构化状态模型负责主预测，另设一个低频 renderer 只渲染关键帧或失败样本。renderer 用来发现状态模型的视觉盲点、生成可解释 debug 证据和辅助人工评审，避免每一步都支付视频生成成本。

**核心假设**：渲染应服务于审计和数据闭环，而不是定义世界模型能力。  
**主要风险**：审计器与状态模型共享错误表征时，可能产生“看起来合理”的错误视频。

## 7. 当前推荐的主线

首轮采用 **A + 长时序物理语言扩展**：先学习一个小规模、可导出的离散 physical-language 表示；再训练局部 token predictor、multi-horizon MCP、persistent physical memory 和 transition composition/global consistency。PhiZero 风格 renderer 降为可选审计模块，B 作为连续 latent 强 baseline，C 作为结构化事件表示扩展，E 作为工程化可视化策略。动作条件控制暂不纳入阶段一。

第一阶段不追求复现论文的 128 张 A100 训练规模，也不把大规模视频数据下载作为启动条件。先用公开的小数据或仿真环境验证“离散物理语言能否在不生成未来像素的情况下稳定预测更长的被动物理序列”。

### 7.1 进一步收敛后的论文主张

建议暂用一个能概括当前问题的名称：**Hierarchical Global-Consistent Physical Language Model（HG-PLM，层级全局一致物理语言模型）**。论文主方法只回答一个问题：在不生成未来像素的情况下，离散 physical language 能否通过持久物理记忆和跨片段一致性稳定预测更长的状态转移序列？

为了避免把 MCP、scheduled sampling、RAG、熵正则和物理损失写成多个并列贡献，建议把主张收敛成一句话：

> **现有物理语言模型主要在固定窗口内进行局部 token 自回归；我们引入 multi-horizon token 监督、持久物理记忆和跨片段全局一致性，使离散 physical language 在不生成未来像素的情况下稳定预测更长的被动物理演化。**

主线收敛为两项：

1. **层级离散预测**：以 256-token chunk 和 32-token transition block 为两个时间尺度，进行 multi-horizon token supervision；
2. **可观测物理约束**：用 projector 把 block/chunk 表示接到 benchmark 可测状态，再加入条件化的状态、连续性和能量/动量约束；LaWM 的 DEL 形式先作为独立增强和消融，不把它预先写成首轮的 transition solver。

Persistent memory、boundary consistency 和 transition composition 是上述长程预测的实现机制；scheduled sampling 和历史 token 加噪是训练稳定性消融；未来的 DAgger、MPC 和控制回报评价单独作为后续阶段。

### 7.1.1 更新后的可执行版本

当前更稳妥的论文定位是：

> **在冻结 PhiZero tokenizer/decoder 并保留离散 physical-language token 作为主推理接口的前提下，改造 Reasoner，使其从 4 秒固定窗口扩展到 16--32 秒长时序 rollout，并用层级多步监督、可观测状态投影和条件化物理约束减少跨窗口漂移。**

这个表述比“PhiZero 是单步模型”更准确。PhiZero 的 Reasoner 不是只输出一个 token，而是在 4 秒窗口内用 next-token 自回归目标生成 256 个离散 physical-language token。真正的问题是：这个目标主要监督固定窗口内的局部 token 序列，长时序只能通过 sliding-window 递归外推，跨窗口的物理状态、事件阶段和误差累积没有显式约束。

首轮创新收敛为两项，物理一致性内部再分成核心机制和辅助约束：

1. **层级离散 MCP/MTP**：高层预测下一个 256-token chunk，低层预测 chunk 内的 32-token transition blocks；用 multi-horizon supervision 缓解局部 next-token 目标的短视。
2. **投影器和条件化物理约束**：将 token block/chunk embedding 映射到连续的 generalized coordinate 或 benchmark 状态，使用状态、轨迹连续性和适用场景下的能量/动量残差约束长程 rollout；LaWM 的离散 Lagrangian/DEL transition 作为后续机制级增强。

投影头负责把连续 latent 接到 benchmark 可测的状态/事件；能量 loss 只在数据字段足够完整且场景近似守恒时作为条件化辅助项。`render → re-encode` 是长 rollout 的重锚定对照，不是物理一致性的核心机制。

因此，“离散 token + 长程 + 显式物理约束”可以作为方法组合的主线，但“第一个”必须先作为待检索假设写进 related-work 任务，不能在立项文档中直接断言。

### 7.1.2 LaWM 借鉴后的物理一致性定义

本项目首轮只借鉴 LaWM 的连续状态表达、作用量/残差诊断和物理约束组织方式，不宣称已经复现 LaWM 的变分积分器。原因是 PhiZero 的 Reasoner 仍然负责离散 token 的下一段预测，而 LaWM 的核心是由离散 Euler--Lagrange（DEL）条件直接定义连续 latent 的转移。两者分成两级：

1. **主线（可直接实现）**：token block/chunk embedding → projector → `q`/状态量，使用 `L_state`、`L_smooth` 和有条件 mask 的 `L_energy`；Reasoner 仍决定下一段 token。
2. **增强（单独消融）**：在同一 projector 上增加 learned discrete Lagrangian 和 `L_DEL` residual；只有当下一状态由有限步 DEL solver 产生时，才称为 hybrid LaWM variant。

主线结构为：

```text
physical-language tokens
        ↓
continuous block/chunk representation e
        ↓
projector: e → q (and v)
        ↓
state / continuity / conditional physics losses
        ↓
32-token blocks / 256-token chunks
```

主线训练项写成：

```text
L_total =
    L_AR + λ_block L_block + λ_chunk L_chunk
  + λ_state M_state L_state
  + λ_smooth L_smooth
  + λ_energy M_conservative L_energy
```

其中 `L_state` 只在数据提供状态字段时启用，`L_smooth` 约束相邻预测状态或 block 边界的跳变，`L_energy` 只在质量、速度、外力、接触等条件足够明确的仿真片段启用。`L_smooth` 不是 LaWM 的 stationary-action 条件，只是对离散长序列有效的连续性正则。LaWM 的 `L_DEL`、latent energy drift 和 benchmark-specific physical invariance score 单独作为增强或诊断，不能混写成一个适用于所有数据的能量守恒 loss。

这里的 `q` 是学习到的 generalized coordinate，不默认等于 `(x,y,vx,vy)`。是否能从 `q` 读出这些变量，要由 probe 和 benchmark schema 决定。连续 projector 分支不能替代离散 token CE，而是为离散 token rollout 提供可观测的状态锚点；只有加入 DEL solver 并让 solver 产生下一状态时，连续分支才升级为 transition carrier。

LaWM 的二阶 transition 需要两个相邻 latent states。首轮只在 DEL 增强实验中确认初始化方式：使用两个观测/token blocks、使用首个 chunk 内的相邻 blocks，或学习一个由首帧和 caption 条件化的初始速度；不能在没有说明的情况下直接宣称已实现二阶物理积分。

### 7.1.3 哪些前提还没有完全确认

以下事项需要在实验前确认，否则会影响方法是否成立：

| 待确认点 | 为什么关键 | 建议验证 |
|---|---|---|
| token/block representation 是否可读出 benchmark 状态 | LaWM 的 `q` 不默认是真实坐标，投影头输出必须由数据标签和 benchmark 决定 | schema audit、Linear/MLP probe、事件 probe |
| 仿真数据是否提供统一坐标和速度 | Phyco、TDW、Physion、CLEVRER 的状态字段和相机坐标未必一致 | 先做数据 schema audit，不直接承诺可合并 |
| LaWM 的 DEL transition 是否值得升级为主机制 | 无外力、耗散、接触和形变场景不满足同一个 unforced variational assumption，且需要二阶状态 | 先用 projector + 条件损失跑通，再在 physics-clean 子集做 DEL residual/solver 消融 |
| 能量/动量是否可作为辅助 loss | 掉落、摩擦、碰撞、外力和非弹性接触会改变机械能 | 由数据字段建立 `M_conservative`，先作为诊断，再决定是否反传 |
| 二阶 latent transition 如何初始化 | LaWM 需要 `q_{k-1}, q_k`，PhiZero 原生 reasoner 只有首帧和 caption | 仅在 DEL 增强实验中比较双观测初始化、相邻 token block 初始化和学习初速度 |
| render → re-encode 是否值得 | 它会重新引入 decoder 成本和渲染误差，也可能改变 token 分布 | 必须和 token-only rollout、hidden-state carry rollout 对照 |
| PhiZero 原训练集是否可获得 | 公开代码不重发 5M reasoner 训练集；重新训练可能只能用自备 JSONL | 先记录可下载数据、许可证和存储预算 |
| PhiZero passive baseline 如何设 prompt | 原 Reasoner 输入包含 caption/action intent；纯被动预测需要固定 prompt 规则 | 在 matched-data 里固定 caption 生成策略或使用 GT caption |
| 16--32 秒是否能公平评价 | PhiZero 原生窗口是 4 秒；长序列需要滑窗、重渲染或重编码 | 报告 4s、8s、16s、32s 分段指标和计算成本 |

### 7.1.3 推荐保留与暂缓的模块

首轮主方法保留：

- PhiZero tokenizer/decoder 冻结；
- Reasoner 上增加层级离散 MCP heads；
- probe 决定连续 generalized coordinate 到 benchmark 状态的 projection head；
- 状态 grounding、block 边界连续性和条件化 energy/momentum loss；
- token-only、hidden-state carry、render-reencode 三种 rollout 对照；
- 4s 到 32s 的长度课程和外推评测。

LaWM 风格的 latent Lagrangian、DEL residual 和有限步 differentiable solver 放入后续增强/消融；只有 solver 真正产生下一状态时，才使用“hybrid LaWM variant”的表述。

暂缓进入主方法：

- DPO 偏好优化：作为二阶段失败兜底，不能与 MCP 和物理残差同时首次启用；
- Q-Former LoRA：只有 probe 失败且 Reasoner hidden state 也不可读时再启用；
- 无条件严格能量守恒：只作为可满足条件的仿真子集诊断，不能当成所有物理场景的统一 loss；
- “第一个” novelty：等 related work 和 scoop check 后再决定措辞。

### 7.2 核心、支撑项和 tricks

| 组件 | 定位 | 首轮处理 |
|---|---|---|
| 离散 transition physical token | 核心表示 | 必做；与连续 latent、pixel 和 object-state baseline 对照 |
| Hierarchical multi-horizon token prediction（MCP/MTP） | 核心训练目标 | 必做；高层 chunk CE + 低层 block CE |
| Projector + observable physics constraints | 物理一致性核心支撑 | 必做；状态/连续性/条件化能量约束，报告长程稳定性 |
| LaWM-style variational transition | LaWM 借鉴的增强机制 | 后续消融；只有 solver 真正生成下一状态时才作为 hybrid variant |
| Persistent physical memory | LaWM context/实现机制 | 必做；记录跨 block 状态、阶段和不确定性 |
| Boundary/composition/global consistency | 物理一致性辅助机制 | 必做；验证窗口拼接和递归组合 |
| Projection/state grounding | 可观测性支撑项 | 由 benchmark 标签决定输出变量和监督形式 |
| Scheduled sampling | 训练稳定性消融 | 与 flat autoregressive baseline 分开比较 |
| 历史 token 加噪 | 鲁棒性 regularizer | 后加；只在历史误差敏感实验中启用 |
| 熵正则/码本利用率约束 | 防离散 token 坍缩 | 仅用于离散 tokenizer，并报告 perplexity/usage |
| RAG 相似轨迹检索 | 外部记忆增强 | 作为 baseline/消融，不能算主创新 |
| token 重复惩罚 | 解码技巧 | 只有出现重复退化时才加 |
| 能量/动量损失 | 域特定物理先验 | 只在有质量、速度、接触和外力定义的仿真数据上启用 |

首轮实验只保留一条清晰增量链：`local AR → hierarchical MCP → MCP + projector/state grounding → MCP + conditional smoothness/physics → optional LaWM DEL ablation`。scheduled sampling、历史噪声、RAG、熵正则和 render-reencode 作为单独消融，不能与主方法同时无控制地堆叠。

### 7.2.1 可选增强的边界

| 借鉴方法 | 可能接入位置 | 当前定位 |
|---|---|---|
| EAGLE 式 hidden-state prediction | MCP head 同时预测未来 hidden state | 只做 hidden-space baseline，不能替代 token CE |
| Transformer-XL 式 memory | chunk 之间传递 detached hidden state | 与 token-only 对照，观察是否减少边界跳变 |
| 长度课程 | 4s → 8s → 16s → 32s | 训练策略，不能单独算 novelty |
| DPO/偏好优化 | 仿真器生成守恒/漂移 rollout pair | 二阶段兜底，需要可靠的偏好构造和 KL 控制 |
| Q-Former LoRA | tokenizer 物理信息不足时 | 只有 probe 与 Reasoner hidden state 都失败才启用 |

### 7.3 物理一致性损失与诊断的边界

主线的物理约束作用在 projector 输出的可观测状态和 block 边界：

```text
q_k = P(e_k)
v_k = (q_k - q_{k-1}) / h
L_smooth = ||q_{k+1} - 2q_k + q_{k-1}||²
L_energy = M_conservative · drift(E(q_k, v_k))
```

这里 `q` 是学习到的中间状态，`L_smooth` 只表达轨迹连续性，`L_energy` 只在数据条件允许时启用。能量约束可以降低漂移，但不能单独保证正确的物理转移，也不能在存在外力、摩擦或非弹性碰撞时强行设成零漂移。

LaWM 增强分支才使用变分 transition：

```text
L_d(q_k, q_{k+1}; η)
R_DEL = D2 L_d(q_{k-1}, q_k; η) + D1 L_d(q_k, q_{k+1}; η)
q̂_{k+1} = Solve_N(q̂_{k-1}, q̂_k; η)
```

在这个增强分支中，`L_DEL` 才是作用量驻定条件；如果下一状态仍然由普通 Reasoner 产生，`L_DEL` 只是 auxiliary residual，不能称为完整 LaWM transition。只有在有限步 DEL solver 产生 `q̂_{k+1}`、再由 token head 生成离散 physical language 时，才使用“hybrid LaWM variant”的表述。

投影头和能量项承担不同角色：

- projection head 将 generalized coordinate 接到 benchmark 可测状态/事件；
- `L_state` 在有 GT 状态时提供 grounding；
- `L_energy` 只在 `M_conservative=1` 的近似封闭仿真片段启用；
- `EnergyDrift`、`R_DEL`、PIS、state RMSE 和 event metrics 分开报告。

Latent energy 与 simulator physical energy 不能混为一谈：

```text
E_latent(q,v) = 0.5 * vᵀ Mθ(q,η) v + Vθ(q,η)
E_phys = kinetic + potential + contact/work terms
```

真实视频没有可靠质量、三维速度、接触冲量和外力时，不强行使用 `E_phys`；使用数据集提供的轨迹、mask、事件和物理不变量指标。

## 7.4 Benchmark 分工与主结果边界

截至 2026-10-07，三个新增 benchmark 应放在不同层级，不能合成一个总分：

| Benchmark | 主要测量 | WAM 的使用方式 | 当前阶段定位 |
|---|---|---|---|
| **ChronoPhyBench** | 历史视频条件下的 next-state frame selection、chronological sorting、standard/hallucination QA | 用于外部视觉物理理解验证；若模型只输出 token/state，需要增加一个轻量 adapter 或把 MuJoCo 轨迹转成同格式的候选帧 | **次主评测** |
| **Morpheus** | 生成视频是否满足守恒量和运动方程，提供 dynamical score、physical invariance 等物理知觉指标 | 只对 renderer 输出或 state-to-video 审计输出使用；token-only 主模型不能直接声称通过 Morpheus | **渲染审计/物理诊断** |
| **WorldOdysseyBench** | 交互式 world model 的逐帧动作、视觉漂移、可控物理和记忆 | 先借用 segment drift、physics/3D consistency 和 memory 的拆分思想；动作接口、WASD 连续交互和 memory protocol 留到控制阶段 | **后置控制评测** |

这里统一使用 **WorldOdysseyBench** 这个项目名称。当前公开摘要描述的是 600+ 个测试案例、自然/城市/室内场景、第一/第三人称视角和 10--60 秒 WASD 连续交互；实验记录仍需锁定论文版本和下载日期。

当前阶段的主结果仍应来自自建 MuJoCo 状态真值和长时域 rollout：`state error`、`event F1`、`event time error`、`boundary jump`、`energy/momentum drift`、`calibration` 和推理成本。ChronoPhyBench 用于验证模型是否能从视觉历史判断下一状态，Morpheus 用于验证可选渲染是否违反物理，WorldOdysseyBench 不作为无动作模型的主榜单。

### 7.4.1 六个 benchmark 的适配度排序

| 优先级 | Benchmark | 对当前 WAM 的最合适用途 | 主要原因 |
|---:|---|---|---|
| 1 | **ChronoPhyBench** | 当前无动作主线的外部视觉评测 | 直接测历史视频到下一状态和多帧时间顺序，和被动 physical dynamics 最接近 |
| 2 | **Morpheus** | projector/energy loss 的渲染审计 | 用守恒量和运动方程评分，物理诊断最硬，但要求视频或 state-to-video 输出 |
| 3 | **Physics-IQ Verified** | 和 PhiZero 的公平视频 baseline | PhiZero 已有公开结果，适合验证长程增强是否损害原有视频物理能力 |
| 4 | **WorldOdysseyBench** | Future 动作条件和长时域控制 | 长度和稳定性最符合最终愿景，但需要 WASD/action 接口，当前被动模型不能直接参赛 |
| 5 | **PhyGround** | 按物理定律拆分的 renderer 诊断 | 13 条物理定律和逐定律指标有解释性，但主要是生成视频 + judge |
| 6 | **WorldModelBench** | 通用 world-model 兼容性补充 | 覆盖面广、PhiZero 可对齐，但物理指标较粗，不能替代状态真值和长 rollout |

因此当前论文不应选择一个 benchmark 代替全部评价，而应采用：

```text
主结果：MuJoCo state/event long-horizon
外部被动评测：ChronoPhyBench
物理约束审计：Morpheus
PhiZero 视频对照：Physics-IQ Verified + PhyGround + WorldModelBench
后置控制评测：WorldOdysseyBench
```

## 7.5 MuJoCo-WAM v1 的数据定位

MuJoCo 数据集首轮服务三个目标：训练 projector 和条件化物理约束、提供 16--32 秒长时域状态真值、构造不依赖未来像素的主评测。模型输入仍然是被动观测历史，不输入动作；数据包可以保存 `ctrl`、外力和接触力，但这些字段只用于审计、能量 mask 和后续控制阶段。

建议采用三档规模：

| 规模 | 轨迹数 | 用途 | 视频保存策略 |
|---|---:|---|---|
| Smoke | 3,600（12 类 × 300） | 跑通生成、token 对齐、probe 和 loss | 全部低分辨率或只保存状态 |
| **Paper v1** | **36,000（12 类 × 3,000）** | 主训练、验证、长时域和跨参数泛化 | 状态全量保存；约 25% 渲染视频，另保留 3,000 条 PhiZero 分辨率对照片段 |
| Stress/OOD | 6,000 额外轨迹 | 未见质量/摩擦/重力/拓扑/相机组合 | 以状态和关键帧为主 |

Paper v1 的每条轨迹建议模拟 32 秒，发布 8 FPS 的观测序列，并保留更高频的状态采样。这样可得到约 9.2M 个 8 FPS 帧；训练窗口从同一轨迹切出，但 train/val/test 必须按轨迹、模板和参数组合切分，不能把相邻窗口分到不同 split。

## 8. 必须保持诚实的边界

- `physical language` 在 PhiZero 中是学习到的离散状态转移符号，不等于可读的自然语言，也不等于已知物理方程；
- 不把重建质量、视频观感或 token 可视化直接当成世界模型正确性的证据；
- 必须报告遮挡、接触、长时域、窗口边界、分布外外观和事件组合的失败案例；
- 如果代码仓库或权重未公开，项目只能做论文结构分析和小规模概念验证，不能声称完成 PhiZero 数值复现；
- 所有候选 idea 都需要通过与 pixel/latent/object-state baseline 的等算力对照和反事实测试。

## 9. 待确认事项

相关路线的调查记录见 [`references/manifests/literature_scan.md`](../references/manifests/literature_scan.md)，其中区分了 latent video prediction、latent-action world model、事件/物理推理、视频 tokenizer 和 PhiZero physical language。

1. `WAM终局` 已初始化独立本地 Git；是否需要绑定并推送到已有远端；
2. 首轮环境优先选 LIBERO/BridgeData 类机器人数据、视频物理数据，还是先用可控的 2D/小型仿真环境；
3. 目标是先做 idea 设计与 smoke test，还是立即进入完整训练和论文复现；
4. 可用 GPU、存储和网络代理条件决定是否下载大模型、视频数据和 Wan/PhiZero 相关权重。
