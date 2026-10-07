# 执行计划与 Git 操作：从 PhiZero 代码审计到非像素世界模型验证

**版本**：Research execution draft v0.4
**日期**：2026-10-07
**执行原则**：先文档、再代码审计、再最小可运行实验；每一步都有完成勾选、验收条件、commit 和 push 命令。没有远端时只做本地 commit，不声称已经 push。

## 1. Git 仓库与恢复规则

目标目录：`/home/bubble/类脑计算/WAM终局`

当前目录为空且未检测到可用 Git 元数据。若确认独立管理，初始化：

```bash
cd /home/bubble/类脑计算/WAM终局
git init
git checkout -b main
git add doc
git commit -m "docs: define WAM physical-language research plan"
```

配置真实远端后再执行：

```bash
git remote add origin <真实的远端仓库地址>
git push -u origin main
```

后续每步通用流程：

```bash
git status
git add <本步产生的代码、配置、文档和小型清单>
git commit -m "<type>: <说明本步完成的事实>"
git push  # 只有已配置 origin 且用户希望同步时执行
```

不要提交模型权重、原始视频、仿真数据、密钥、代理配置、日志和大规模输出。没有远端地址时，`git push` 明确记为“待配置”，不伪造结果。

## 2. 阶段总览

| 阶段 | 目标 | 主要产物 | 完成 |
|---|---|---|---|
| P0 | 文档与研究边界冻结 | `doc/*.md`、Git 初始 commit | [x] 文档已生成 |
| P1 | PhiZero 代码和论文接口审计 | repo 状态、依赖、入口、许可证报告 | [ ] |
| P2 | 最小 tokenizer/表示 smoke test | token 统计、重建/预测 sanity check | [ ] |
| P3 | 不渲染像素的 dynamics baseline | latent/物理语言 multi-step 预测 | [ ] |
| P4 | HG-PLM 长程 Reasoner | 层级 MCP、LaWM transition、状态 grounding、长 rollout | [ ] |
| P5 | 结构化事件图扩展 | graph/event ablation | [ ] |
| P6 | 可选 renderer 审计闭环 | 失败样本渲染、跨外观/embodiment | [ ] |
| P7 | 论文级对照和结论 | 等算力表、失败案例、idea 选择 | [ ] |
| Future | 动作条件控制 | DAgger、MPC、机器人任务 | 后置 |

## 3. P0：文档先行（当前步骤）

- [x] 生成项目背景与研究定位；
- [x] 生成技术栈与张量/接口约定；
- [x] 生成本执行计划，写明每步验收、commit 和 push；
- [x] 明确首选 idea：physical-language-only + long-sequence global consistency；
- [x] 记录 PhiZero 的限制：经验性符号、视觉不可观测状态、固定时长和算力成本；
- [x] 已初始化独立本地 Git 并提交文档和审计清单；
- [ ] 与用户确认真实远端、首轮环境和算力预算。

### P0 验收

三份文档内容互相引用；没有把 PhiZero 的论文结果写成已完成的本地复现；所有未确认事项都被列出。

### P0 Git

```bash
cd /home/bubble/类脑计算/WAM终局
git init
git checkout -b main
git add doc
git commit -m "docs: define WAM physical-language research plan"
git remote add origin <真实远端地址>  # 仅在用户提供后执行
git push -u origin main               # 仅在 origin 存在后执行
```

## 4. P1：下载并审计 PhiZero 代码

- [ ] 目标位置创建为 `references/repos/PhiZero`；
- [ ] 使用用户给出的官方 URL clone 或下载源码；
- [ ] 记录实际 commit、远端 URL、下载时间和代码大小；
- [ ] 读取 README、依赖文件、训练/推理入口、数据下载说明和许可证；
- [ ] 检查仓库是否真的包含 tokenizer、reasoner、decoder 或仅有项目页面；
- [ ] 与本地 `PHIZERO.pdf` 的方法章节逐项对照；
- [ ] 若网络失败，先使用 `127.0.0.1:7897`，再使用国内镜像；写入失败原因和替代来源；
- [ ] 生成 `references/manifests/repo_status.md` 和 `references/manifests/versions.yaml`。

### P1 验收

能够回答：仓库是否可运行、需要什么 Python/显卡/权重、哪些数据公开、哪些训练步骤无法在本地复现、官方代码与论文是否一致。不能运行时输出具体错误，而不是标记为通过。

### P1 Git

```bash
git add references/manifests
git commit -m "audit: record PhiZero repository and paper interfaces"
git push
```

## 5. P2：最小 tokenizer 与表示 smoke test

- [ ] 选 8--32 个短视频或仿真轨迹，固定随机种子和 manifest；
- [ ] 若官方 tokenizer 依赖大模型，先写轻量替代 encoder/Q-Former/FSQ，用于接口验证；
- [ ] 验证相邻状态 transition token 的顺序、长度、词表利用率和重复率；
- [ ] 比较 same-scene/different-motion 与 different-scene/same-motion 的 token 距离；
- [ ] 可选地使用 first frame + token 做低成本重建，确认 token 未只编码静态外观；
- [ ] 保存 token、输入、版本和可视化，不把重建观感当作最终结论。

### P2 验收

至少一个合成或仿真单元测试通过：交换首帧而保留 transition representation 时，运动模式应大体保留；交换 token 而保留首帧时，未来状态应发生可测变化。

### P2 Git

```bash
git add src/tokenizer scripts configs outputs/tokenizer_smoke
git commit -m "feat: add tokenizer and transition representation smoke test"
git push
```

## 6. P3：先预测状态，不生成像素

- [ ] 实现 `dynamics.predict(observation_history, horizon)`；
- [ ] 建立 pixel/video、VAE latent、物理语言和 consequence-only 四个 baseline；
- [ ] 训练短 horizon，再逐步增加 rollout 长度；
- [ ] 记录 multi-step state error、事件 F1、uncertainty calibration、延迟和显存；
- [ ] 对离散物理语言增加 local token、token block、persistent memory 和 global consistency 对照；
- [ ] 用状态/事件 probe、局部 token block 干预、组合一致性和跨外观保持来评价物理可解释性；
- [ ] 进行随机 token、时间错位 token、历史截断和遮挡输入的反事实测试；
- [ ] 检查模型是否只预测外观变化而没有预测物理状态和事件；
- [ ] 把失败样本按 perception、representation、dynamics、long-horizon composition 分类。

### P3 验收

在同一数据和算力预算下，physical-language-only 至少能在一个未来状态、物理事件或长时域稳定性指标上达到 pixel/continuous-latent baseline 的可比水平，同时显著降低 rollout 成本；若没有，保留负结果并停止无依据扩展。

### P3 Git

```bash
git add src/dynamics src/evaluation configs outputs/dynamics_smoke
git commit -m "feat: evaluate non-pixel future-state prediction"
git push
```

## 6.1 阶段一：被动无像素 rollout 与 PhiZero 的公平比较协议

当前阶段暂不研究控制，不输入动作，也不评价动作选择。PhiZero 的完整系统是“物理语言 reasoner + 视频 diffusion decoder”，适合评价未来视频生成；本项目的 Passive Physical Dynamics 不生成未来视频，主要评价被动未来状态预测、物理事件预测、长时域稳定性和推理效率。因此阶段一的主问题固定为：

> 在相同观测历史下，模型能否不生成未来像素而准确预测未来物理状态和事件，并以更低成本保持长时域稳定？

### 统一输入和接口

阶段一所有模型在时刻 `t` 接收相同的当前观测或观测历史 `o_{\leq t}` 和预测 horizon `H`，不提供动作输入。

```text
观测历史 o_{≤t}
        ↓
未来物理状态/transition token 预测
        ↓
状态、轨迹和物理事件评价
```

我们的模型输出 `z_hat_{t+1:t+H}` 或状态/事件后果；PhiZero baseline 可以继续通过其 decoder 生成未来视频。评价时使用统一的观测历史、未来状态真值、事件标签和外部跟踪器，不能把控制回报或动作排序提前混入阶段一结论。

### 阶段一必须报告的模型

| 模型 | 未来像素 | 重新读取真实观测 | 用途 |
|---|---:|---:|---|
| PhiZero passive/adapted | 是 | 可选 | 生成式物理预测基线 |
| Ours-passive-open-loop | 否 | 否 | 测试 Passive MCP 单独的多步预测能力 |
| Ours-passive-reanchored | 否 | 是 | 测试 MCP + 观测重锚定 |
| Pixel/video predictor | 是 | 可选 | 像素预测参考基线 |

必须同时画出开放环和重新观测版本随 horizon 增长的误差曲线。若重锚定有效，`Ours-passive-reanchored` 的误差累积应低于 `Ours-passive-open-loop`。这里的“闭环”只表示预测、观测、重新编码，不表示动作控制。

### 阶段一评价指标

1. **未来物理预测**
   - 位置、速度和相对距离误差；
   - 接触、碰撞、掉落、停止等事件 F1；
   - 事件发生时间误差；
   - 长时域误差增长；
   - 不确定性校准。

2. **效率和系统代价**
   - 单段未来预测延迟；
   - 每段未来轨迹的推理时间；
   - 显存峰值；
   - 单次 rollout 的 FLOPs 或 GPU 时间。

不要把我们的 latent/token CE 直接与 PhiZero 的视频 FVD 当作同一指标。对于物理预测，优先使用仿真器状态和外部视觉跟踪结果；对于视频生成，单独报告视频基线指标，或把 renderer 作为可选审计模块。

### PISA、仿真和机器人数据的分工

当前阶段以 PISA/Kubric 和其他被动视频为主，用于掉落、碰撞、轨迹和视觉物理表示测试。MuJoCo、ManiSkill、DM Control、LIBERO 和 RoboMimic 的动作条件数据暂存为后续控制阶段，不纳入阶段一主结论。

### 公平性设置

至少报告两种结果：

- **Matched-data**：PhiZero 和本方法使用相同数据、相同观测历史、相同划分、相同 horizon 和尽可能匹配的训练预算，用于比较方法本身；
- **Official-pretrained**：直接使用 PhiZero 官方权重与本方法比较，明确标注这是完整系统能力比较，不能解释为严格架构公平比较。若将 PhiZero 用于纯被动预测，需要记录动作意图如何设置，不能把原生动作条件模型直接当作无条件 baseline。

消融顺序固定为：

```text
PhiZero passive/adapted
Pixel/video predictor
Ours one-step passive
Ours hierarchical MCP
Ours hierarchical MCP + LaWM transition
Ours hierarchical MCP + LaWM + projection/state grounding
Ours full model + conditional energy
Ours full model + observation re-anchoring
```

主张应限定为：PhiZero 可能更适合未来视频生成和视觉呈现；Passive Physical Dynamics 若在物理事件、长时域稳定性和推理效率上更好，则说明物理 token 可以在不生成未来像素的情况下承担被动预测任务。动作条件预测、DAgger、MPC 和机器人控制作为后续阶段单独验证。

## 6.2 推荐训练配方（阶段一：被动预测）

### Stage 0：数据与表示准备

- [ ] 用仿真轨迹建立 `(observation_history, state, event, next_state)` manifest；
- [ ] 用 PISA/Kubric 视频建立外观变化与掉落运动的预训练/验证 split；
- [ ] 训练或冻结 transition encoder，检查同一运动换首帧、同一首帧换运动的表示距离；
- [ ] 离散 token 必须记录 codebook usage、perplexity、重复率和 scene/motion probe。

### Stage 1：教师强制的被动多步预测

定义长度为 `H` 的转移目标 `z_{t+1:t+H}`，先用真实历史和观测序列做 teacher forcing；当前阶段没有动作条件：

```text
L_MCP = Σ_h w_h · CE(z_{t+h}, pθ(z_{t+h} | o_{≤t}, z_{t+1:t+h-1}))
```

连续 latent 或结构化状态可以把 `CE` 换成 masked regression/event loss。必须与 one-step baseline 对照，不能只展示训练 loss。

### Stage 1b：物理语言长序列扩展

- [ ] 将单个序列切成固定长度或事件对齐的 token blocks；
- [ ] 增加 `persistent physical memory`，每个 block 结束时更新阶段、运动趋势、接触关系和不确定性摘要；
- [ ] 增加 next-1/next-2/next-3 block 预测，作为离散 physical-language 版本的 MCP/MTP；
- [ ] 增加 boundary consistency 和 transition composition loss；
- [ ] 通过 token-to-observable bridge 计算轨迹连续性、事件顺序和接触持续性约束；
- [ ] 只在仿真状态足够完整时加入条件化能量/动量残差；
- [ ] 与“只增加序列长度”的 flat autoregressive baseline 对照，验证收益是否来自全局机制。

### Stage 2：处理 rollout 漂移

- [ ] 先做 scheduled sampling：逐步提高历史预测 token 的比例；
- [ ] DAgger 暂不启用，保留到动作条件控制阶段；
- [ ] 不要在首轮同时使用多种 rollout 扰动，否则无法知道收益来自哪里；
- [ ] 历史 token 加噪作为 scheduled sampling 之后的鲁棒性消融。

### Stage 3：被动重新观测评估

模型不执行动作，只在预测若干步后重新读取真实观测，再预测下一轮未来状态。报告开放环与重新观测版本的状态误差、事件 F1、长时域漂移、窗口边界跳变和每段推理延迟。RAG 只作为无检索/有检索对照，不进入主模型定义。

### Stage 4：LaWM transition、状态 grounding 和条件化物理约束

- [ ] 离散 tokenizer 才启用熵正则，并报告词表利用率而非只报总 loss；
- [ ] 将 token block/chunk embedding 投影为连续 generalized coordinate；
- [ ] 实现 LaWM 风格的离散 Lagrangian、DEL residual 和有限步可微 solver；
- [ ] 使用 `L_lat + L_DEL + L_mass` 训练 variational transition，使物理结构参与 rollout 规则；
- [ ] 在 benchmark 提供状态标签时启用 `L_state`，不得预先假设统一的 `(x,y,vx,vy)`；
- [ ] 仿真器有质量、速度、接触和外力时，再通过 `M_conservative` 启用能量/动量辅助 loss；
- [ ] 对摩擦、碰撞、动作做功和外力显式建模，不能默认“总能量恒定”；
- [ ] 单独记录 `DEL residual`、latent energy drift、simulator physical energy drift、PIS 和 state RMSE；
- [ ] token 重复惩罚只在固定 horizon 自回归解码出现重复时启用。

阶段一关键消融：one-step passive、flat long AR、hierarchical MCP、MCP + LaWM transition、MCP + projection/state grounding、MCP + LaWM + projection、full model + conditional energy、persistent memory、transition composition、scheduled sampling、history-token noise、observation re-anchoring、无 first-frame condition、无离散瓶颈、无结构化关系、无 uncertainty head、无 renderer、短/长 horizon、随机/错配 token、遮挡与分布外观测。DAgger、action condition 和 MPC 属于后置控制阶段。

动作条件的 DAgger、MPC 和控制回报评价保留到 Future 阶段，不作为当前训练配方的必需项。

## 6.3 预研确认清单：哪些事还没讨论充分

以下事项必须在真正训练前确认，避免把不成立的假设写进主方法：

- [ ] **数据可得性**：PhiZero 原 5M reasoner 训练集不随代码重发；若无法获得，只能用自建 captioned JSONL 或公开仿真/视频子集继续训练；
- [ ] **状态标签字段**：逐一审计 Phyco、TDW、Physion、CLEVRER、IsaacLab/MuJoCo 输出，确认是否有统一坐标、速度、质量、接触、事件和相机参数；
- [ ] **token 对齐方式**：确认 256 个 physical tokens 与 33 帧/8 个 latent transition 的对应关系，不能默认一个 token 对应一个物理时间步；
- [ ] **对象对齐方式**：明确使用单主对象、固定 simulator object id，还是 permutation-invariant set prediction，避免多物体 token 被错误监督；
- [ ] **MCP 推理含义**：区分 training-only MCP、parallel proposal + verification 和普通递归 AR，不能把训练多步监督直接写成推理加速；
- [ ] **chunk 时间尺度**：从实际 temporal stride 推导 transition block 的秒数，不能直接假设 0.25 秒或 token-level 物理时间；
- [ ] **prompt 策略**：PhiZero Reasoner 需要 caption/action intent；纯被动长预测必须固定 prompt 生成规则，避免 baseline 因 prompt 不公平；
- [ ] **LaWM transition 适用条件**：确定 generalized coordinate 的层级、二阶 rollout 的初始两状态、context `η` 的保持范围、DEL solver 迭代数和稳定性诊断；
- [ ] **物理监督适用条件**：由 benchmark schema 决定 `L_state`、事件 loss 和 `L_energy`，使用 `M_state`/`M_conservative`，不能预先假设所有数据都有 `(x,y,vx,vy)`；
- [ ] **闭环成本**：render-reencode 必须报告延迟、显存和 round-trip token drift，并与 token-only/hidden-state carry 对照；
- [ ] **novelty 检索**：对“离散 physical token + long-horizon + explicit physics residual”做 scoop check 后再使用“第一个”表述；
- [ ] **算力预算**：Reasoner full SFT、MCP head 训练、decoder inference 和 16--32s rollout 的 GPU/存储预算需要单独估算。

### 数据角色与标签可得性

| 数据 | 当前已确认的信息 | 可用于 | 训练前必须确认 |
|---|---|---|---|
| PhiZero 原 5M clips | 论文报告由真实视频和仿真视频组成；公开代码不重发完整 Reasoner JSONL | 继续预训练/复现原分布（若数据可得） | 下载权限、许可证、caption 和 token JSONL |
| Phyco 126K | PhiZero 论文把它列为 tokenizer SFT 的仿真来源 | token/block 表示、LaWM 状态对齐候选 | 是否公开 GT position/velocity、object id、相机坐标、长序列相邻片段 |
| TDW / Physion / Physion++ | 可作为独立仿真/视觉物理数据源 | 投影头补充、跨域测试 | 状态字段、坐标系、对象追踪和 train/test split |
| CLEVRER / ComPhy | 碰撞和事件推理数据 | event probe、碰撞专项 | 是否有足够长轨迹和可对齐状态 |
| IsaacLab / MuJoCo 自建轨迹 | 可直接导出 qpos/qvel/contact/force | 长轨迹训练、LaWM transition、状态 grounding、条件化物理残差、长度外推 | 场景、质量、外力、摩擦、初始两状态和渲染设置固定 |
| Physion / Physics-IQ / PhyGround / WorldModelBench | 主要是评测或理解基准 | 保持短程能力和物理事件评测 | 是否能把 token prediction 映射到统一指标 |

“PhiZero 原训练集”和“Phyco 有 GT 状态”在当前仓库审计中都不能自动视为已获得；在数据 manifest 完成前，只能写成计划用途，不能写成已可训练资源。

## 6.4 更新后的开工顺序

### Step 1：probe 与数据 schema audit

- [ ] 冻结 tokenizer，在 Phyco/TDW/Physion/CLEVRER 的小样本上导出 physical tokens；
- [ ] 建立 `(token/block_embedding, hidden_state, benchmark_state/event)` 对齐表；
- [ ] 根据 benchmark schema 选择 Linear/MLP probe，不预先固定 `(x,y,vx,vy)`；
- [ ] 做事件 probe，检查 token 是否包含 collision、contact、bounce、stop 等事件阶段；
- [ ] 按 probe、跨域泛化和标签可得性决定投影头 A、B 或 fallback；
- [ ] 确认 LaWM 二阶 transition 的初始两状态和 generalized coordinate 的层级。

### Step 2：离散版 MCP baseline

- [ ] 先实现最小 K=4 multi-token/block prediction；
- [ ] 再加入 token corruption、multi-layer feature fusion 和 chained MCP heads；
- [ ] 与 one-step local AR、flat long AR、final-layer-only head 对照；
- [ ] 暂不加入物理 loss，先确认 MCP 本身能降低长 horizon token/state/event error。

### Step 3：投影头与条件化物理残差

- [ ] 在 hierarchical MCP 上先加入 LaWM-style generalized coordinate、discrete Lagrangian 和 DEL solver；
- [ ] 训练 `L_lat + L_DEL + L_mass`，检查二阶 rollout 是否降低长程 drift；
- [ ] 再加入 projection/state grounding，使用 benchmark 提供的状态/事件标签；
- [ ] 最后在 `M_conservative=1` 的仿真子集加入 conditional energy/momentum loss；
- [ ] 对比 token embedding、block embedding、Reasoner hidden state 作为 `q` 的输入；
- [ ] 分别报告 token CE、chunk/block CE、DEL residual、latent energy drift 和 benchmark physical metrics。

### Step 4：长 rollout 与重锚定

- [ ] 评估 token-only rollout 到 4s、8s、16s、32s；
- [ ] 加入 hidden-state carry，评估是否减少跨 chunk 边界跳变；
- [ ] 加入 render-reencode，记录 round-trip token drift、延迟、显存和视频质量；
- [ ] 明确 render-reencode 是上界/诊断还是默认推理模式。

### Step 5：主实验与消融

- [ ] 主配置：PhiZero original、hierarchical MCP、+LaWM transition、+LaWM+projection、full model、full model+reanchor；
- [ ] MCP ablation：corruption rate、fusion layers、head depth、initialization；
- [ ] LaWM ablation：无 DEL transition、post-hoc trajectory refinement、solver iterations、无 context `η`、无 mass conditioning；
- [ ] projection/physics ablation：A vs B、`λ_state`、`λ_energy`、token/block embedding vs hidden state；
- [ ] 主曲线：trajectory/state error、event F1、boundary jump、DEL residual、latent/physical energy drift、PIS、latency vs horizon；
- [ ] 保原榜：Physics-IQ Verified、PhyGround、WorldModelBench，确认短程能力没有明显下降。

## 6.5 PISA Experiments 的正确角色

PISA 很适合验证“模型是否理解掉落/碰撞/运动后果”，但不能单独支撑完整的动作条件 WAM：

- `pisabench/real.zip`：361 个真实掉落视频，适合作为真实物理泛化测试；不含完整动作、质量、三维状态或 MuJoCo oracle，不能直接做 DAgger；
- `pisabench/sim.zip`：60 个仿真测试视频，含 seen/unseen object-background split 和 segmentation mask，适合作为 sim2real/组合泛化测试；
- `training_data/psft.zip` 与 `training_data/oro.zip`：用于 PISA 的视频扩散 post-training，适合作为视频/物理表示预训练或渲染对照，但需先检查是否有动作和状态字段；
- PISA 仿真基于 Kubric/PyBullet/Blender，不是 MuJoCo，因此不能把 PISA 的 mask 当成 DAgger 所需的动力学真值。

推荐数据分工：

| 数据 | 训练/评估角色 | 可支持的结论 |
|---|---|---|
| PISA/Kubric simulation | transition encoder、掉落事件预训练 | 表示能否编码运动/碰撞 |
| MuJoCo、ManiSkill 或 DM Control 自生成轨迹 | MCP、scheduled sampling、DAgger、MPC | 动作条件预测和闭环控制 |
| LIBERO/RoboMimic | 机器人视觉-动作迁移 | 操作任务的 action-conditioned 泛化 |
| PISA real | 真实视觉物理 hold-out | sim2real 的掉落/碰撞合理性 |

PISA 的视频指标主要面向生成视频。若主模型不生成像素，必须把模型输出映射到 object trajectory、contact/event 或 mask-level 预测，再报告状态/事件/规划指标；不能用视频 FVD 代替非像素世界模型评价。

## 7. Future：动作条件反事实规划和控制（后置）

- [ ] 当前阶段不执行；
- [ ] 对同一观测生成多个候选动作；
- [ ] 用真实后果、仿真器或人工小样本标签验证动作排序；
- [ ] 对目标位移、接触成立、碰撞风险、终止和不确定性分别评价；
- [ ] 比较每一步渲染视频与只预测结构化后果的延迟/显存/成本；
- [ ] 测试长时域闭环和分布外动作，而不是只看 one-step loss；
- [ ] 形成 `outputs/counterfactual_report.md`，写清哪些结果支持或反驳主假设。

### Future Git

```bash
git add src/planner outputs/counterfactual_report.md configs
git commit -m "eval: add action-conditioned consequence benchmark"
git push
```

## 8. P5：事件图/关系图扩展（可选）

- [ ] 定义最小对象、接触、支撑、相对位置和阶段事件 schema；
- [ ] 训练 graph/event dynamics，与离散 token 和连续 latent 对比；
- [ ] 做局部关系干预：只修改一条边或一个事件，检查后果是否局部且合理；
- [ ] 报告跟踪错误传播，避免把检测器错误误报为 dynamics 优势。

### P5 Git

```bash
git add src/dynamics/graph src/evaluation outputs/graph_ablation
git commit -m "feat: add structured event-graph world model ablation"
git push
```

## 9. P6：可选 renderer 和跨外观迁移

- [ ] 仅对关键帧、失败样本或审计样本启用 renderer；
- [ ] 比较同一 transition token 在不同首帧/外观下的运动一致性；
- [ ] 检查 renderer 是否掩盖 dynamics 错误；
- [ ] 若有不同 embodiment 或 sim-to-real 数据，测试 transition 表示的迁移；
- [ ] 明确 renderer 是诊断工具还是模型必需组件。

### P6 Git

```bash
git add src/renderer outputs/render_audit
git commit -m "feat: add optional transition rendering audit"
git push
```

## 10. P7：论文级汇总和停止条件

- [ ] 阶段一 baseline 使用相同观测历史、horizon、数据 split 和算力记录；Future 阶段再统一动作条件；
- [ ] 阶段一同时报告 state/event prediction、长时域漂移、窗口边界一致性、efficiency 和 calibration；Future 阶段再报告 counterfactual planning 与 closed-loop success；
- [ ] 至少保留一个失败案例集和一个负结果；
- [ ] 对主 idea 做 novelty/related-work 检索，区分 PhiZero 的直接延伸与真正的新问题；
- [ ] 若 physical-language model 在未来状态、物理事件或长时域稳定性指标上不优于 continuous-latent/pixel baseline，则停止扩展并改写研究问题；
- [ ] 生成最终 idea 选择报告和后续论文大纲。

### P7 Git

```bash
git add outputs/final_report.md doc references/manifests
git commit -m "docs: summarize WAM world-model findings"
git push
```

## 11. 当前阻塞与沟通点

- GitHub 当前在本环境中出现 DNS/网络失败，代码下载需在下一步使用代理或镜像重试；
- 论文中的训练规模（128 张 A100、约 10K 小时预训练视频）远超默认本地 smoke test，不能直接承诺完整数值复现；
- 首轮数据集、GPU、磁盘和真实远端尚未确认；
- 如果用户确认“先本地、无远端、先小规模 idea 验证”，可以直接推进 P1--P3；如果目标是论文数值复现，需要先提供服务器和数据访问条件。
