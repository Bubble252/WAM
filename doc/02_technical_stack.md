# 技术栈与实现规格：物理语言、非显式动力学与可选渲染

**版本**：Research design draft v0.4
**日期**：2026-10-07
**原则**：先冻结接口与评价，再下载大仓库、模型和数据；所有状态表示都必须能被保存、重放、干预和比较。

## 1. 目录规划

```text
WAM终局/
├── doc/
│   ├── 01_project_background.md
│   ├── 02_technical_stack.md
│   └── 03_execution_plan.md
├── references/
│   ├── papers/                 # 论文 PDF、BibTeX、SHA256
│   ├── repos/                  # PhiZero 官方代码及只读快照
│   └── manifests/              # 数据、权重、commit、许可证清单
├── src/
│   ├── tokenizer/              # 状态转移/物理语言编码器
│   ├── dynamics/               # 非显式或结构化 forward model
│   ├── planner/                # 反事实动作评价与规划
│   ├── renderer/               # 可选的未来视频渲染和审计器
│   └── evaluation/             # multi-step、干预、控制和效率指标
├── configs/
├── scripts/
├── outputs/
└── logs/
```

大模型权重、原始视频、仿真数据、训练 checkpoint 和大规模输出不提交 Git；只提交下载清单、配置、代码、指标摘要和可复现实验说明。

## 2. 参考代码和论文资产

当前已存在：

- 论文：`/home/bubble/类脑计算/参考/WAM/PHIZERO.pdf`；
- 论文首页给出的项目页：`https://Phi-Zero.github.io/`；
- 用户指定的代码地址：`https://github.com/yaoyao-jpg/PhiZero`。

目标下载位置：

`/home/bubble/类脑计算/WAM终局/references/repos/PhiZero`

下载后必须保存：

1. 实际 clone commit、远端 URL、下载时间；
2. README、依赖文件和入口脚本的审计摘要；
3. 代码许可证和模型/数据使用限制；
4. 若仓库为空、私有、与论文不匹配或网络不可达，写入 `references/manifests/repo_status.md`，不伪造成功。

## 3. 推荐软件环境

首轮以 Python/PyTorch 为主，具体版本以官方仓库实际依赖为准：

- Python 3.10/3.11（先读取 `requirements.txt`、`environment.yml` 或 `pyproject.toml`）；
- PyTorch、TorchVision、Transformers、Accelerate；
- NumPy、SciPy、scikit-learn、Pillow/OpenCV；
- Einops、Safetensors、TensorBoard/WandB；
- 若复现 PhiZero tokenizer，再审计 Wan/VAE、Diffusers、FSQ、Q-Former 和 LoRA 依赖；
- 若做仿真控制，再按环境增加 Gymnasium、LIBERO、PyBullet 或 MuJoCo；
- Git LFS 只用于确实需要拉取的小型模型资产，禁止无审计地拉取整个权重目录。

下载失败时按项目约束依次尝试：

1. `http://127.0.0.1:7897` 代理；
2. 国内镜像或 ModelScope/Hugging Face 镜像（记录替代来源）；
3. 只下载代码和小型配置，先完成接口审计；
4. 记录 URL、时间、错误、替代方案和 SHA256，不把网络失败隐藏为“已完成”。

## 4. 三种状态表示

### 4.1 PhiZero 风格离散物理语言

输入视频先通过时空编码器得到 latent state `x_i`。共享 transition-level Q-Former 读取相邻状态 `(x_i, x_{i+1})`，再通过投影与 FSQ 得到离散序列 `z_i`。首帧提供静态外观，`z` 主要承担状态变化。

建议保留以下张量契约：

```text
x:       [B, t, C, h, w]       # 时空编码器状态
q_i:     [B, M, D_q]            # 每个相邻状态对的 transition queries
z:       [B, N]                 # FSQ 离散物理语言
z_emb:   [B, N, d]              # 给 reasoner/decoder 的 embedding
```

首轮不要默认采用论文中的 33 帧、32 transition symbols 或 `(8,5,5,5,5,5)` FSQ 配置；这些是待审计配置，应根据显存和小数据实验缩放。

### 4.2 长时序物理语言模型

阶段一的核心不只是把 `N` 变大，而是让多个局部 transition token block 能够组合成一条稳定的长轨迹。推荐使用三层接口：

```text
局部层：z^local_{k,1:B}       # 第 k 个时间块内的 transition tokens
记忆层：m_k                    # 跨块保存的 physical state summary
全局层：y_{1:kB}               # 轨迹、事件和可观测物理量
```

递归更新可以抽象为：

```python
z_block = local_predict(context, memory, horizon=B)
memory_next = memory_update(memory, z_block)
observable = token_to_observable(z_block, memory_next)
```

`memory` 不能只是上一个 token 的 embedding；它应尽量编码阶段、运动趋势、接触关系、可见性和不确定性。`token_to_observable` 不要求恢复完整像素，可以只输出轨迹、速度方向、接触/碰撞状态和事件阶段，供全局一致性损失和评价使用。

离散 token ID 本身没有可计算的物理距离，因此物理约束不应直接施加在整数 ID 上。建议通过三条路径建立约束：

1. **边界一致性**：相邻 block 在位置、速度趋势、接触关系和事件阶段上连续；
2. **转移组合一致性**：先预测 `A` 再预测 `B` 与直接预测组合块 `A⊕B` 的 observable 结果一致；
3. **物理可观测量残差**：在有标注或仿真真值时，对轨迹连续性、事件顺序、接触持续性以及条件化的能量/动量残差进行约束。

```text
L = L_local
  + λ_mtp L_multi_horizon
  + λ_mem L_state_summary
  + λ_comp L_transition_composition
  + λ_phys L_observable_physics
```

其中 `L_multi_horizon` 对应 Next Forcing/MTP 的多步监督；`L_state_summary`、`L_transition_composition` 和 `L_observable_physics` 才是物理语言长序列方向的核心新增接口。

### 4.3 连续非显式 latent

使用 VAE/时空 encoder 或 JEPA-style predictor 获得 `s_t`，由 forward model 预测 `s_{t+1:t+H}`。训练目标可组合：

```text
L = L_pred + λ_inv L_inverse + λ_contrast L_temporal
    + λ_policy L_policy_consistency + λ_uncertainty L_calibration
```

连续 latent 不要求人类可解释，但必须通过 probe 和 intervention 检查是否包含位置、接触、速度方向和目标进展信息。

### 4.4 结构化事件/关系图

节点可包含对象类别、位置、速度、可见性和置信度；边可包含接触、支撑、包含、相对距离和遮挡；事件可包含 `approach/grasp/lift/place/release/collision`。图 dynamics 预测边和节点的增量，并输出不确定性。

此表示适合后续控制阶段，因为动作候选可以只修改局部边或事件，再进行后果预测，不需要重建整段视频。

## 5. 推理和训练接口

阶段一只实现被动观测历史到未来物理语言的预测；`planner`、动作条件接口和控制评价保留给 Future 阶段，目录结构提前保留只是为了避免后续重构。

### 5.0 Reasoner 改造接口

首轮冻结 PhiZero tokenizer、FSQ、diffusion decoder 和 Wan/VAE 基座，只改 Reasoner 与新增小模块。基础输入仍是首帧或观测历史、caption/prompt 和 physical-language token target。

```text
Frozen tokenizer/decoder
        │
        ▼
Reasoner
  ├── main AR branch: next-token CE
  ├── discrete MCP heads: next-1/next-2/next-3 block CE
  ├── physics projection head: token/hidden → observable state
  └── optional memory: previous chunk hidden state
```

离散版 MCP 不直接复制连续 flow-matching 的噪声设定，而是用 token-level corruption 作为高噪声 analogue：

```text
corruption_rate ∈ {0.0, 0.3, 0.5, 0.7}
fusion_layers = {1/4, 1/2, 3/4, final}
mcp_depths = {next-1, next-2, next-3}
loss_weights = {0.5, 0.2, 0.1}
```

这些值是首轮搜索空间，不是固定结论。必须与 flat long AR、no-corruption、final-layer-only 和 random-init head 对照。

MCP 的“多 chunk 并行”首先指训练时同时提供多个未来 block 的监督，不代表推理时自动得到同样的加速。阶段一必须分别报告：

1. **training-only MCP**：MCP head 只提供梯度，推理时丢弃；
2. **parallel proposal**：MCP head 一次提出多个未来 block，再用主 AR 分支验证或接受；
3. **普通递归 AR**：逐 token 或逐 block 生成，作为速度和准确性的基线。

如果没有 block verification 或 speculative decoding，不能把 MCP 写成推理加速方法。

### 5.0.1 Projector 与 LaWM 借鉴的物理约束

首轮将 LaWM 当作物理约束和诊断的参考，而不是直接把 DEL solver 放进主 rollout。PhiZero 的离散 token 仍然是训练目标和对外接口；Reasoner 负责预测下一段 token，projector 把 token block/chunk 表示接到可观测状态。

```text
PhiZero token blocks
        │
        ├── high-level chunk head: next 256-token chunk CE
        └── low-level block head: 8 × 32-token transition blocks CE
                │
                ▼
        block/chunk embedding e_k
                │
                ▼
        projector: e_k → q_k (and v_k)
                │
                ├── state/event grounding
                ├── block boundary and trajectory smoothness
                └── conditional energy/momentum residual
```

`q` 是学习到的 generalized coordinate 或 benchmark 状态表示，不默认等于真实 `(x,y,vx,vy)`。不能把整数 token id 直接当作可微物理坐标。主线仍由离散 token CE 决定预测，projector 只提供可观测的状态锚点。

#### Projection/state grounding

投影头先作为一次性 probe，再决定如何接入训练：

| 版本 | 输入 | 网络 | 使用条件 |
|---|---|---|---|
| A | 单 block/chunk embedding | `Linear(d,128) -> GELU -> Linear(128,s)` | benchmark 状态可由浅层映射读出 |
| B | 整段 token/block embedding | 1D causal conv 或 2 层 tiny transformer | 状态需要时间积分或上下文 |
| fallback | Reasoner hidden state | 同 A/B | token embedding 不够但 hidden state 可读 |

这里的 `d` 和状态维度 `s` 都必须从 checkpoint 与 benchmark schema 读取。`R²` 只作为工程诊断，不能预先规定所有数据都必须预测 `(x,y,vx,vy)`。多物体场景仍需明确单主对象、固定 object id 或 permutation-invariant set 对齐。

#### 主线 combined objective

```text
L_total =
    L_AR
  + λ_block L_block
  + λ_chunk L_chunk
  + λ_state M_state L_state
  + λ_smooth L_smooth
  + λ_energy M_conservative L_energy
```

- `L_AR`：PhiZero 原始 next-token CE；
- `L_block`：32-token transition block 的局部多步 CE；
- `L_chunk`：下一个 256-token chunk 的 CE；
- `L_state`：只有 benchmark 提供状态标签时启用；
- `L_smooth`：相邻预测状态或 block 边界的连续性约束，不等同于 stationary-action 条件；
- `L_energy`：只有质量、速度、接触、外力等字段足够完整时启用。

`M_state` 和 `M_conservative` 是数据条件 mask。能量约束只在适用场景启用，不能对摩擦、外力或非弹性碰撞片段强行要求零漂移。

#### LaWM DEL 增强消融

在主线 projector 跑通后，再增加 learned discrete Lagrangian 和 `L_DEL` residual：

```text
v_k = (q_{k+1} - q_k) / h
q̄_k = (q_k + q_{k+1}) / 2
L_d(q_k, q_{k+1}; η)
    = h [0.5 * v_kᵀ Mθ(q̄_k, η) v_k - Vθ(q̄_k, η)]

R_DEL =
    D2 L_d(q_{k-1}, q_k; η)
  + D1 L_d(q_k, q_{k+1}; η)
```

如果下一状态仍由普通 Reasoner 生成，`L_DEL` 只是 auxiliary residual；只有由有限步 DEL solver 产生 `q̂_{k+1}`、再由 token head 生成 physical-language tokens，才称为 hybrid LaWM variant。二阶初始化必须在 schema audit 中明确：两个观测/token blocks、首个 chunk 内的相邻 blocks，或由首帧和 caption 预测初速度。

同时报告两类能量：

```text
E_latent(q,v) = 0.5 * vᵀ Mθ(q,η) v + Vθ(q,η)
E_phys = simulator-defined kinetic + potential + contact/work terms
```

`E_latent` 用于 LaWM 增强的机制诊断；`E_phys` 只在仿真器提供对应字段时计算。真实视频没有可靠质量、三维速度和外力时，不强行计算 `E_phys`。

### 5.0.2 三种长 rollout 模式

| 模式 | 过程 | 作用 |
|---|---|---|
| token-only | Reasoner 直接递归生成下一段 token | 最低成本主对照 |
| hidden-state carry | 每个 chunk 传递上一段 Reasoner hidden state | 检验 Transformer-XL 式记忆是否减少漂移 |
| render-reencode | token 解码成视频，再用 frozen tokenizer 重编码 | 最强重锚定，但成本最高，需单独报告 |

render-reencode 不能默认算免费闭环。它会重新依赖 decoder 的视觉质量，也可能把 decoder 错误注入下一轮 token。因此它更适合作为“长程稳定上界/诊断模式”，而不是首轮唯一推理路径。

### 5.0.3 Token block 与时间对齐

PhiZero 的 released path 固定输入 33 帧、8 FPS 视频，并输出 256 个 physical-language symbols。论文方法将其解释为 9 个时间 latent 状态之间的 8 个 transition，每个 transition 使用 32 个 symbols。因此首轮应把一个 32-token group 作为最小 transition block；它不应被误称为一个物理时间步。层级模型的 high-level unit 是完整 256-token chunk，LaWM-style generalized coordinate 可以在 block 或 chunk 层建立，但选择必须与 rollout horizon 和二阶初始化方式一起记录。

实际 block 时长必须由 tokenizer 的 temporal stride 和导出张量确认。不能直接把 256 token 均匀除成 0.25 秒 chunk，也不能默认单 token 具有独立的 `(x,y,vx,vy)` 语义。所有 MCP horizon、投影头标签和 16--32 秒外推曲线都要在 manifest 中记录：

```text
video_fps
num_frames
token_count
tokens_per_transition_block
num_transition_blocks
effective_block_duration
```

### 5.1 物理语言 tokenizer

```python
z = tokenizer.encode(video_or_latent_sequence)
recon = optional_renderer.decode(first_frame, z)
```

`encode` 必须支持批处理、保存 token、导出版本和随机种子；`decode` 是可选审计路径，不得成为 dynamics smoke test 的硬依赖。

### 5.2 非显式 dynamics

```python
z_future, unc = dynamics.predict(
    observation_history=obs_history,
    horizon=H,
)
```

最小返回值包括未来状态表示、每步不确定性和模型版本。Future 阶段加入动作条件后，再要求 planner 在相同 `obs_t` 下比较多个动作候选；阶段一不展示动作采样。

### 5.3 物理后果评价

```python
score = evaluator.evaluate_physical_consequences(
    observation_history=obs_history,
    prediction=z_future,
)
```

阶段一评价至少覆盖：multi-step state error、接触/终止事件 F1、事件时间误差、窗口边界跳变、推理延迟、显存、单位 rollout 成本和不确定性校准。动作排序、控制成功率和 episode return 留到 Future 阶段。

### 5.4 闭环重锚定与 PhiZero 对比接口

当前阶段的闭环只表示被动预测中的“重新观测和重锚定”，不执行动作。每轮预测一段未来物理 token，读取真实新观测，再重新调用 encoder 和 dynamics：

```python
while not done:
    z_t = encoder.encode(observation)
    futures = dynamics.predict(observation_history, horizon=H)
    observation = observation_stream.next()
```

真正的动作条件闭环、候选动作规划和 MPC 后置到 Future 阶段。因此，“不生成未来像素”不等于“不使用视觉观测”。模型仍然在每个真实时间步读取当前观测，只是不用 diffusion decoder 合成尚未发生的画面。

与 PhiZero 的统一对照接口为：

```text
同一观测历史 o_{≤t}
    ├── PhiZero: physical-language rollout → video decoder（可选）
    └── Ours:    passive physical-transition rollout
```

阶段一的物理预测评价使用真实仿真状态、事件标签或统一的外部跟踪器。不能因为 PhiZero 生成了视频，就只用视频观感评价它；也不能因为本模型不生成视频，就用 token 交叉熵宣称它已经理解物理。两者都必须落到位置、速度、接触、碰撞和事件时间上。

建议将以下曲线作为最小实验图：

```text
horizon H → state/event error
horizon H → re-anchored prediction error
rollout length → latency and GPU cost
```

其中 `Ours-passive-open-loop` 与 `Ours-passive-reanchored` 的差异用于证明重新观测对预测漂移的影响；`PhiZero passive/adapted` 与 physical-language-only 长 rollout 的差异用于展示未来像素渲染对被动物理预测准确性和系统成本的影响。动作条件的控制成功率和 MPC 成本留到后续阶段。

## 6. 训练数据与标签策略

首轮按成本从低到高：

1. 仿真状态和被动视频作为近似真值，验证 dynamics 和物理事件接口；
2. 小规模真实视频，通过冻结 encoder/tokenizer 生成转移标签；
3. 后续再扩大到动作丰富视频和机器人 demonstration；
4. 最后才考虑大规模 PhiZero 风格视频 tokenizer/reasoner 训练。

标签分为三层：

| 标签 | 来源 | 用途 |
|---|---|---|
| 像素/视频 | 原始视频 | 只训练或检查 tokenizer/renderer |
| latent/物理语言 | tokenizer 或 teacher encoder | 训练 dynamics、预测 token |
| 状态/事件/后果 | 仿真器、检测跟踪、人工小样本 | 评价、干预、规划和校准 |

不能把 tokenizer 自己生成的 token 当成无偏 ground truth；必须用外部事件、仿真状态或动作结果做独立评价。

## 7. 基线与消融

至少建立：

- pixel/video prediction；
- 常规 VAE latent prediction；
- object-state 或 scene-graph prediction；
- PhiZero-style discrete physical language；
- consequence-only predictor（不渲染未来视频）；
- oracle simulator state（仅作为上界，不作为公平 baseline）。

阶段一关键消融：one-step passive、flat long AR、hierarchical MCP、MCP + projector/state grounding、MCP + conditional physics、optional MCP + DEL residual、optional hybrid DEL solver、hidden-state carry、render-reencode、scheduled sampling、history-token noise、observation re-anchoring、无 first-frame condition、无离散瓶颈、无结构化关系、无 uncertainty head、无 renderer、短/长 horizon、随机/错配 token、遮挡与分布外观测。DAgger、action condition 和 MPC 属于后置控制阶段。

## 8. 可复现记录

每个运行目录保存：`config.yaml`、Git commit、依赖锁定、数据 manifest、模型 revision、随机种子、硬件摘要、日志路径、指标 JSON、失败样本索引和 SHA256。结果表必须同时报告“是否生成像素”和“预测表示类型”，避免把不同任务混在一个分数中。
