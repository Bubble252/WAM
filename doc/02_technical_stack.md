# 技术栈与实现规格：物理语言、非显式动力学与可选渲染

**版本**：Research design draft v0.2
**日期**：2026-09-28
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

其中 `Ours-passive-open-loop` 与 `Ours-passive-reanchored` 的差异用于证明重新观测对预测漂移的影响；`PhiZero passive/adapted` 与 consequence-only 的差异用于展示未来像素渲染对被动物理预测准确性和系统成本的影响。动作条件的控制成功率和 MPC 成本留到后续阶段。

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

阶段一关键消融：one-step passive、passive MCP、scheduled sampling、history-token noise、observation re-anchoring、无 first-frame condition、无离散瓶颈、无结构化关系、无 uncertainty head、无 renderer、短/长 horizon、随机/错配 token、遮挡与分布外观测。DAgger、action condition 和 MPC 属于后置控制阶段。

## 8. 可复现记录

每个运行目录保存：`config.yaml`、Git commit、依赖锁定、数据 manifest、模型 revision、随机种子、硬件摘要、日志路径、指标 JSON、失败样本索引和 SHA256。结果表必须同时报告“是否生成像素”和“预测表示类型”，避免把不同任务混在一个分数中。
