# 非像素世界模型 idea 调查记录

**调查日期**：2026-10-07
**主要来源**：PhiZero 预印本的 Related Work/Appendix D、公开论文元数据、PhiZero 官方 README。该记录用于生成研究假设，不代表每个方向都已完成复现。

## 1. 相关路线对照

| 路线 | 代表工作 | 预测对象 | 对 WAM 的启发 | 仍缺的问题 |
|---|---|---|---|---|
| 潜在视频预测 | V-JEPA、WorldDreamer、iVideoGPT | latent/token 序列 | 可以跳过像素，直接学习未来表示 | latent 是否包含可干预的动作后果，不能只用 representation probe 判断 |
| 交互式生成环境 | Genie、Pandora、Pan、Astra | action + video/latent rollout | 支持动作条件与长时域交互 | 每一步视频生成昂贵，视觉合理不等于因果正确 |
| latent-action world model | Genie、Adaworld、Motus、Co-evolving latent action、DILA、Factored LAWMs | 从相邻观测反推 latent action，再做 forward model | 可以从无动作视频学习可控制的转移变量 | 多数面向控制/特定 embodiment，开放世界物理状态覆盖有限 |
| 物理/事件推理 | Physion、IntPhys2、ComPhy、CLEVRER | 物体、碰撞、事件和合理性判断 | 可以用独立事件指标评价 world model，而不是只看视频质量 | 事件标签、跟踪和遮挡鲁棒性仍是瓶颈 |
| 视频 tokenizer | Divot、Vtok、Vidtwin、TivTok、content-frame/motion-latent 分解 | 压缩 token、内容/运动分解 | 为物理语言提供 tokenizer 和 decoder 设计 | 压缩 token 可能仍编码外观，必须做交换首帧/交换转移的干预测试 |
| 变分 latent world model | **LaWM（Least Action World Models）** | learned generalized coordinates、离散 Lagrangian、DEL variational transition | 物理结构可以直接定义 latent rollout rule，而不只是作为 post-hoc loss；可借鉴 `L_lat + L_DEL + L_reg`、PIS、DEL residual 和 energy-drift diagnostics | LaWM 的连续 latent 与 PhiZero 离散 token 不同；二阶状态初始化、耗散/接触和 token-to-latent bridge 需要重新设计 |
| 显式物理约束 | PhysAlign、simulator-in-the-loop、WISA、Think before diffuse | 物理特征、3D 或规则 | 可做强对照或独立 evaluator | simulator/3D 依赖大，且可能把先验限制在特定领域 |
| Physical language | PhiZero、UNIT（论文引用） | 离散状态转移符号 | reason-then-render，支持跨外观/embodiment 接口 | 符号仍是经验性的，不直接对应物理量或方程 |
| 时间物理推理 benchmark | **ChronoPhyBench** | 历史视频、下一状态候选、时间排序和冲突文本 QA | 可检验模型是否真的使用视觉历史，并为 physical-language-only 模型提供外部视觉对照 | 主要是选择/问答协议，不直接给出连续轨迹误差或 32 秒 rollout |
| 物理视频生成评估 | **Morpheus** | 真实实验视频、轨迹方程和守恒量评分 | 可作为 renderer/state-to-video 的物理审计器，独立于主观观感 | token-only 输出不能直接套官方视频评分；现象和守恒假设必须匹配 |
| 交互式长时域评测 | **WorldOdysseyBench** | WASD 交互视频、逐帧动作、视觉漂移、物理和记忆 | 可借用 drift、interaction physics 和 memory 的分项设计 | 当前是动作条件 interactive world model；不适合作为阶段一无动作主榜单 |

## 2. 关键判断

1. **“不生成像素”并不自动构成新颖性**：V-JEPA、WorldDreamer、latent-action world models 已经说明未来表征可以在 latent/token 空间预测。新意需要落在可干预的 physical-language 接口、反事实后果、跨任务/跨 embodiment 泛化或等算力收益上。
2. **PhiZero 的差异在中间变量用途**：它不是只用 latent 做视频压缩，而是把离散转移作为 reasoner 的显式预测目标，再可选地渲染视频。WAM 可以进一步把 renderer 从主推理路径移到审计路径，直接评价规划和控制。
3. **latent action 与 physical language 要区分**：latent action 主要表示“采取了什么动作/控制效果”，physical language 试图表示“世界如何演化”。实验必须加入无动作观察、同动作不同场景、同转移不同外观三种对照，避免把二者混为一谈。
4. **评价应从视频质量转为后果质量**：至少需要 multi-step state error、事件/接触 F1、反事实动作排序、闭环 success、uncertainty calibration 和推理成本；视频只作为失败审计或可视化。
5. **最有风险的是因果幻觉**：renderer 可以把错误 latent 变成视觉上连贯的视频。必须使用错动作、错 token、局部关系干预和遮挡测试，验证模型的因果敏感性。
6. **物理一致性不能只写成能量正则**：LaWM 的关键差异是用离散 Euler--Lagrange 条件定义 latent transition。WAM 首轮采用 projector、状态/连续性约束和条件化能量项；LaWM 的 `L_DEL`/solver 作为独立增强与消融。只有 solver 真正定义下一 latent state 时，才把该配置称为 hybrid LaWM variant；能量 drift 与 PIS 始终作为独立诊断。
7. **六个 benchmark 必须分工使用**：ChronoPhyBench 负责视觉历史和下一状态/时间顺序，Morpheus 负责可渲染输出的物理知觉审计，Physics-IQ Verified、PhyGround 和 WorldModelBench 负责 PhiZero 可对齐的视频物理对照，WorldOdysseyBench 负责未来交互控制、漂移和记忆。它们不能平均成一个总分，也不能用其中一个替代 MuJoCo 的连续状态真值。

## 3. 形成的可检验 idea

### I1：Physical-language-only consequence model

使用 PhiZero 风格 tokenizer 得到离散转移 `z`，但不训练或不调用未来视频 renderer；reasoner/dynamics 直接预测 `z_future`、结构化后果和不确定性。核心比较是与 pixel rollout、VAE latent rollout、object-state rollout 的等算力闭环控制。

### I2：Factorized physical language

借鉴 factored latent action 和对象关系图，把状态转移拆为实体级变化、接触/拓扑变化、全局环境变化三组 token。局部动作只更新相关因子，再由 planner 组合，检验长时域和多实体组合泛化。

### I3：Renderer-as-auditor

主模型只预测状态/事件；renderer 低频渲染关键帧、随机抽样或失败案例。使用 renderer 发现 representation collapse、不可见状态遗漏和跨外观迁移失败，而不是把渲染结果作为主训练目标。

### I4：Counterfactual consequence ranking

给定同一观测和多个候选动作，只输出目标位移、接触、碰撞、终止概率及 uncertainty，直接训练动作排序。该设置适合检验“控制是否真的需要未来视频”。

### I5：跨 embodiment 的转移语言

同一 `z` 在人类、仿真机械臂和真实机器人首帧条件下解码/执行，评价运动关系、阶段事件和成功率是否保持。必须把外观迁移与动作接口迁移分开，否则无法判断是物理语言有效还是 renderer 风格迁移。

## 4. 近期开工优先级

1. I1 + I4：最容易在小型仿真上证伪，直接对应用户提出的“不生成像素也能预测未来”；
2. I2：若 I1 中一维 token 在遮挡、多实体和长时域失败，再引入结构化关系；
3. I3：用于调试和论文图，不把它变成额外主模型；
4. I5：需要多 embodiment 数据和动作接口，作为后续扩展。

## 5. 需要避免的结论跳跃

- token 可视化像“语言”不等于它具有语义；
- 低重建误差不等于未来状态正确；
- 动作条件视频看起来合理不等于反事实因果正确；
- 跨场景相似 token 不等于跨 embodiment 可执行；
- 单步预测提升不等于长时域闭环提升；
- 比 pixel rollout 更快只有在相同输入、horizon、硬件和 batch 下测量才成立。

## 6. 公开入口

- PhiZero：<https://github.com/yaoyao-jpg/PhiZero>；论文：<https://arxiv.org/abs/2607.28624>
- V-JEPA：<https://openreview.net/forum?id=YdMdFe8FJm>
- Genie：<https://arxiv.org/abs/2402.15391>
- WorldDreamer：<https://arxiv.org/abs/2401.09985>
- Pandora：<https://arxiv.org/abs/2406.09455>
- iVideoGPT：<https://arxiv.org/abs/2405.17589>
- Physion：<https://arxiv.org/abs/2106.08261>
- ComPhy：<https://arxiv.org/abs/2205.01089>
- Motus：<https://arxiv.org/abs/2512.13030>
- AdaWorld：<https://arxiv.org/abs/2503.18938>
- Learning latent action world models in the wild：<https://arxiv.org/abs/2601.05230>
- Factored latent action world models：<https://arxiv.org/abs/2602.16229>
- DILA：<https://arxiv.org/abs/2605.15725>
- ChronoPhyBench：<https://arxiv.org/abs/2606.07962>；代码/数据接口：<https://github.com/huangchong-yan/ChronoPhyBench>
- Morpheus：<https://arxiv.org/abs/2504.02918>；代码：<https://github.com/physics-from-video/Morpheus>
- WorldOdysseyBench：<https://huggingface.co/papers/2606.31672>；arXiv 记录：<https://arxiv.org/abs/2606.31672>
- MuJoCo 状态与接触字段：<https://mujoco.readthedocs.io/en/latest/programming/simulation.html>、<https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html>
- LaWM：本地参考文档 [`参考/WAM/LaWM.pdf`](../../../参考/WAM/LaWM.pdf)
