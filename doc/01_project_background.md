# WAM 终局：非显式像素世界模型与物理语言预测

**版本**：Research design draft v0.1  
**日期**：2026-09-26  
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

## 2. 与 PhiZero 的关系和差异

PhiZero 的主线是：

```text
视频 V ── tokenizer ──> 离散物理语言 z
首帧 I0 + 动作意图 c ── reasoner ──> z_future
I0 + z_future ── diffusion decoder ──> 未来视频 V_future
```

本项目保留 `z` 作为显式可检查的中间变量，但优先验证更省算力的变体：

```text
观测 o_t + 动作 a_t ── dynamics/reasoner ──> 状态转移 z_{t:t+H}
状态转移 z + 任务目标 ── planner/policy ──> a_{t:t+H}
（可选）o_t + z ── renderer ──> 未来视频，仅用于可视化或审计
```

PhiZero 当前仍是数据驱动的经验性状态转移表示，不是可解释的符号物理定律；论文也指出视觉不可观测的触觉、微观粒子和长时域转移仍是限制。WAM 的研究重点因此放在“非显式但可预测、可干预、可验证”的表示，而不是宣称已经获得真正的物理定律。

## 3. 初步 idea 候选

### Idea A：Physical-language-only world model（首选）

训练 tokenizer 将相邻 latent state 的变化编码成离散 token 或短序列；训练 dynamics model 只预测未来物理语言，不解码视频。下游用状态预测误差、动作可达性、接触事件和规划成功率评价。视频 decoder 只保留为离线诊断器。

**核心假设**：对于控制和规划，预测“物体/关系/运动如何变化”比生成每一帧的纹理更重要。  
**最小验证**：在小型机器人操作或 2D/3D 物理环境中，与像素预测、VAE latent 预测、object-state 预测比较相同算力下的 multi-step state error 与 success rate。  
**主要风险**：物理语言可能只记住动作类别，丢失绝对位置、遮挡下的状态和细粒度接触信息。

### Idea B：Non-explicit world model with task-sufficient latent（任务充分而非物理显式）

不强制 token 具有人工可读语义，只要求 latent 能同时支持未来状态判别、反事实动作排序和策略规划。使用 inverse dynamics、forward dynamics、contrastive predictive coding 和 policy value consistency 约束，避免把“离散 token 可读”误认为“世界模型有效”。

**核心假设**：一个对任务足够的非显式表示可能比固定的物理语言词表更容易优化，也更适合连续控制。  
**主要风险**：latent 可能退化成策略特征，跨任务复用和科学解释性不足。

### Idea C：事件图/关系图物理语言

把每个时间窗编码为对象节点、接触边、相对位姿、速度方向、容器关系和事件（接触、支撑、遮挡、破碎、流入等），再用图 transformer 预测图的增量。与 PhiZero 的一维 FSQ token 比较，检验结构化关系是否减少 token 数量并改善组合泛化。

**核心假设**：对象关系和事件比纯序列 token 更容易进行局部干预与长时域滚动。  
**主要风险**：对象检测/跟踪错误会把 perception 错误传给 dynamics；开放世界中的关系词表难以固定。

### Idea D：不渲染像素的反事实动作世界模型

给定观测、动作候选和目标，模型直接预测一组结构化后果：目标位移、接触是否成立、抓取是否稳定、碰撞风险、终止条件和不确定性。训练时用视频 tokenizer 或仿真状态产生弱标签；推理时只在候选动作之间排序。

**核心假设**：动作选择只需要可比较的未来后果，不需要完整未来视频。  
**最小验证**：固定相同 policy backbone，比较 pixel rollout、latent rollout 和 consequence-only rollout 的规划成功率、延迟和显存。

### Idea E：世界模型的“渲染审计器”而非“渲染核心”

让结构化状态模型负责主预测，另设一个低频 renderer 只渲染关键帧或失败样本。renderer 用来发现状态模型的视觉盲点、生成可解释 debug 证据和辅助人工评审，避免每一步都支付视频生成成本。

**核心假设**：渲染应服务于审计和数据闭环，而不是定义世界模型能力。  
**主要风险**：审计器与状态模型共享错误表征时，可能产生“看起来合理”的错误视频。

## 4. 当前推荐的主线

首轮采用 **A + D**：先学习一个小规模、可导出的物理语言/状态转移表示；再训练只预测结构化后果的 dynamics model，并把 PhiZero 风格 renderer 降为可选审计模块。B 作为连续 latent 强 baseline，C 作为结构化表示扩展，E 作为工程化可视化策略。

第一阶段不追求复现论文的 128 张 A100 训练规模，也不把大规模视频数据下载作为启动条件。先用公开的小数据或仿真环境验证“去掉像素生成后，预测和控制是否仍然成立”。

### 4.1 进一步收敛后的论文主张

建议暂用一个能概括整条链路的名称：**Closed-Loop Physical Transition Model（CPTM，闭环物理转移模型）**。论文主方法只回答一个问题：在不生成未来像素的情况下，动作条件的转移 token 是否足以支持多步预测和闭环决策？

为了避免把 MCP、scheduled sampling、DAgger、RAG、MPC、熵正则和物理损失写成七个并列贡献，建议把主张收敛成一句话：

> **给定当前观测和动作候选，模型在物理转移 token 空间预测未来状态后果；训练用多步转移监督和仿真器纠偏减少 rollout 漂移，推理只在闭环观测处重新锚定，不需要逐步生成未来像素。**

其中只有三件事构成主线：

1. **表示**：外观与状态变化分离的 transition token/latent；
2. **预测**：动作条件的多步未来转移预测，并直接支持反事实动作排序；
3. **闭环**：执行一步、重新观察、重新预测的 MPC 式验证协议。

MCP（需要在论文中明确定义为“多步转移一致性监督”）是训练目标；DAgger 是在有仿真器真值时用于纠偏的数据收集机制；闭环重锚定是推理与评估协议。三者共同支撑主张，但不要把它们包装成三个独立创新。

### 4.2 核心、支撑项和 tricks

| 组件 | 定位 | 首轮处理 |
|---|---|---|
| transition physical token/latent | 核心表示 | 必做；与 pixel/VAE/object-state baseline 对照 |
| 动作条件多步预测（MCP） | 核心训练目标 | 必做；报告 1/4/8/16 步误差和反事实排序 |
| 闭环重新锚定（MPC） | 核心验证协议 | 必做；每执行一步用真实观测更新状态 |
| DAgger 式仿真器纠偏 | 关键支撑机制 | 只有 MuJoCo/ManiSkill 等有状态真值时启用 |
| Scheduled sampling | 训练替代/消融 | 与 DAgger 分开比较，首轮不要同时堆叠 |
| 历史 token 加噪 | 鲁棒性 regularizer | 后加；只在历史误差敏感实验中启用 |
| 熵正则/码本利用率约束 | 防离散 token 坍缩 | 仅用于离散 tokenizer，并报告 perplexity/usage |
| RAG 相似轨迹检索 | 外部记忆增强 | 作为 baseline/消融，不能算主创新 |
| token 重复惩罚 | 解码技巧 | 只有出现重复退化时才加 |
| 能量/动量损失 | 域特定物理先验 | 只在有质量、速度、接触和外力定义的仿真数据上启用 |

首轮实验只保留一条清晰增量链：`one-step → MCP → MCP+scheduled sampling`，然后在独立实验中替换为 `MCP+DAgger`。RAG、历史噪声、熵正则和物理残差都不能进入首版主结果。

### 4.3 物理损失的边界

通用真实视频不能直接使用能量/动量守恒损失。物体质量、三维速度、相机标定、接触冲量和外力通常不可观测；有摩擦、碰撞、驱动器或线缆时，系统本来也不是封闭守恒系统。推荐将其改成**带条件的物理残差**：只在仿真器提供 `qpos/qvel/contact/force` 的片段上计算，并把动作做功、摩擦耗散和外力项纳入残差。真实视频阶段使用轨迹、接触和事件指标，不强行套守恒公式。

## 5. 必须保持诚实的边界

- `physical language` 在 PhiZero 中是学习到的离散状态转移符号，不等于可读的自然语言，也不等于已知物理方程；
- 不把重建质量、视频观感或 token 可视化直接当成世界模型正确性的证据；
- 必须报告遮挡、接触、长时域、分布外动作和反事实动作的失败案例；
- 如果代码仓库或权重未公开，项目只能做论文结构分析和小规模概念验证，不能声称完成 PhiZero 数值复现；
- 所有候选 idea 都需要通过与 pixel/latent/object-state baseline 的等算力对照和反事实测试。

## 6. 待确认事项

相关路线的调查记录见 [`references/manifests/literature_scan.md`](../references/manifests/literature_scan.md)，其中区分了 latent video prediction、latent-action world model、事件/物理推理、视频 tokenizer 和 PhiZero physical language。

1. `WAM终局` 已初始化独立本地 Git；是否需要绑定并推送到已有远端；
2. 首轮环境优先选 LIBERO/BridgeData 类机器人数据、视频物理数据，还是先用可控的 2D/小型仿真环境；
3. 目标是先做 idea 设计与 smoke test，还是立即进入完整训练和论文复现；
4. 可用 GPU、存储和网络代理条件决定是否下载大模型、视频数据和 Wan/PhiZero 相关权重。
