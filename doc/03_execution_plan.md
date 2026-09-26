# 执行计划与 Git 操作：从 PhiZero 代码审计到非像素世界模型验证

**版本**：Research execution draft v0.1  
**日期**：2026-09-26  
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
| P4 | 反事实动作与规划评价 | action ranking、state error、success | [ ] |
| P5 | 结构化事件图扩展 | graph/event ablation | [ ] |
| P6 | 可选 renderer 审计闭环 | 失败样本渲染、跨外观/embodiment | [ ] |
| P7 | 论文级对照和结论 | 等算力表、失败案例、idea 选择 | [ ] |

## 3. P0：文档先行（当前步骤）

- [x] 生成项目背景与研究定位；
- [x] 生成技术栈与张量/接口约定；
- [x] 生成本执行计划，写明每步验收、commit 和 push；
- [x] 明确首选 idea：physical-language-only + consequence-only；
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

- [ ] 实现 `dynamics.predict(observation, action, horizon)`；
- [ ] 建立 pixel/video、VAE latent、物理语言和 consequence-only 四个 baseline；
- [ ] 训练短 horizon，再逐步增加 rollout 长度；
- [ ] 记录 multi-step state error、事件 F1、uncertainty calibration、延迟和显存；
- [ ] 进行随机 token、时间错位 token、错误动作和遮挡输入的反事实测试；
- [ ] 检查模型是否只预测动作类别而没有预测结果状态；
- [ ] 把失败样本按 perception、representation、dynamics、planning 分类。

### P3 验收

在同一数据和算力预算下，consequence-only 或 physical-language-only 至少能在一个控制/规划指标上达到 pixel baseline 的可比水平，同时显著降低 rollout 成本；若没有，保留负结果并停止无依据扩展。

### P3 Git

```bash
git add src/dynamics src/evaluation configs outputs/dynamics_smoke
git commit -m "feat: evaluate non-pixel future-state prediction"
git push
```

## 6.1 推荐训练配方

### Stage 0：数据与表示准备

- [ ] 用仿真轨迹建立 `(observation, action, state, event, next_state)` manifest；
- [ ] 用 PISA/Kubric 视频建立外观变化与掉落运动的预训练/验证 split；
- [ ] 训练或冻结 transition encoder，检查同一运动换首帧、同一首帧换运动的表示距离；
- [ ] 离散 token 必须记录 codebook usage、perplexity、重复率和 scene/motion probe。

### Stage 1：教师强制的动作条件多步预测

定义长度为 `H` 的转移目标 `z_{t+1:t+H}`，先用真实历史和真实动作做 teacher forcing：

```text
L_MCP = Σ_h w_h · CE(z_{t+h}, pθ(z_{t+h} | o_t, a_{t:t+h-1}, z_{t+1:t+h-1}))
```

连续 latent 或结构化状态可以把 `CE` 换成 masked regression/event loss。必须与 one-step baseline 对照，不能只展示训练 loss。

### Stage 2：处理 rollout 漂移

- [ ] 先做 scheduled sampling：逐步提高历史预测 token 的比例；
- [ ] 独立做 DAgger：让模型在仿真中 rollout，访问这些偏离专家分布的状态，再由 MuJoCo/ManiSkill oracle 提供状态/后果标签并聚合数据；
- [ ] 不要在首轮同时使用 scheduled sampling 和 DAgger，否则无法知道收益来自哪里；
- [ ] 历史 token 加噪只作为第三个鲁棒性消融。

### Stage 3：闭环 MPC 评估

每次只执行一步或短动作 chunk，重新读取观测，再预测下一轮后果。报告开放环 rollout 与闭环 rollout 的差距、动作排序准确率、成功率、状态误差和每步延迟。RAG 只作为无检索/有检索对照，不进入主模型定义。

### Stage 4：物理残差和 token 稳定性

- [ ] 离散 tokenizer 才启用熵正则，并报告词表利用率而非只报总 loss；
- [ ] 仿真器有质量、速度、接触和外力时，再启用能量/动量残差；
- [ ] 对摩擦、碰撞、动作做功和外力显式建模，不能默认“总能量恒定”；
- [ ] token 重复惩罚只在固定 horizon 自回归解码出现重复时启用。

## 6.2 PISA Experiments 的正确角色

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

## 7. P4：反事实动作和规划

- [ ] 对同一观测生成多个候选动作；
- [ ] 用真实后果、仿真器或人工小样本标签验证动作排序；
- [ ] 对目标位移、接触成立、碰撞风险、终止和不确定性分别评价；
- [ ] 比较每一步渲染视频与只预测结构化后果的延迟/显存/成本；
- [ ] 测试长时域闭环和分布外动作，而不是只看 one-step loss；
- [ ] 形成 `outputs/counterfactual_report.md`，写清哪些结果支持或反驳主假设。

### P4 Git

```bash
git add src/planner outputs/counterfactual_report.md configs
git commit -m "eval: add counterfactual action consequence benchmark"
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

- [ ] 所有 baseline 使用相同观测、动作、horizon、数据 split 和算力记录；
- [ ] 同时报 state prediction、counterfactual planning、closed-loop success、efficiency 和 calibration；
- [ ] 至少保留一个失败案例集和一个负结果；
- [ ] 对主 idea 做 novelty/related-work 检索，区分 PhiZero 的直接延伸与真正的新问题；
- [ ] 若 consequence-only 在控制指标上不优于 latent/pixel baseline，则停止扩展并改写研究问题；
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
