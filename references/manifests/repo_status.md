# PhiZero 仓库审计

- **来源**：https://github.com/yaoyao-jpg/PhiZero
- **本地路径**：WAM终局/references/repos/PhiZero
- **下载方式**：git clone --depth 1，通过 http://127.0.0.1:7897
- **HEAD commit**：2d1b3e83a139027b01aadd41be6c6052bf76fd3a
- **树对象**：d3bcb004fedbc3131fc394a4fd21426b37fdd29f
- **本地大小**：156M（含仓库内示例资产；未下载外部 checkpoint）
- **论文**：https://arxiv.org/abs/2607.28624；本地副本：参考/WAM/PHIZERO.pdf
- **项目页**：https://phi-zero.github.io/

## 当前可见内容

- DiffSynth-Studio/：tokenizer、FSQ、视频 decoder、训练与渲染入口；
- ms-swift/：Physical Language Reasoner 推理、数据准备和训练入口；
- examples/：推理、重建、GT 提取、tokenizer 训练、reasoner 训练和 motion transfer 启动脚本；
- data/：小型示例/案例元数据，完整训练视频不在仓库；
- environments/：CUDA 12.4、PyTorch 2.6.0、Transformers 5.12.1 等环境文件；
- ckpt/：README 预期的目录，但公开 clone 中未包含 checkpoint；
- 外部 Wan2.2 DiT/VAE 与 PhiZero checkpoint 需要从 Hugging Face/ModelScope 另行下载。

## 重要运行约束

1. reasoner 与 decoder 是两个环境，均依赖 FlashAttention 2.8.3；
2. tokenizer/decoder 需要 Wan2.2 base DiT/VAE；reasoner 需要 Qwen3-VL 类模型和 DeepSpeed；
3. 官方脚本通过环境变量传入 Python、checkpoint、CSV 和外部模型路径；
4. README 的完整训练规模和论文训练规模都远大于当前本地 smoke test；
5. 当前只完成代码下载和静态审计，尚未安装环境、下载权重或运行推理。

## 许可与数据

仓库根目录含 LICENSE，第三方子目录 DiffSynth-Studio/ 和 ms-swift/ 各有自己的许可证文件。外部模型、视频数据和 Hugging Face/ModelScope 资源必须单独审查许可证，不因代码仓库开源而默认可任意使用。

## 静态 smoke test

- `python3` 当前为 3.8.10，直接编译会因仓库使用 Python 3.10 的 `match` 语法失败；
- 使用 `/usr/bin/python3.10 -m compileall` 编译 `DiffSynth-Studio/diffsynth`、`ms-swift/swift` 和 `ms-swift/examples` 成功；
- 尚未执行 import、GPU 推理或训练，因为当前环境未安装官方 CUDA/PyTorch/FlashAttention 依赖，也未下载 checkpoint。
