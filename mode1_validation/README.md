# Qwen3-VL-4B-Instruct 的 Mode 1 精度验证

从 SDK 根目录执行以下命令，本次使用了本机 nxp Python 环境。

```bash
/Users/liwiii/venv/nxp/bin/python mode1_validation/compare.py \
  /Users/liwiii/huggingface/Qwen/Qwen3-VL-4B-Instruct \
  mode1_onnx/model.onnx
```

脚本离线读取 checkpoint 和 ONNX 外置权重，运行后写入同目录的 `results.json`。可用 `--results` 指定另一份报告。验收通过时退出码为 0，指标未达标时为 1。报告记录环境版本、模型文件 SHA-256、两个种子下的 token ID，以及每个生成位置的全词表 logits 误差。

## 数据和推理边界

`cases.json` 定义了五道带参考答案的题。两道纯文本题覆盖算术和翻译，三道视觉题覆盖颜色、计数和左右关系。脚本依据文件中的坐标和颜色生成固定的 384 × 384 RGB 图像。每题单独使用原始 chat template，无 padding；视觉样本的网格为 1 × 24 × 24。

基准使用 Transformers 自带的 `Qwen3VLForConditionalGeneration.from_pretrained(...)`，从原始文本和像素调用 `generate`。在语言模型的输入处取得 prompt embeddings、三轴位置和三个 DeepStack 特征，并按图像 token 的实际位置填入 ONNX。首次 ONNX 调用从空 KV cache 开始，之后读取自身上一轮的全部 36 层 KV，用自身输出 token 查原始 embedding 表，再预测下一 token。对齐范围为语言解码器和 LM head；视觉塔及输入 embedding 为两路共享的上游。

两路均在 CPU 上用 FP32、batch size 1、贪心解码，最多生成 8 个 token，遇 EOS 停止。Transformers 显式关闭采样、beam search 和模型默认生成参数的回填。两路均打开确定性计算，分别用种子 0 和 2026 完整复跑。同一后端的两次 token 序列和原始 logits 均须一致。

## 判定方法

每个样本比较 prefill 末位置及随后每一个 decode 位置的 151,936 个原始 logits，包含预测 EOS 的位置。若自由生成发生分歧，保留两路自由生成的答案，并另外沿 Transformers 的 token 前缀计算同位置误差。答案评分忽略大小写、首尾空白和末端标点。

本实验为 FP32 跨实现回归预设以下阈值。所有生成位置的 top-1、所有样本的自由生成 token 序列均须相同。每个位置的最大绝对误差须 ≤ 0.002、全词表 RMSE 须 ≤ 0.0002、余弦相似度须 ≥ 0.999999。任务答案正确率另行统计。

样本量适合检查短序列转换回归。视频、多图、长上下文以及 Ara 量化后的 DVM 需单独验证。

## 本次实测结果

| 样本 | Transformers | ONNX Runtime |
|---|---|---|
| addition | 42 | 42 |
| translation | Blue | Blue |
| color | red | red |
| count | 3 | 3 |
| spatial | circle | circle |

| 阶段 | Top-1 一致位置 | 最坏最大绝对误差 | 最坏 RMSE | 最低余弦相似度 |
|---|---:|---:|---:|---:|
| prefill | 5/5 | 0.0001506805 | 3.780871e-05 | 0.999999999973 |
| decode | 6/6 | 0.0001516342 | 2.869667e-05 | 0.999999999968 |

五个样本均通过，两个种子在各自后端得到的逐步 logits 最大差均为 0。ORT 加载时移除了执行图未引用的 cos_cache 和 sin_cache 两个 initializer；它们供后续 Ara 分片阶段使用，本次未验证分片阶段。
