# Qwen3-VL-4B 语言分片与精度验证

## 本次实测

最终产物使用 128-token prefill 块，prefill/decode 缓存均为 1024。生成了 36 + 36 + 1 张图。37 份 HDF5 Caffe 权重中的 398 个模型参数与 ONNX 权重在目标布局上逐元素一致，145 个额外归一化 bias 均为零，见 `weights.json`。

| Caffe FP32 执行条件 | 对照 | Top-1 一致 | 最坏最大绝对误差 | 最坏 RMSE | 最低余弦相似度 | 判定 |
|---|---|---:|---:|---:|---:|---|
| 固定 IMAGE 表、默认分块 softmax | Mode 1 ONNX | 3/11 | 43.77138 | 10.56051 | −0.670288 | 未通过 |
| 固定 IMAGE 表、默认分块 softmax | Transformers | 3/11 | 43.77127 | 10.56053 | −0.670288 | 未通过 |
| 按提示提供 RoPE、按绝对位置处理第二块 | Mode 1 ONNX | 11/11 | 0.00010396 | 0.00001882 | 0.999999999986 | 通过 |
| 按提示提供 RoPE、按绝对位置处理第二块 | Transformers | 11/11 | 0.00015188 | 0.00003914 | 0.999999999962 | 通过 |

默认路径五题的首个预测均与参考不同。两道纯文本题的固定 IMAGE RoPE 表与实际文本位置不对齐，表项最大绝对差约为 2。三道图像题的 RoPE 与参考最大只差 5.96e−8，但在 Caffe 第 0 层第 0 个注意力头，第二块的首个 query（全局位置 128）只看到了 key 0；正确的可见区间是 0…128。颜色题在位置 166 只看到了 39 个 key，对照执行可见 167 个。

位置处理对齐后，五题的完整参考 token 序列（含 EOS）均逐位置匹配，对应答案依次为算术 42、翻译 Blue、颜色 red、计数 3、空间关系 circle。`results.json` 与 `position_aware.json` 分别保存两次实验的实际接口、每题位置、token ID、可见 key 及全词表误差。通过结论限于带上述位置处理的 Caffe CPU FP32 对照；未运行 INT4 DVM 或板端程序，不能据此认定板端存在相同的掩码行为。

## 生成

在 SDK 根目录执行：

```bash
./split_qwen3vl_mode2.sh mode1_onnx/kinara_run_config.yaml mode2_1024
```

宿主机需要 Docker、Python 3 和 PyYAML。脚本用 Mode 1 配置定位原始模型的 config.json、model.onnx 及同目录的外置权重，生成独立的运行配置，然后调用 SDK 的 qwen3_vl_4b/language/split.py。分片器读取 ONNX initializer，从 SDK 模板构造 Ara 的 Caffe 图，写出权重、浮点及量化 embedding/RoPE 表、分片配置和编译用校验输入。此步对应 Mode 2 的语言分片生成部分，到 .prototxt 和 .caffemodel 为止。

本次使用 prmpt_prc_sizes=[128]、prmpt_context_size=1024、tkn_gen_sizes=[1024]、specd_size=1。128 表示每次 prefill 消化的位置数，1024 表示总上下文缓存容量。提示预算应包括模板、文本和视觉 token，并为生成留出空间。144 个视觉 token 可跨块；166 个位置的提示按 128 + 38 处理，第二块右侧填零，取其第 38 个有效位置的隐藏状态送入 LM head。跨块时所有层保留各自的 KV，位置及注意力掩码需要覆盖前块。

| 产物，相对 mode2_1024/llm | 数量 | 输入输出与权重 |
|---|---:|---|
| prmpt_prc_128/split_i/split_i.prototxt | 36 | 第 i 层 prefill，i=0…35 |
| tkn_gen_1024/split_i/split_i.prototxt | 36 | 第 i 层单 token decode，i=0…35 |
| caffemodels/split_i.caffemodel | 36 | 生成 prefill 图时写出该层 HDF5 权重；同层 decode 的 dvrun_config.yaml 引用同一文件 |
| lmhead/split_36/lmhead.prototxt、lmhead.caffemodel | 1 图、1 权重 | 最后一层隐藏状态经 RMSNorm 和共享输出投影，生成词表分数 |
| assets/ | 1 个目录 | 包含模型参数、embedding、RoPE 表及其量化版本 |

语言模型有 36 层。每个推理阶段按层串接 split_0 到 split_35，再调用一次独立的 LM head。可检查每个分片 YAML 的 network.inet 与 network.iwt 找到真实的图和权重。检查输入放在 imgs/chksum0，本实验另行使用同一组真实推理输入。

## 分片接口

下表用 SDK 的轴顺序给出 **单个分片** 的张量形状。在本实验的 Caffe CPU forward 中，包括 scale 和 valid_till 在内的 blob 均以 float32 填写。dvrun_config.yaml 另指定编译时的 scale 为 int8、valid_till 为 uint16；权重 INT4 及 LM head 输出 INT32 是后续编译设置。本次比较的是编译前的浮点路径。

| 输入名称 | Prefill | Decode |
|---|---|---|
| position_ids_embedded | [1,1,128,256] | [1,1,1,256] |
| input_ids_embedded / input_ids_embedded_scale | [1,1,128,2560] / [1,1,128,40] | [1,1,1,2560] / [1,1,1,40] |
| past_keys / past_keys_scale | [1,1,1024,1024] / [1,1,1024,16] | 同左 |
| past_values / past_values_scale | [1,1,1024,1024] / [1,1,16,1024] | 同左 |
| valid_till | [1,1,1,1] | [1,1,1,1] |
| deepstack_input_ids_embedded / deepstack_input_ids_embedded_scale | 仅第 0、1、2 层，[1,1,128,2560] / [1,1,128,40] | 无此入口 |
| past_values_requant / past_values_requant_scale | 无此入口 | [1,64,1,1024] / [1,1,1,1024] |

position_ids_embedded 携带 cos/sin 数值；每个位置 256 个数。嵌入和 K 按通道每 64 个值分组，V 按 token 轴每 64 个位置分组。因此 K 和 V 的数值形状相同，但 scale 形状不同。每个块和每层使用对应的视觉残差，纯文本位置填零。

缓存由运行程序管理。验证脚本先执行分片的 K/V 投影，将该层新产生的 K/V 写入当前绝对 token 位置，然后继续执行读取 past_keys 和 past_values 的注意力。Decode 的 V 按 64 个位置成组回写。FP32 主 K/V 在每层共占 8 MiB，36 层合计 288 MiB；这些数值不含 scale、临时张量和运行时开销。

| 阶段 | 输出名称 | 数值 / scale 形状 |
|---|---|---|
| Prefill，第 0…2 层 | deepstack_residual / deepstack_residual_scale | [1,1,128,2560] / [1,1,128,40] |
| Prefill，第 3…35 层 | mlp_residual / mlp_residual_scale | [1,1,128,2560] / [1,1,128,40] |
| Prefill，每层新 K | k_norm_reshape_permute_rope_reshape_permute / k_norm_scale_reshape_permute_rope_reshape_permute | [1,1,128,1024] / [1,1,128,16] |
| Prefill，每层新 V | v_proj_reshape_permute / v_proj_scale_reshape_permute | [1,1,128,1024] / [1,1,2,1024] |
| Decode，每层隐藏状态 | mlp_residual / mlp_residual_scale | [1,1,1,2560] / [1,1,1,40] |
| Decode，每层新 K | k_norm_permute_rope / k_norm_scale_permute_rope | [1,8,1,128] / [1,8,1,2] |
| Decode，每层 V 回写块 | requant / requant_scale | [1,64,1,1024] / [1,1,1,1024] |
| LM head，输入 | input_ids_embedded / input_ids_embedded_scale | [1,1,1,2560] / [1,1,1,40] |
| LM head，输出 | lm_head / lm_head_scale | [1,151936] / [1,151936] |

## 复现精度对比

在 SDK 根目录执行。第一步复用 mode1_validation 的五个样本和官方模型的视觉输入，保存相同 token 前缀下的 Transformers 和 Mode 1 ONNX 完整 logits。第二步加载新生成的真实 Caffe 图和 .caffemodel，按层执行 prefill、使用分片自身的 KV 执行 decode，再运行 LM head。所有预测使用 argmax，比较包括 EOS 的 11 个位置。teacher forcing 使首个分歧后的 logits 仍对应同一前缀。

```bash
fixtures=$(mktemp -d)
model=/Users/liwiii/huggingface/Qwen/Qwen3-VL-4B-Instruct
/Users/liwiii/venv/nxp/bin/python mode2_validation/prepare.py "$model" mode1_onnx/model.onnx "$fixtures"
run_check() {
  docker run --rm --platform linux/amd64 --user "$(id -u):$(id -g)" --group-add 0 \
    -e HOME=/tmp -e GLOG_minloglevel=2 -e OPENBLAS_NUM_THREADS=4 -e OMP_NUM_THREADS=4 \
    -e PYTHONPATH=/dv2/python:/dv2/python/caffe/proto \
    -v "$PWD/mode2_1024/llm:/m:ro" -v "$PWD/mode1_onnx:/onnx:ro" \
    -v "$fixtures:/refs:ro" -v "$PWD/mode2_validation:/check" \
    aracompiler:0.2 python -u "/check/$1" "${@:2}"
}
run_check check_weights.py /onnx/model.onnx /m /check/weights.json
run_check check_splits.py /m /refs /check/results.json
run_check check_splits.py /m /refs /check/position_aware.json --position-aware
```

上述函数适用于 Bash。报告先落盘，若任一阈值超限，脚本返回 1。沿用 Mode 1 的逐位置阈值：top-1 完全相同，最大绝对误差 ≤ 0.002、RMSE ≤ 0.0002、余弦相似度 ≥ 0.999999，对两个参考分别判断。

默认检验直接使用生成的 IMAGE 浮点 RoPE 表和 prefill 图的 attention_softmax。额外的位置对照实验使用与当前 prompt 对应的 RoPE，并用同一套 decode 图的 attention_softmax 按绝对 KV 长度逐行计算第二个 prefill 块的注意力。这个对照用来区分生成权重和宿主位置处理对结果的影响。权重文件及图文件在两次运行中均以只读方式打开。
