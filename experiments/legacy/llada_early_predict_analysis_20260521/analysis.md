# LLaDA Early-Layer Position Prediction Analysis

## Goal

验证是否可以用前几层 `hidden_states -> lm_head -> logits` 提前预测本轮可能被修改的位置，并用：

```text
prompt positions + decoded token positions + predicted mask positions
```

作为后续 attention 的参与位置，从而跳过大部分 `[mask]` 位置计算。

## Run Setup

运行目录：`/data0/ysy/sparse`

原始命令保存在 `run_command.txt`，核心配置如下：

```text
CUDA_VISIBLE_DEVICES=4
model_path=/data0/ysy/models/LLaDA2.1-mini
gen_length=128
block_length=32
steps=32
threshold=0.5
editing_threshold=0.0
temperature=0.0
local_demo_sparse_attn=False
layer_candidate_ratio=0.1 ... 1.0
```

原始输出：`ratio_sweep_raw.log`

解析结果：

- `exact_decode_recall.csv`
- `recall_by_ratio.csv`
- `layer_summary.csv`
- `selected_ratio_table.md`
- `threshold_summary.txt`

## Metrics

### exact_decode_recall

对每层 hidden state 过 `lm_head`，然后直接复用最终 generate 的 decode 规则：

```text
mask_transfer_index | editing_transfer_index
```

与最终层真实修改位置比较：

```text
exact_decode_recall = hit / final_modified_positions
```

它回答的是：**把最终解码策略原样提前套到这一层，位置决策是否一致。**

### recall@ratio

对每层 logits 计算候选 confidence，然后按 ratio 做 top-k，比较是否覆盖最终层真实修改位置：

```text
recall@ratio = |early_topk_positions ∩ final_modified_positions| / |final_modified_positions|
```

它回答的是：**这一层的 confidence ranking 是否足以作为 sparse position predictor。**

## Results

| layer | exact | r@0.3 | r@0.5 | r@0.7 | r@0.9 | r@1.0 |
|---:|---:|---:|---:|---:|---:|---:|
| 00 | 0.394 | 0.314 | 0.457 | 0.670 | 0.872 | 1.000 |
| 01 | 0.473 | 0.521 | 0.718 | 0.835 | 0.931 | 0.995 |
| 02 | 0.527 | 0.452 | 0.713 | 0.872 | 0.968 | 0.995 |
| 03 | 0.479 | 0.340 | 0.505 | 0.723 | 0.910 | 0.995 |
| 04 | 0.452 | 0.282 | 0.441 | 0.622 | 0.830 | 0.995 |
| 05 | 0.394 | 0.277 | 0.447 | 0.660 | 0.867 | 0.995 |
| 06 | 0.356 | 0.239 | 0.383 | 0.612 | 0.867 | 0.995 |
| 07 | 0.372 | 0.250 | 0.441 | 0.654 | 0.872 | 0.995 |
| 08 | 0.340 | 0.250 | 0.452 | 0.649 | 0.862 | 0.995 |
| 09 | 0.378 | 0.250 | 0.457 | 0.628 | 0.862 | 0.995 |
| 10 | 0.362 | 0.207 | 0.340 | 0.527 | 0.851 | 0.984 |
| 11 | 0.378 | 0.229 | 0.404 | 0.617 | 0.872 | 0.989 |
| 12 | 0.537 | 0.261 | 0.404 | 0.569 | 0.819 | 0.963 |
| 13 | 0.676 | 0.271 | 0.468 | 0.638 | 0.862 | 0.973 |
| 14 | 0.824 | 0.314 | 0.500 | 0.707 | 0.899 | 0.952 |
| 15 | 0.888 | 0.351 | 0.532 | 0.702 | 0.872 | 0.936 |
| 16 | 0.936 | 0.399 | 0.580 | 0.734 | 0.904 | 0.963 |
| 17 | 0.920 | 0.436 | 0.580 | 0.750 | 0.899 | 0.957 |
| 18 | 0.936 | 0.447 | 0.601 | 0.777 | 0.904 | 0.952 |
| 19 | 1.000 | 0.883 | 0.926 | 0.973 | 1.000 | 1.000 |

## Findings

### 1. 直接提前套最终 decode 逻辑不够稳定

前几层的 `exact_decode_recall` 不高：

```text
layer 1: 0.473
layer 2: 0.527
layer 3: 0.479
```

这说明 early layer 的 token-level decode 决策和最终层还不完全一致。  
因此不建议直接把最终层的 transfer/edit 决策原样提前当作 sparse 位置集合。

### 2. confidence ranking 有早期预测价值

虽然 exact 不高，但 `recall@ratio` 明显更好：

```text
layer 1, ratio=0.5: recall=0.718
layer 2, ratio=0.5: recall=0.713
layer 1, ratio=0.7: recall=0.835
layer 2, ratio=0.7: recall=0.872
```

这说明前几层已经能比较好地把最终会修改的位置排到较高 confidence 区间里。  
也就是说，early layer 更适合做 **ranking predictor**，而不是直接做 exact decode predictor。

### 3. layer 1/2 是更有实际价值的候选层

从节省计算的角度看，越早越有价值。

当前结果里：

- `layer 1` 在 `ratio=0.5` 已达到 0.718 recall。
- `layer 2` 在 `ratio=0.7` 达到 0.872 recall。
- `layer 3` 以后没有稳定优于 layer 1/2。

因此优先建议继续测试：

```text
predict_layer = 1
ratio = 0.5 / 0.7
```

如果质量下降明显，再尝试：

```text
predict_layer = 2
ratio = 0.7
```

### 4. 低 ratio 过激

`ratio=0.3` 对前几层 recall 不够：

```text
layer 1: 0.521
layer 2: 0.452
```

这意味着只保留 30% mask 位置时，容易漏掉最终真实会修改的位置。  
除非后续有补救机制，否则不建议从 `ratio=0.3` 起步。

### 5. ratio=0.9 recall 高但节省有限

`ratio=0.9` 时前几层 recall 已经很高：

```text
layer 1: 0.931
layer 2: 0.968
```

但这只跳过 10% mask 位置，计算收益有限。它更适合作为质量上限或 sanity setting，而不是最终目标。

## Recommended Predictor Rule

建议当前 sparse 选位逻辑先使用 confidence top-k：

```python
logits = lm_head(hidden_states[predict_layer])
x0, x0_p = sample_with_temperature_topk_topp(logits[:, -block_length:, :])

mask_scores = where(active_block_mask, x0_p, -inf)
k = ceil(active_mask_count * ratio)
predicted_mask_positions = topk(mask_scores, k)

selected_positions = decoded_positions + predicted_mask_positions
selected_positions = sort(selected_positions)
```

其中：

- `decoded_positions` 包括 prompt 和已经解码出的 token，始终保留。
- `ratio=1.0` 应该选择所有 active mask，使 selected positions 退化为 dense 全量位置。
- 当 `active_mask_count` 很小时，可以直接 dense fallback，避免 gather/scatter 开销大于收益。

## Conclusion

当前单 prompt 结果支持这个方向：

```text
前几层，尤其 layer 1/2，具备提前预测 query positions 的能力。
```

但它们更适合作为 confidence ranking predictor，而不是 exact decode predictor。

建议下一步实验：

1. 用 `layer 1, ratio=0.5` 跑 sparse generation，观察文本质量和速度。
2. 用 `layer 1, ratio=0.7` 作为更稳的质量配置。
3. 如质量仍不稳定，再测试 `layer 2, ratio=0.7`。
4. 扩展到多 prompt，统计平均 recall 和输出质量，避免单样本结论过拟合。
