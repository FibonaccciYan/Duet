# NarrativeQA 长上下文速度测试集（80 条）

2026-09-17 在 n33 集群 CPU 上构建，结果下载至本目录。

## 来源与筛选

- 来源：已有 LongBench NarrativeQA test JSONL，共 200 条；不是完整 NarrativeQA 的全部划分。
- 对数据中保存的完整 `prompt` 编码；不额外应用 chat template，不添加 special tokens。
- 长度定义：分别用 SDAR-8B-Chat-b32、LLaDA2.1-mini 的 tokenizer 编码。
- 筛选：两种 tokenizer 下原始 prompt 都严格大于 32768 tokens；共有 85 条满足。
- 抽样：Python `random.Random(42).sample(..., 80)` 无放回抽样，然后按原始行号排序。
- 两种模型、三个长度使用相同的 80 个 source_id。不同 tokenizer 的截断边界可能不同。
- 80 条指独立问答样本；不额外保证 80 个不同故事或文档。
- 截取规则：`original_ids[:N//2] + original_ids[-N//2:]`，不插入分隔符。
- `N` 分别为 8192、16384、32768；保留的前后 token 数完全相等。
- 这是对完整 prompt（包含任务说明、故事、问题）的头尾截取，并非只截故事正文。

## 文件

|目录/文件|内容|
|---|---|
|`sdar/8192.jsonl`、`16384.jsonl`、`32768.jsonl`|SDAR tokenizer 精确长度版本，各 80 条|
|`llada21/8192.jsonl`、`16384.jsonl`、`32768.jsonl`|LLaDA2.1 tokenizer 精确长度版本，各 80 条|
|`selected_originals.jsonl`|80 条原始完整记录、原始行号和两套 token 长度|
|`selection_audit.json`|200 条候选的 token 长度、资格和最终选择索引|
|`manifest.json`|来源 SHA-256、规则、tokenizer 版本及构建文件校验和|
|`tokenizers/`|本次使用的 tokenizer 文件快照|
|`build_dataset.py`|可重建脚本，在集群执行|
|`verify_dataset.py`|下载后完整性与切片验证|
|`load_samples.py`|读取和为生成预算预留窗口的辅助函数|

## 正确使用

**速度测试以 `input_ids` 为准。** `prompt` 是这些 tokens 的解码文本，便于查看；
由于 BPE 拼接边界或字节边界，解码后再编码不一定得到原来的 token IDs。
每条记录含 `text_roundtrip_exact`、`text_retokenized_tokens`，没有隐藏这一差异。
因此不要在加载时重新 tokenize，不要再次套 chat template。

```python
import torch
from load_samples import load_samples

for item in load_samples("sdar", 8192):
    inputs = torch.tensor([item["input_ids"]], dtype=torch.long, device="cuda")
    # 将 inputs 交给对应模型生成；这份数据集不修改任何算法参数。
```

32768 是纯输入长度。如果模型最大窗口为 32768，生成 256/768 tokens 时，
必须另外预留生成空间，不能直接输入 32768 tokens 再声称未超窗：

```python
from load_samples import reserve_generation_window
item = reserve_generation_window(item, gen_length=768, max_window=32768)
# 仅在超窗时进一步头尾截取；32768 -> 32000，原文件不变。
```

保存的 answers 便于追溯；头尾截断可能删除作答证据，不能直接将此集合视为
未经改变的 NarrativeQA 质量基准。选集不依据模型预测或速度结果。

集群副本：`/data0/gs/dataset_preparation/narrativeqa_speed80_20260917/dataset`
构建使用 `CUDA_VISIBLE_DEVICES=""`，没有使用或调整 GPU 占用程序。
