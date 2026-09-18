# 统一测试脚本

推荐入口为 `scripts/unified/`。原脚本保存在 `scripts/original/`，算法实现在
`src/reference/` 和 `src/optimized/`。S Mode 专用入口已删除，历史结果未删除。

## 入口清单

| 方法 | LongBench 质量 | 短 bench 质量 | 速度全量 |
|---|---|---|---|
| Dense | `dense_longbench_quality.sh` | `dense_short_quality.sh` | `dense_speed_full.sh` |
| Sparse | `sparse_longbench_quality.sh` | `sparse_short_quality.sh` | `sparse_speed_full.sh` |
| LoSA | `losa_longbench_quality.sh` | `losa_short_quality.sh` | `losa_speed_full.sh` |
| FOCUS | `focus_longbench_quality.sh` | `focus_short_quality.sh` | `focus_speed_full.sh` |

四个短 bench 入口和速度入口**全部使用 optimized**。LongBench 默认保持原 QA
脚本的 reference 方法语义（dense 为 Efficient optimized 兼容入口），可用
`--implementation optimized` 显式切换。不能把旧脚本默认 reference 与新短 bench
optimized 的输出差异，当成脚本重构的逐位等价验证。

## 先查看参数，不启动模型

从任意目录运行 `.sh` 都会自动定位仓库：

```bash
bash scripts/unified/sparse_speed_full.sh --help
bash scripts/unified/sparse_speed_full.sh --model both --print-config
bash scripts/unified/losa_short_quality.sh --model sdar --dry-run
```

启动器解释器使用 `PYTHON`，默认 `python3`；`--python` 指定子任务解释器。
n33 示例：

```bash
export PYTHON=/data0/ysy/anaconda3/envs/sparse/bin/python
```

**真正执行必须显式提供 `--gpu`。** 无此参数只允许查看计划。参数优先级为
CLI > `--config` JSON 默认值 > `scripts/configs/defaults.json`。
不会让外部遗留的 `THRESHOLD`、`GEN_LENGTH` 等变量悄悄覆盖短 bench 计划；
修改这些值请用 CLI 或 `--set`。HF 缓存、离线环境变量仍可使用。

## 默认协议：哪些保持，哪些明确改变

| 项目 | 原来源与统一入口行为 |
|---|---|
| SDAR confidence threshold | **统一默认 0.95**，这是用户明确修正；原脚本历史 0.85 不再是新入口默认 |
| LLaDA threshold / editing | 0.7 / 0.5 |
| 短 bench 实现 | **统一 optimized**，用户明确要求；原实现仍保留在 original |
| 短 bench Dense 任务 | 原 dense wrapper：GSM8K、HumanEval、MMLU、Math |
| 短 bench 其他方法任务 | 原 `eval.sh` 默认 GSM8K；全部任务请显式传 `--tasks` |
| LLaDA 短 bench 生成长度 | 16384，所有任务；**不是 256** |
| SDAR Dense 短 bench 生成长度 | 4096，所有任务 |
| SDAR 其他短 bench 长度 | GSM8K 4096，HumanEval 768，MMLU 128，Math 512，Math500 2048 |
| MMLU few-shot | **四种方法、两模型统一为 5-shot**，这是用户明确修正；其他任务为 0 |
| SDAR 短 bench remasking | Dense: low_confidence_static；FOCUS: low_confidence_dynamic；Sparse/LoSA: sequential |
| 短 bench Sparse 参数 | 继承 `eval.sh`：LLaDA interval4/ratio0.7，SDAR interval1/ratio0.5；SDAR prefix 默认关闭 |
| optimized LoSA GQA | group_mean，使用冻结 optimized 语义；不能误继承旧评测适配器的 per_query_head |
| 短 bench 评分 | 仍由 `eval_instruct/eval.sh` 和 lm-eval 执行，模板/任务 YAML 不换 |
| HumanEval | 保留官方评分与缩进归一化诊断，后者不替代官方成绩 |
| 速度协议 | NarrativeQA80；两个 warmup；三个 E2E repeats；独立 event pass |

SDAR 的 remasking **没有因 threshold 改为 0.95 而自动统一成 dynamic**。
需要不同策略时显式传 `--remasking-strategy`。
保持协议的含义是“对应原入口的默认值保持”，不是抹平原入口之间的差异。

## 短 bench

```bash
# 最小 smoke：每个实际任务数据集一条；32-token 是显式 smoke 覆盖。
bash scripts/unified/sparse_short_quality.sh \
  --model llada21 --gpu 0 --stage smoke \
  --tasks gsm8k humaneval mmlu math --limit 1 --gen-length 32 \
  --offline --output results/sparse_short_smoke

# 正式任务，不继承上面 smoke 的短输出长度。
bash scripts/unified/losa_short_quality.sh \
  --model sdar --gpu 2 --stage full \
  --tasks gsm8k humaneval mmlu math --output results/losa_short
```

`--limit 1` 由 lm-eval 按实际子数据集执行：MMLU 的各 subject、Math 的各分支
各取一条，不是修改任务定义只保留一个 subject。`--gen-length 32` 的 smoke 分数
没有正式质量意义。HumanEval 会运行生成代码，沿用原脚本的 code-eval 流程。

Dense 短 bench 默认 `--stage all`（先 smoke 后 full），其他入口默认 full；
可显式用 `--stage smoke/full/all`。Smoke 不会自动启动 full，除非选择 all。
`--offline` 使用已有 Hugging Face 缓存；缺缓存会失败，不会伪造样本。
它同时设置 datasets、hub、evaluate 三个离线开关。旧 datasets 版本读取新缓存时，
公共入口仅对 primitive `List[Value]` 的元数据提供等价 `Sequence[Value]` 兼容，
不修改样本或磁盘缓存；嵌套结构不做猜测转换，直接报错。

常用覆盖：`--threshold`、`--editing-threshold`、`--gen-length`、`--fewshot`、
`--block-length`、`--steps`、`--temperature`、`--port`、`--seed`。
短 bench EOS 行为保持评测适配器原语义，不提供静默切换。

方法参数示例：

```bash
bash scripts/unified/sparse_short_quality.sh --model sdar --gpu 0 \
  --set PREFIX_SPARSE=true --set PREFIX_TOKEN_BUDGET=512 \
  --set SPARSE_DLM_SELECTION_INTERVAL=4 --print-config
```

短 bench `--set` 使用原环境键，如 `SPARSE_DLM_RATIO`、`FOCUS_ALPHA`、
`PAPER_LOSA_TOKEN_BUDGET`、`PAPER_LOSA_GQA_MODE`、`MOE_EXPERT_PATCH`。
不支持的键报错。最终模型参数与实际 patch 名称写入运行日志/评测结果。

## LongBench

```bash
bash scripts/unified/focus_longbench_quality.sh \
  --model sdar --gpu 2 --implementation optimized \
  --data-dir /path/to/longbench --stage smoke --limit 1 \
  --max-context-tokens 1024 --output results/focus_longbench_smoke
```

默认任务是 hotpotqa、triviaqa、narrativeqa、qasper、multifieldqa_en；
生成长度分别为 32、32、128、128、64，gov_report 可选，长度 512。
保留原 QA tokenization/chat template、头尾截断、EOS early-stop、seed42 和评分器。
LongBench 正式数据未随本次同步；`--data-dir` 或 `LONGBENCH_DATA` 指定其位置。

- `--profile default`：原五任务 QA 协议。
- Sparse `--profile strict-budget`：原 256/512/1024 严格预算矩阵，默认 optimized。
- Dense `--profile all-tasks`：原 LLaDA manifest 全任务入口，不支持 SDAR。
- `--resume-from predictions.jsonl`：沿用原断点文件语义，写入新的输出目录。
- 方法参数使用小写，如 `--set prefix_token_budget=512`、`--set losa_gqa_mode=group_mean`。
- LongBench 不用 `--gen-length` 强行统一不同任务的生成长度。

## 速度全量

数据已实际放入仓库：`data/narrativeqa_speed80/`，不是符号链接或外部路径占位符。
包括两套 tokenizer、六个精确 input_ids JSONL、manifest、来源样本与构建审计。
这些文件不在 `.gitignore` 中，应与代码一起提交；无需额外下载测速文本。

```bash
# 每个方法只测自己，不偷偷重跑 dense。
bash scripts/unified/dense_speed_full.sh --model both --gpu 0 \
  --output results/dense_speed
bash scripts/unified/sparse_speed_full.sh --model both --gpu 2 \
  --output results/sparse_speed

# 最小测速 smoke。
bash scripts/unified/losa_speed_full.sh --model sdar --gpu 0 \
  --contexts 8192 --stage smoke --samples 1 --warmups 2 --repeats 1 \
  --output results/losa_speed_smoke
```

正式默认：每模型/长度80条，contexts8192/16384/32768，生成256，block32/steps32，
temperature0，EOS提前返回关闭。32K 自动预留生成窗口：32512+256=32768。
输入以保存的 IDs 为准，不解码后重新编码。

Sparse 正式默认开启 query+prefix、budget256，其余采用冻结运行时默认；
LoSA group_mean/page16/active5/budget256/fused state/graph，shared-KV关闭；
FOCUS alpha1.5，LLaDA native Flash、SDAR默认 SDPA；
Dense 固定 Efficient Attention。公共 MoE/RMSNorm 继续由原运行时安装。

`--samples`、`--contexts`、`--warmups`、`--repeats`、`--gen-length` 可修改，
越窗/非 block 对齐输入会报错。方法覆盖例如 `--set ratio=0.7`、
`--set selection_interval=4`、`--set graph_replay=false`、`--set alpha=1.5`。

计时保持原驱动：E2E 含 prefill/选择/缓存/生成/收尾，不含加载、读取、保存；
event 单独运行并校验 tokens 与逐 block step 数。统一单步汇总仅排除 block 首步，
包含 editing 和 LoSA initialization/prime/capture/replay，使用总时间/总数量，
不沿用旧 Sparse `steady` 字段的“排除前两步”统计。

## 输出与资源安全

每个入口保存 `plan.json`、逐 job 日志、`status.json`、`execution.json`、
`summary.json`、`summary.csv`、`REPORT.md`，全部成功才生成 `COMPLETE.json`。
测速 cell 继续保存 manifest、warmup、sample/partial JSON、tokens/文本、step_records。
质量任务的官方结果保留在原 evaluator 输出中，不用新汇总替换原评分。

拒绝复用非空输出目录。没有 `--run-in-background` 隐式启动行为；默认前台执行，
可由用户显式用 nohup 等管理。中断会终止本入口启动的子进程组，不停止其他任务。

GPU0/GPU2：正常用完释放，**不启动 holder**。只有显式 `--gpu 5 --restore-holder`
才会调用原 GPU5 wrapper，正常/失败退出均尝试恢复既有 holder。GPU忙时拒绝运行，
不会抢占或杀死未知进程。输出目录中保存失败状态，不把部分结果标记为完成。

## 验证

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python -m pytest tests -q
python data/narrativeqa_speed80/verify_dataset.py
git check-ignore data/narrativeqa_speed80/sdar/32768.jsonl  # 应无输出
```

数据 manifest 的16项哈希会由回归测试核验。GPU smoke 的显式缩短参数、
成功/失败日志与实际 optimized patch 记录单独保存；通过 smoke 不等于质量达标。
