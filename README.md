## `llm-compressor` W8A8

In this example, we will apply weight and activation quantization to `Llama-3-8B-Instruct`.

## Apply Compression

### Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install git+https://github.com/vllm-project/llm-compressor.git
```

### Run `GPTQ` and `SmoothQuant`

```bash
python3 apply w8a8.py
```

## Evaluate and Benchmark

### Install

```bash
python3 -m venv vllm-venv
pip install vllm==0.5.1 lm_eval==0.4.3
```

### Run `lm-eval`

```bash
MODEL=Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token
TP_SIZE=1
FEWSHOT=5
LIMIT=1000
BATCH_SIZE="auto"

lm_eval --model vllm \
  --model_args pretrained=$MODEL,tensor_parallel_size=$TP_SIZE,distributed_executor_backend="ray" \
  --tasks gsm8k --num_fewshot $FEWSHOT --limit $LIMIT \
  --batch_size $BATCH_SIZE
```

Results:
```bash
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.762|±  |0.0135|
|     |       |strict-match    |     5|exact_match|↑  |0.763|±  |0.0135|
```

### Run Benchmark

Run the benchmark script to see performance.

```bash
python3 benchmark.py --model Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token
```

```bash
* ==========================================================
* Total Time:                   73.74
* Total Generations:            1000


* Generations / Sec:            13.56
* Generation Tok / Sec:         3053.14
* Prompt Tok / Sec:             7688.24


* Avg Generation Tokens:        225.13
* Avg Prompt Tokens:            566.90
* ==========================================================
```

