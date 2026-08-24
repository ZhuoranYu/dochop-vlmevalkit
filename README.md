# DocHop: Benchmarking Out-of-domain Multi-hop Reasoning in Information-Dense Documents

Zhuoran Yu, Le Thien Phuc Nguyen, Jaden Park, Xinyi Gu, Zexue He, Soochahn Lee, Rogerio Feris, Yong Jae Lee

*ICML 2026*

This repo is a fork of [open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit) that adds the **DocHop** benchmark and its evaluation harness. All the credit for the underlying evaluation toolkit — model wrappers, inference pipeline, dataset framework — goes to the original VLMEvalKit project.

For environment setup, supported models/benchmarks beyond DocHop, and general toolkit usage, see the original upstream README, kept here at [`docs/VLMEvalKit_README.md`](docs/VLMEvalKit_README.md).

## DocHop Benchmark

**DocHop** is a document-grounded chart QA benchmark: each example is a document page containing text and one or more charts, with a question that requires cross-referencing both. 2,074 examples across 6 task types (`value_retrieval`, `counting`, `numeric_reasoning`, `ranking`, `hypothetical`, `fact_checking`), spanning reasoning `depth` 2-5 and `chart_num` 2/3/4/6.

- Dataset class: `vlmeval/dataset/dochop.py` → registers as `DocHop`
- Data: released publicly on HuggingFace at [`zhuoranyu336/dochop`](https://huggingface.co/datasets/zhuoranyu336/dochop); `DATASET_URL`/`DATASET_MD5` in `dochop.py` point there
- Custom eval logic (`evaluate_heuristic` in `dochop.py`) breaks accuracy down by task/depth/chart_num, saved as `<eval_file>_acc.csv`

### Setup

```bash
conda create -n dochop python=3.10 -y
conda activate dochop
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` already includes what DocHop's model configs need (`google-genai` for Gemini, `requests` for the raw-HTTP Claude client, `openai` for GPT). No extra `anthropic` SDK dependency — `vlmeval/api/claude.py` talks to the Anthropic API directly over `requests`.

`transformers` is pinned to `4.49.0` because `vlmeval/vlm/__init__.py` eagerly imports every VLM wrapper class (including ones DocHop doesn't use), and newer `transformers` releases have removed symbols some of those wrappers import at module load time — so an unpinned/latest `transformers` breaks `import vlmeval` entirely, even for API-only usage. This pin is only verified for the DocHop/API-model path below; for open-source/local models, see the version table linked in "Running open-source / local models".

If this is a machine with GPUs (for running open-source/local models rather than just API models), `torch`/`torchvision` in `requirements.txt` need to match that machine's CUDA setup — check `nvidia-smi` and install the matching PyTorch build before `pip install -r requirements.txt`, rather than assuming the pinned versions here are CUDA-correct for the new box.

### API keys

Create a `.env` file at the repo root (gitignored, never commit it) with whichever of these you need:

```
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=...
ANTHROPIC_BACKEND=official
GOOGLE_API_BACKEND=...
DASHSCOPE_API_KEY=...
```

`vlmeval/__init__.py` loads `.env` automatically on import (via `load_env()`), so these become available as env vars for every run without exporting them manually.

### Data location (optional)

VLMEvalKit caches datasets in `$LMUData` (defaults to `~/LMUData` if unset). On first run, since `DocHop` is public on HuggingFace, it will auto-download `DocHop.tsv` into that directory for you — no manual data placement needed. Only set `LMUData` if you want the cache to live somewhere other than your home directory (e.g. a larger disk):

```bash
export LMUData=<path to your preferred cache dir>   # optional
```

If you're on a machine with its own shared HuggingFace cache, you may also want:

```bash
export HF_HOME=<path to your HF cache>   # optional
```

### Running an eval

For example, with a proprietary API model:

```bash
python run.py --data DocHop --model GeminiPro2-5 --verbose --reuse
```

`--reuse` matters: it resumes from a per-example temp pickle (`{model}_{dataset}_supp.pkl`) if a previous run was interrupted, instead of re-paying for already-answered examples. Note: examples that failed mid-run (e.g. ran out of API credits) get cached as failures too and will be silently skipped on a naive rerun — see the note in `vlmeval/inference.py` around `ignore_failed` if that happens; the fix is to strip `FAIL_MSG` entries from that temp pickle before rerunning.

### Running open-source / local models

`requirements.txt` pins `transformers==4.49.0`, which is only verified for the proprietary API models above. Several open-source model families in `vlmeval/config.py` need specific, mutually incompatible `transformers` versions (and some need extra backends like LMDeploy for larger checkpoints) — see the "Transformers Version Recommendation" section in [`docs/VLMEvalKit_README.md`](docs/VLMEvalKit_README.md) for known-good versions per model family, and set up a separate environment per family as needed.

### Models registered for DocHop

See `vlmeval/config.py` for the full list of config names available to `--model` (e.g. `GeminiPro2-5`, `Gemini3-1-Pro-max`, `Claude5_Sonnet`, `Claude4_5_Haiku`) and their underlying model IDs / generation settings.

`vlmeval/api/claude.py` supports both Claude thinking mechanisms: `enable_thinking`+`budget_tokens` (pre-adaptive models) and `effort` (adaptive-only models, sets `output_config.effort`). Check which mechanism the target Claude model actually supports before assuming one works — passing `budget_tokens` to an adaptive-only model (e.g. `claude-sonnet-5`) returns a 400.
