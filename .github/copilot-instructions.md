ROLE: You are my pair ML engineer. I run the code; you write it. 2-day budget.
OS: Linux. Docker is installed. Training on Colab T4. Inference is CPU-only.

PROJECT: PromptShield — a fine-tuned prompt-injection/jailbreak classifier that screens
LLM input, served via FastAPI with a Streamlit demo, deployed as a Docker HF Space.

NON-NEGOTIABLE RULES
- NEVER invent, estimate, or "illustrate" a metric. All numbers come from artifacts/metrics.json
  produced by a real run. If unknown, write TODO.
- Never silently substitute a dataset, model, split, or metric. If blocked, tell me and offer options.
- Every script accepts --limit N and --dry-run so it runs on CPU in <60s.
- After any data step, print shapes, label distribution, and 3 sample rows.
- Pin exact versions in requirements files after the first successful run. seed=42 everywhere.
- Small increments: files + exact run commands + expected output. No refactors unless asked.
- If a fix risks >45 min of debugging, stop and propose a simpler fallback.
- Defensive security research only. Do not generate new harmful payloads.

DATA (verified IDs — do not change)
1. PRIMARY: neuralchemy/Prompt-injection-dataset [Apache 2.0]
   config "full": train 14036 / val 941 / test 942 -> transformer training
   config "core": train 4391 / val 941 / test 942 -> classical baselines
   cols: text, label(1=malicious,0=benign), category, severity, group_id, augmented, source, tags
2. CROSS-SOURCE EVAL ONLY: S-Labs/prompt-injection-dataset [MIT], test split (~2.1k, balanced)
   cols: text, label
3. OVER-REFUSAL EVAL ONLY: allenai/wildjailbreak [gated]
   load_dataset("allenai/wildjailbreak","eval",delimiter="\t",keep_default_na=False)
   cols: vanilla, adversarial, tactics, completion, data_type
   (~210 adversarial_benign, ~2000 adversarial_harmful)
   It labels HARMFULNESS, not injection. NEVER merge into training. Optional: skip if gate fails.
Do NOT use deepset/prompt-injections (662 rows, inflates scores).

AUDIT (required before training): verify no group_id spans splits; hash-compare primary train
against both external eval sets (normalize: lowercase, strip whitespace + zero-width chars);
remove overlaps from eval; log counts to artifacts/audit.json.

MODELS
- Baseline A: TF-IDF + LogisticRegression on "core"; save pipeline to artifacts/baseline_tfidf.joblib
- Baseline B: protectai/deberta-v3-base-prompt-injection-v2, zero training
- MAIN: microsoft/deberta-v3-small, max_len=256, batch 32, lr 2e-5, 3 epochs,
  warmup_ratio 0.1, fp16=True (T4 = Turing, NO bf16), early stop on val PR-AUC
- STRETCH: distill to distilbert-base-uncased (T=2, alpha=0.5), ONNX + int8 dynamic quant

EVAL: pick threshold on VAL for >=95% malicious recall, FREEZE it, never tune on test.
Report precision/recall/F1/ROC-AUC/PR-AUC/confusion matrix/FPR/FNR on: primary test,
cross-source test, and wildjailbreak adversarial_benign subset. Plus per-category recall,
and CPU latency p50/p95 (1 thread, batch 1, 200 runs after 20 warmup).
Save: metrics.json, threshold.json, predictions.csv, per_category.csv, latency.json,
errors.csv, results_table.md, and PNG plots.
Published reference (cite as dataset-card numbers): DeBERTa-v3-small 95.1 acc/0.959 F1/8.5 FPR;
RandomForest 96.3/0.969/6.9.

ENV: Colab has torch preinstalled — do NOT pin/upgrade torch in requirements-train.txt.
Local/Docker: install CPU torch from https://download.pytorch.org/whl/cpu.
Tokenizer needs sentencepiece + protobuf. If TrainingArguments rejects eval_strategy vs
evaluation_strategy, check the installed transformers version instead of guessing.
Artifacts + weights go to HF Hub via huggingface_hub, not git.

DOCKER (core deliverable, test locally before deploying)
- python:3.11-slim base
- install CPU torch from the CPU wheel index, then requirements-app.txt (no torch pin inside)
- non-root user UID 1000, writable HF cache (HF_HOME=/tmp/hf), no root at runtime
- start.sh launches uvicorn on 127.0.0.1:8000 and Streamlit on 0.0.0.0:7860
- Streamlit calls the API over HTTP (API_URL env, default http://127.0.0.1:8000); it must NOT
  load the model itself (avoid double memory)
- EXPOSE 7860, CMD ["./start.sh"], healthcheck hits /healthz
- .dockerignore excludes .venv, .git, models, checkpoints, notebooks, caches
- target image <3GB; model downloads from HF Hub at runtime (public repo)

LAYOUT: data.py audit.py baselines.py train.py evaluate.py push_artifacts.py distill.py
export_onnx.py | app/(main.py,model.py,schemas.py,guard.py) | demo/streamlit_app.py | tests/
