# DreamProver: Evolving Transferable Lemma Libraries

Code for the COLM 2026 paper [*DreamProver: Evolving Transferable Lemma Libraries
via a Wake-Sleep Theorem-Proving Agent*](https://openreview.net/pdf?id=Hxx87mfERN): recursive Lean proving, lemma library
learning, and evaluation on new problems.

## Getting Started

Python 3.10+, Git and [Elan](https://github.com/leanprover/elan) are required.
Set `OPENAI_API_KEY` in your experiment shell. Both model roles use `gpt-6-luna`.

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python scripts/setup_dependencies.py
python -m pip install -r requirements.txt
python -m pip install -e '.[sleep,data]'
```

Python code lives in `src/dreamprover/`. Use `python -m dreamprover` for the CLI.
TBPS and Kimina are pinned inside `vendor/` by [dependencies.lock.json](dependencies.lock.json).

### Start the Lean verifier

```bash
export PATH="$HOME/.elan/bin:$PATH"
python scripts/setup_kimina.py --lean-version v4.15.0 --mathlib \
  --metadata-dir artifacts/kimina-v4.15
python scripts/start_kimina.py --setup-dir artifacts/kimina-v4.15 --port 10002
```

Keep the server running in a separate terminal.

## Experiments

```bash
python scripts/setup_tbps.py --lean "$(ELAN_TOOLCHAIN=leanprover/lean4:v4.15.0 elan which lean)" \
  --artifacts artifacts/tbps-v4.15
python scripts/setup_embeddings.py
python scripts/prepare_inequalities.py --check-syntax
dreamprover train --config configs/pipeline/inequalities.yaml --max-cost-usd 25
dreamprover evaluate --config configs/pipeline/inequalities.yaml --max-cost-usd 25
```

Training uses 100 AIPS problems over five cycles. Evaluation compares empty and
learned libraries on 92 567NEQ, 42 ChenNEQ and 20 MO-INT problems. The $25 cap
above is an example. Tune it with `--max-cost-usd`. Spending is shared across
both commands, which resume from checkpoints. Settings are in
[inequalities.yaml](configs/pipeline/inequalities.yaml). Outputs stay in `runs/`.

```bash
dreamprover monitor --run-dir runs/inequalities/gpt-6-luna
dreamprover report --run-dir runs/inequalities/gpt-6-luna
```

### Other datasets

Store examples as JSONL with `name`, `header`, `formal_statement` and
`informal_prefix`. Point the pipeline config at disjoint training and test files,
and use a matching Lean project. For a single proving run, use
`dreamprover prove --config configs/run.yaml data.file_path=YOUR.jsonl`.

## Citation

```bibtex
@inproceedings{zhang2026dreamprover,
  title={DreamProver: Evolving Transferable Lemma Libraries via a Wake-Sleep Theorem-Proving Agent},
  author={Zhang, Youyuan and Sun, Jialiang and Bi, Hangrui and Geng, Chuqin and Ma, Wenjie and Li, Zhaoyu and Si, Xujie},
  booktitle={Conference on Language Modeling}, year={2026}
}
```
