# DreamProver: Evolving Transferable Lemma Libraries

Code for the COLM 2026 paper [*DreamProver: Evolving Transferable Lemma Libraries
via a Wake-Sleep Theorem-Proving Agent*](https://openreview.net/pdf?id=Hxx87mfERN)

## Setup

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

## Example Library Learning
Example library learning for 100 AIPS problems over 5 cycles. Evaluation includes 92 567NEQ, 42 ChenNEQ and 20 MO-INT problems.
```bash
python scripts/setup_tbps.py --lean "$(ELAN_TOOLCHAIN=leanprover/lean4:v4.15.0 elan which lean)" \
  --artifacts artifacts/tbps-v4.15
python scripts/setup_embeddings.py
python scripts/prepare_inequalities.py --check-syntax
dreamprover train --config configs/pipeline/inequalities.yaml --max-cost-usd 1000
dreamprover evaluate --config configs/pipeline/inequalities.yaml --max-cost-usd 1000
```

The budget limit can be controlled by `--max-cost-usd`. Monitor and Report:


```bash
dreamprover monitor --run-dir runs/inequalities/gpt-6-luna
dreamprover report --run-dir runs/inequalities/gpt-6-luna
```

### Custom Datasets:

Store dataset as JSONL file with `name`, `header`, `formal_statement` and
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
