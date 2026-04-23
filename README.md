# Enrichment Budget Optimizer

A genetic algorithm that solves the daily lead-enrichment allocation problem as a 0/1 knapsack variant. Class project for IFSA CS38-09 (Introduction to Artificial Intelligence).

Given a pool of prospected leads, each with an estimated value and a variable enrichment cost, the optimizer selects the subset to enrich that maximizes total expected value under a fixed daily budget.

The full writeup, with PEAS analysis, methodology, results, and figures, is in [`enrichment_optimizer/REPORT.md`](enrichment_optimizer/REPORT.md).

## Layout

```
enrichment_optimizer/
  generate.py       # synthetic lead pool generator
  ga.py             # genetic algorithm (from scratch)
  baselines.py      # greedy, random, and exact DP baselines
  optimize.py       # main entry: GA vs baselines comparison
  experiments.py    # mutation / population / scaling / budget sensitivity
  evaluate.py       # figures + report_sections.md
  REPORT.md         # full project report
  tests/            # 15 unit + integration tests
  data/             # generated CSVs (included in the repo)
  figures/          # generated PNGs (included in the repo)
```

## Running the pipeline

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r enrichment_optimizer/requirements.txt

python -m enrichment_optimizer.generate     # → data/leads_pool*.csv, data/budget.json
python -m enrichment_optimizer.optimize     # → comparison table + data/convergence.csv
python -m enrichment_optimizer.experiments  # → sensitivity CSVs
python -m enrichment_optimizer.evaluate     # → 8 figures + report_sections.md
pytest enrichment_optimizer/ -v             # → 15 tests pass
```

All scripts are seeded (`seed = 42` by default), so reruns produce byte-identical outputs.

## Building the PDF report

From the repo root:

```bash
cd enrichment_optimizer
pandoc REPORT.md -o REPORT.pdf --pdf-engine=xelatex
```

The figures referenced in the report live at `enrichment_optimizer/figures/`; pandoc resolves them relative to `REPORT.md`, so running pandoc from inside `enrichment_optimizer/` works.

## Results snapshot

On a 200-lead synthetic pool with a $0.2142 daily budget:

| Method | Value captured | % of DP optimum |
|---|---:|---:|
| Random (mean of 1000 trials) | $74,974 | 80.4% |
| Greedy (value/cost ratio) | $93,304 | 99.9998% |
| Genetic algorithm | $93,174 | 99.86% |
| Dynamic programming (exact) | $93,305 | 100.00% |

See the report for convergence plots, parameter sensitivity, and scaling behavior.
