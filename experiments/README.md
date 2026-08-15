# Experiments

`prepare_workloads.py` downloads and projects the additional workloads.
`run_experiments.sh` runs the benchmarks and writes timestamped logs to
`experiments/results/`.

## JOB 13d

Prepare the full IMDb workload and five data scales:

```bash
python3 experiments/prepare_workloads.py job13d --download --force
python3 experiments/prepare_workloads.py job13d-scales --download --force
```

Run the full input or one selected scale:

```bash
THREADS=16 REPEATS=1 ./experiments/run_experiments.sh job13d
JOB13D_PERCENT=60 ./experiments/run_experiments.sh job13d-scale
```

The five scale points contain the following cardinalities:

| IMDb data | Input rows | Output rows |
|---:|---:|---:|
| 20% | 4,358,244 | 133,293 |
| 40% | 8,634,195 | 269,135 |
| 60% | 12,896,390 | 399,337 |
| 80% | 17,171,737 | 537,807 |
| 100% | 21,438,043 | 670,390 |

## Email-EuAll

Prepare and run the join-tree workloads:

```bash
python3 experiments/prepare_workloads.py snap-acyclic --download --force
THREADS=16 REPEATS=1 ./experiments/run_experiments.sh snap
```

To run one case from `suite.csv`, set `SNAP_CASE` and use `snap-case`.

## TPC-DS and TPC-H

The driver provides separate commands for runtime and memory measurements:

```bash
SCALE=30 ./experiments/run_experiments.sh q18-jfyan
SCALE=30 ./experiments/run_experiments.sh q18-nonobli
SCALE=30 ./experiments/run_experiments.sh q18-memory

SCALE=30 ./experiments/run_experiments.sh q85-jfyan
SCALE=30 ./experiments/run_experiments.sh q85-nonobli
SCALE=30 ./experiments/run_experiments.sh q85-memory
```

Use `./experiments/run_experiments.sh --help` for all commands and environment
variables.

## Figures

The final plotting data and plotting program are in `plots/`:

```bash
python3 experiments/plots/paper_figures.py
```

Generated figures are written to `experiments/figures/`.
