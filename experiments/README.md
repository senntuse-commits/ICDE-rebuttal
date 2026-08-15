# Experiments

`prepare_workloads.py` prepares the real-data workloads, and
`run_experiments.sh` runs the additional experiments.

## Workloads

```bash
python3 experiments/prepare_workloads.py job13d-scales --download --force
python3 experiments/prepare_workloads.py snap-acyclic --download --force
```

## Run

```bash
THREADS=16 ./experiments/run_experiments.sh job13d
THREADS=16 ./experiments/run_experiments.sh snap
THREADS=16 ./experiments/run_experiments.sh nonoblivious
THREADS=16 ./experiments/run_experiments.sh memory
```

Use `./experiments/run_experiments.sh --help` for individual workloads and
options.

## Figures

```bash
python3 experiments/plots/paper_figures.py
```
