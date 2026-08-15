# Experiments

`prepare_workloads.py` prepares the real-data workloads, and
`run_experiments.sh` runs the additional experiments. By default, experiments
use 16 threads and one measured run; logs are written to
`experiments/results/`.

## Prepare the data

```bash
./experiments/run_experiments.sh prepare-job
./experiments/run_experiments.sh prepare-job-scales
./experiments/run_experiments.sh prepare-snap
./experiments/run_experiments.sh prepare-tpch
```

The JOB commands prepare the IMDb input for JOB 13d and its five data scales.
The SNAP command prepares the Email-EuAll join-tree workloads.

## Run the experiments

| Command | Experiment |
|---|---|
| `job13d` | JFYan, ParYan, and ObliYan on the full JOB 13d input |
| `job13d-scale` | One JOB 13d data scale selected by `JOB13D_PERCENT` |
| `snap` | All Email-EuAll join-tree cases |
| `nonoblivious` | JFYan and NonObliJFYan |
| `memory` | Peak-memory measurements on the representative workloads |
| `benchmark` | TPC-DS Q18/Q85 and TPC-H Q9 |

```bash
THREADS=16 ./experiments/run_experiments.sh job13d
THREADS=16 ./experiments/run_experiments.sh snap
THREADS=16 ./experiments/run_experiments.sh nonoblivious
THREADS=16 ./experiments/run_experiments.sh memory
```

Individual scale points can also be run directly:

```bash
JOB13D_PERCENT=60 ./experiments/run_experiments.sh job13d-scale
SCALE=30 ./experiments/run_experiments.sh q18-jfyan
SCALE=30 ./experiments/run_experiments.sh q18-nonobli
SCALE=30 ./experiments/run_experiments.sh q18-memory
```

The corresponding Q85 commands are `q85-jfyan`, `q85-nonobli`, and
`q85-memory`. Use `./experiments/run_experiments.sh --help` for the complete
command list and optional environment variables.

## Figures

The plotted values are in `experiments/plots/input/`. Generate the paper
figures with:

```bash
python3 experiments/plots/paper_figures.py
```

The generated files are written to `experiments/figures/`.
