# JFYan

Code and experiments for *Parallel Oblivious Acyclic Joins for TEE-based
Encrypted Databases*.

- [Full version](Parallel_Oblivious_Acyclic_Joins_for_TEE_based_Encrypted_Databases__Full_Version_1.pdf)
- [Original submission repository](https://github.com/senntuse-commits/Parallel_Oblivious_Acyclic_Joins_for_TEE-based_Encrypted_Databases)

This version adds:

- the NonObliJFYan baseline;
- email-EuAll and JOB 13d workloads;
- memory profiling and constrained-EPC experiments;
- the data and code used to draw the additional figures.

## Build

The project requires Linux, Intel SGX2, the Intel SGX SDK, CMake, and a C++17
compiler.

```bash
source /opt/intel/sgxsdk/environment
chmod +x exec.sh
./exec.sh -JFYan --profile -t 16
```

## Additional experiments

Prepare the real-data workloads:

```bash
python3 experiments/prepare_workloads.py job13d-scales --download --force
python3 experiments/prepare_workloads.py snap-acyclic --download --force
```

Run the experiments:

```bash
chmod +x experiments/run_experiments.sh
THREADS=16 ./experiments/run_experiments.sh job13d
THREADS=16 ./experiments/run_experiments.sh snap
THREADS=16 ./experiments/run_experiments.sh nonoblivious
THREADS=16 ./experiments/run_experiments.sh memory
```

Generate the additional figures:

```bash
python3 experiments/plots/paper_figures.py
```

See [experiments/README.md](experiments/README.md) for the workload-specific
commands. Generated inputs, logs, and figures are excluded from Git.

The JFYan implementation is under `include/` and `implement/`; SGX host and
enclave code are under `App/` and `Enclave/`.
