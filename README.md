# JFYan

Code and experimental materials for *Parallel Oblivious Acyclic Joins for
TEE-based Encrypted Databases*.

This repository accompanies the original submission. The full version below
includes the additional experimental results, with new material shown in blue.

- [Full version with additional results](Parallel_Oblivious_Acyclic_Joins_for_TEE_based_Encrypted_Databases__Full_Version_1.pdf)
- [Original submission repository](https://github.com/senntuse-commits/Parallel_Oblivious_Acyclic_Joins_for_TEE-based_Encrypted_Databases)

## Main additions

- NonObliJFYan for measuring obliviousness overhead
- Email-EuAll and JOB 13d workloads
- Memory and constrained-EPC experiments
- Data and scripts for the additional figures

## Quick start

The project requires Linux, Intel SGX2, and the Intel SGX SDK.

```bash
source /opt/intel/sgxsdk/environment
./exec.sh -JFYan --profile -t 16
```

See [experiments/README.md](experiments/README.md) for experiment commands and
figure generation.
