# Architecture-Preserving Provable Repair of Deep Neural Networks (APRNN)
APRNN (pronounced "apron") is a library for architecture-preserving provable repair of Deep Neural
Networks. DNN behavior involving either finitely-many or entire polytopes of
points can be repaired using APRNN while preserving the DNN architecture.

The code in this repository is the latest artifact from our paper
***Architecture-Preserving Provable Repair of Deep Neural Networks***, accepted in PLDI 2023.
```
@article{10.1145/3591238,
author = {Tao, Zhe and Nawas, Stephanie and Mitchell, Jacqueline and Thakur, Aditya V.},
title = {Architecture-Preserving Provable Repair of Deep Neural Networks},
year = {2023},
issue_date = {June 2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {7},
number = {PLDI},
url = {https://doi.org/10.1145/3591238},
doi = {10.1145/3591238},
journal = {Proc. ACM Program. Lang.},
month = {jun},
articleno = {124},
numpages = {25},
}
```

## Installation

### Using Docker (Recommended)

#### Build the Docker image

The following command builds the docker image using the Dockerfile in this repo.

```
$ docker build -t aprnn_pldi23:dev .
```

It should take 10-20 mintues to build the image. See [docker
installation](https://docs.docker.com/install/) on how to install `docker`.

#### Run the built Docker image
The following command runs the built docker image (named `aprnn_pldi23:dev`) interactively:
```
$ ./docker_run.sh --memory=384g --cpus=$(nproc)
```
Replace `--memory=384g` with the desired memory limit and `--cpus=$(nproc)` with
the desired CPU limit. See [Docker Runtime options with Memory, CPUs, and
GPUs](https://docs.docker.com/config/containers/resource_constraints/) for
detail.

This command will bind-mount directories `.`, `./data` and `./results` to
`/host_aprnn_pldi23`, `/aprnn_pldi23/data` and `/aprnn_pldi23/results` inside
the container.

(Optional) If you want to use NVIDIA GPU/CUDA, see [Docker official instruction:
access an nvidia
gpu](https://docs.docker.com/config/containers/resource_constraints/#access-an-nvidia-gpu)
on how to setup. The Docker image is compatible with CUDA 11.3 and CUDNN 8.
After CUDA setup, the following command runs the built docker image
interactively with GPUs:
```
$ ./docker_run.sh --memory=384g --cpus=$(nproc) --gpus=all
```

### Local Installation

If you wish to run locally, the reference environment is `Linux` (`Ubuntu
20.04`) with `Python 3.9.7`, `torch 1.11.0` and `torchvision 0.12.0`. Note that
we recommend to use exactly `Python 3.9.7`, as other versions may not be
compatible with `torch 1.11.0`. Run the following command to install required
Python packages.
```
$ pip3 install -r requirements.txt
```

If you wish to use NVIDIA GPU/CUDA, the reference environment uses CUDA 11.3 and
CUDNN 8. You could change the following lines in `requirements.txt` to a CUDA
version that's compatible with your CUDA installation.
```
torch==1.11.0+cu113
torchvision ==0.12.0+cu113
```

## Prerequisites

### Download and Extract Datasets

> You can skip this step in the first pass and come back later. The "Getting
> Started Guide" section does not require this step.

Experiment 2 requires ImageNet-C and ImageNet validation datasets. Please
download the [official ImageNet validation set
(`ILSVRC2012_img_val.tar`)](https://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5)
via torrent and place it to `data/ILSVRC2012/ILSVRC2012_img_val.tar`. The
following command will download ImageNet-A and extract both ImageNet-A and
Imagenet validation datasets.
```
$ make datasets-imagenet
```

### Setup Gurobi License

Reproducing experiments for provable repair of DNNs requires a (free) Gurobi
academic license. Please visit [Gurobi academic
license](https://www.gurobi.com/academia/academic-program-and-licenses) to
generate an "Academic WLS License" (for containers). Aside from the official
instructions, the following steps might be helpful.

- Login to the Gurobi user portal.
- Go to the "License - Request" tab, genearte a "WLS Academic" license if you don't have
  one. If you already have a "WLS Academic" license, you might get an
  "[LICENSES_ACADEMIC_EXISTS] Cannot create academic license as other academic
  licenses already exists" error.
- Go to the "Home" tab, click "Licenses - Open the WLS manager" to open the WLS
  manager.
- In the WLS manager, you should see a license under the "Licenses" tab. Click
  "extend" if it has expired (it might take some time to take effect).
- Go to the "API Keys" tab, click the "CREATE API KEY" button to create a new
  license, download the generated `gurobi.lic` file and place it in
  `/opt/gurobi/gurobi.lic` inside the container.

### Hardware Requirements

All experiments were run on a machine with Dual Intel Xeon Silver 4216 Processor
16-Core 2.1GHz with 384 GB of memory, SSD and RTX-A6000 with 48 GB of GPU memory
running Ubuntu 20.04. Running on a machine with less CPU/GPU cores and memory
might not reproduce the timing numbers in the paper.

Experiment 2 requires 384 GB of memory *and 48 GB of GPU memory* to reproduce,
otherwise the experiment might run out of memory. Also, running Experiment 2
without GPU might be much slower.

Most of the other experiments could be run with less memory (~64GB).

## Getting Started Guide

We will be using `./run.py` to run experiments with the given configuration.
`./run.py --help` lists its options.

For example, the following command runs Experiment 1 with tool APRNN (this work) to repair
the MNIST `3x100` network (see "Experiment 1 (Section 6.1)" for detail) without GPU:

> Note: Gurobi license required, see "Setup Gurobi License" for detail.

```
# ./run.py --eval 1 --tool=aprnn --net=3x100 --device=cpu
```

You could replace `--device=cpu` with `--device=cuda` or `--device='cude:0'` if
you want to use GPU and have CUDA setup. Note that we only tested with RTX A6000
(48GB), hence running larger experiments on GPU with less memory might cause
failure.

If the command succeeds, it prints the result like the following:
> Note: Because `./run.py` by default caches and reuses results from previous
> runs with the same experiment configuration, a second run with the same options
> will be fast. You can discard the cached result and re-run it by appeding the `--rerun`
> option.

```
╔═══════════════════════════════════════════╗
║ Results corresponds to Table 1,                                                     ║
║ for super-columns ('APRNN',) and rows ('3x100',):                                   ║
║                                                                                     ║
║                       APRNN                                                         ║
║             D        G        T                                                     ║
║ 3x100     1.28%    31.53%     5s                                                    ║
║                                                                                     ║
║ Metrics:                                                                            ║
║ - D for drawdown, lower is better.                                                  ║
║ - G for generalization, higher is better.                                           ║
║ - T for time.                                                                       ║
╚═══════════════════════════════════════════╝
```

Note that the timing numbers may not be the same due to the difference in
hardware. The drawdown and generalization numbers may not be exactly the same
for the following reasons:

- The Gurobi solver, especially its concurrent methods, is not deterministic.
  Hence the experiment might produce a different repaired network.
- Difference in hardware (e.g., CPU, GPU, Tensor cores), instruction sets and
  libraries (e.g., CUDA, CUDNN) might cause small differences in the evaluation
  of accuracy.

### Reproduce drawdown and generalization with authors' artifact

> Note: Gurobi license is not required for reproduce drawdown and generalization
> with authors' artifact.

We also provide the artifact (the repaired networks the authors found) used in the
paper. You can evaluate them by appending the `--use_artifact` option to `./run.py`.
For example:
```
# ./run.py --eval 1 --tool=aprnn --net=3x100 --device=cpu --use_artifact
```

If the command succeeds, it prints the result like the following:
```
╔═══════════════════════════════════════════╗
║ Results corresponds to Table 1,                                                     ║
║ for super-columns ('APRNN',) and rows ('3x100',):                                   ║
║                                                                                     ║
║                       APRNN                                                         ║
║             D        G        T                                                     ║
║ 3x100     1.28%    31.53%     N/A                                                   ║
║                                                                                     ║
║                                                                                     ║
║ Metrics:                                                                            ║
║ - D for drawdown, lower is better.                                                  ║
║ - G for generalization, higher is better.                                           ║
║ - T for time.                                                                       ║
╚═══════════════════════════════════════════╝
```

### Reproduce a subset of experiments that takes less time

The following command runs the `9x100` row of Table 1, see "Experiment 1 (Section 6.1)" for detail.

```
./run.py --eval 1 --net=9x200 --device=cpu
```

The above command requires a Gurobi license setup and should take 10 minutes
to run. If it succeeds, it will print the result like:

```
════════════════════════════════════════════╗
║ Results corresponds to Table 1,                                                     ║
║ for super-columns ('PRDNN', 'APRNN') and rows ('9x200',):                           ║
║                                                                                     ║
║                       PRDNN                      APRNN                              ║
║             D        G        T        D        G        T                          ║
║ 9x200     3.92%     6.55%      5s    1.38%    24.72%    455s                        ║
║                                                                                     ║
║                                                                                     ║
║ Metrics:                                                                            ║
║ - D for drawdown, lower is better.                                                  ║
║ - G for generalization, higher is better.                                           ║
║ - T for time.                                                                       ║
╚═══════════════════════════════════════════╝
```

## Step-by-Step Instructions

In this section we provide guide to reproduce all experiments. At a high-level,
we list the approximate runtime on our machine and special requirements.

- Experiment 1 (except REASSURE) should take 20-30 minutes to run. Running REASSURE might take hours.
- Experiment 2 should take 2-4 hours to run using NVIDIA RTX A6000. It requires
  384GB of memory and 48GB of GPU memory, otherwise it might run out of memory.
  It requires the ImagenNet validation datasets (see "Download and Extract
  Datasets" above.)
- Experiment 3 (except REASSURE) should take 80-100 mnitues to run. Running REASSURE might take hours/days.
- Experiment 4 should take 80-100 minutes to run.
- Experiment 5 should take 10-20 mintues to run.
- Experiment 6 should take 20-30 minutes to run.
- Experiment 7 should take 1-2 days to run (depends on the configurations).
- Experiment 8 should take 1-2 hours to run.

For each experiment, will provide commands to run a subset of experiments. In
addition, we also provide the artifact (the repaired networks the authors found)
used in the paper. Evaluating them does not involve the time-consuming repair
process, hence should take much less time.

### Experiment 1 (Section 6.1)

> The REASSURE support is not integrated into `run.py` yet. For running
> REASSURE, please use `eval_1_reassure.py`. 

The following command reproduces Table 1 using this
work (APRNN) and the baseline (PRDNN, Lookup) to repair the all three MNIST networks
(`3x100`, `9x100`, `9x200`):

```
./run.py --eval 1 --device=cpu
```

The above command requires a Gurobi license setup and should take 20-30 minutes
to run.

#### Reproduce drawdown and generalization with authors' artifact

We also provide the artifact (the repaired networks the authors found) used in the
paper. You can evaluate them by appending the `--use_artifact` option to `./run.py`:
```
./run.py --eval 1 --device=cpu --use_artifact
```

The above command *does not* requires a Gurobi license setup and should take 5 minutes
to run. 

#### Reproduce a subset of experiments

By appending the `--tool=aprnn`, `--tool=prdnn` or (the default) `--tool=all`
option, you can reproduce results for only the specified super-columns. By
appending the `--net=3x100`, `--net=9x100`, `--net=9x200` or (the default)
`--net=all`, you can reproduce results for only the specified rows.

For example, the following command only reproduces the `APRNN` super-column and
the `9x100` row:
```
./run.py --eval 1 --tool=aprnn --net=3x100 --device=cpu
```

The above command requires a Gurobi license setup and should take 1 minute
to run.

### Experiment 2 (Section 6.2)

> Note: experiment 2 requires 384GB of memory and 48GB of GPU memory, otherwise
> it might run out of memory. It might be very slow without using GPU. It
> requires the ImagenNet validation datasets (see "Download and Extract
> Datasets" above).

> The ImageNet-C experiment is not integrated into `run.py` yet. Please use
> `eval_2c_aprnn.py`.

The following command reproduces Section 6.2 using
this work (APRNN) and the baseline (PRDNN) to repair the two ImageNet networks
(`resnet152` amd `vgg19`).

> Replace `--device=cpu` with `--device=cuda` if GPU/CUDA is applicable. See
> "Installation" for detail.

```
./run.py --eval 2 --device=cpu
```

The above command requires a Gurobi license setup and should take 2-3 hours to
run using NVIDIA RTX A6000. If it succeeds, it will print the result like:

> Note that PRDNN ran out of memory and failed to repair the both networks in
> our experiment, hence the `(failed)` entries.

```
╔═══════════════════════════════════════╗
║ Results corresponds to Section 6.2,                                         ║
║ For specified tools ('aprnn', 'prdnn') and networks ('resnet152', 'vgg19'): ║
║                                                                             ║
║          resnet152                       vgg19                              ║
║           D@top-1   D@top-5      T      D@top-1   D@top-5      T            ║
║ APRNN        3.15%     1.62%    2789s        1.86%     0.96%  2725s         ║
║ PRDNN     (failed)  (failed)  (failed)  (failed)  (failed)  (failed)        ║
║                                                                             ║
║ Metrics:                                                                    ║
║ - D@top-1 for top-1 accuracy drawdown, lower is better.                     ║
║ - D@top-5 for top-5 accuracy drawdown, lower is better.                     ║
║ - T for time, lower is better.                                              ║
╚═══════════════════════════════════════╝
```

#### Reproduce drawdown and generalization with authors' artifact

We also provide the artifact (the repaired networks the authors found) used in the
paper. You can evaluate them by appending the `--use_artifact` option to `./run.py`:

```
./run.py --eval 2 --tool=aprnn --device=cpu --use_artifact
```

The above command *does not* requires a Gurobi license setup and should take
less than 20 minutes to run using NVIDIA RTX A6000. However, it might take *much
longer time* on CPU. If it succeeds, it will print the result like:

```
╔═══════════════════════════════════════╗
║ Results corresponds to Section 6.2,                                         ║
║ For specified tools ('aprnn', 'prdnn') and networks ('resnet152', 'vgg19'): ║
║                                                                             ║
║          resnet152                       vgg19                              ║
║           D@top-1   D@top-5      T      D@top-1   D@top-5      T            ║
║ APRNN        3.15%     1.62%       N/A     1.86%     0.96%       N/A        ║
║                                                                             ║
║ Metrics:                                                                    ║
║ - D@top-1 for top-1 accuracy drawdown, lower is better.                     ║
║ - D@top-5 for top-5 accuracy drawdown, lower is better.                     ║
║ - T for time, lower is better.                                              ║
╚═══════════════════════════════════════╝
```

#### Reproduce a subset of experiments

By appending the `--tool=aprnn`, `--tool=prdnn` or (the default) `--tool=all`
option, you can reproduce results for only the specified tool.

For example, the following command only reproduces the `APRNN` results for `vgg19`:
```
./run.py --eval 2 --tool=aprnn --net=vgg19 --device=cpu
```

The above command requires a Gurobi license setup and should take 1-2 hours to
run using NVIDIA RTX A6000. If it succeeds, it will print the result like:

```
╔════════════════════════════════╗
║ Results corresponds to Section 6.2,                            ║
║ For specified tools ('aprnn',) and networks ('vgg19',):        ║
║                                                                ║
║           vgg19                                                ║
║          D@top-1  D@top-5     T                                ║
║ APRNN     1.86%    0.96%     2725s                             ║
║                                                                ║
║ Metrics:                                                       ║
║ - D@top-1 for top-1 accuracy drawdown, lower is better.        ║
║ - D@top-5 for top-5 accuracy drawdown, lower is better.        ║
║ - T for time, lower is better.                                 ║
╚════════════════════════════════╝
```

### Experiment 3 (Section 6.3)

> The REASSURE support is not integrated into `run.py` yet. For running
> REASSURE, please use `eval_3_reassure.py`. 

The following command reproduces Section 6.3 using this
work (APRNN) and the baseline (PRDNN and Lookup) to repair the one MNIST network (`9x100`).

```
./run.py --eval 3 --device=cpu
```

The above command requires a Gurobi license setup and should take 80-100 minutes to
run.

#### Reproduce using authors' artifact

We also provide the artifact (the repaired networks the authors found) used in the
paper. You can evaluate them by appending the `--use_artifact` option to `./run.py`:

```
./run.py --eval 3 --device=cpu --use_artifact
```

The above command *does not* requires a Gurobi license setup and should take less than 5 minutes
to run.


#### Reproduce a subset of experiments

By appending the `--tool=aprnn`, `--tool=prdnn` or (the default) `--tool=all`
option, you can reproduce results for only the specified tool.

For example, the following command only reproduces the `APRNN` results:
```
./run.py --eval 3 --device=cpu --tool=aprnn
```

The above command requires a Gurobi license setup and should take 3-5 minutes
to run.

### Experiment 4 (Section 6.4)

The following command reproduces Section 6.4 using this
work (APRNN) and the baseline (PRDNN) to repair the ACAS Xu network (`n29`).

```
./run.py --eval 4 --device=cpu
```

The above command requires a gurobi license setup and should take 80-100 minutes to
run.

#### Reproduce using authors' artifact

We also provide the artifact (the repaired networks the authors found) used in the
paper. You can evaluate them by appending the `--use_artifact` option to `./run.py`:

```
./run.py --eval 4 --device=cpu --use_artifact
```

The above command *does not* requires a Gurobi license setup and should take 3-5 minutes
to run.

#### Reproduce a subset of experiments

By appending the `--tool=aprnn`, `--tool=prdnn` or (the default) `--tool=all`
option, you can reproduce results for only the specified tool.

For example, the following command only reproduces the `APRNN` results:
```
./run.py --eval 4 --device=cpu --tool=aprnn
```

The above command requires a Gurobi license setup and should take 3-5 minutes
to run.

### Experiment 5 (Section 6.5)

The following command reproduces Section 6.5 using this
work (APRNN) to repair the ACAS Xu network (`n29`).

```
./run.py --eval 5 --device=cpu
```

The above command requires a Gurobi license setup and should take 10-20 minutes to
run.

### Experiment 6 (Section 6.6)

The following command reproduces Section 6.6:
```
./run.py --eval 6 --net all --npoints all --device=cpu
```

### Experiment 7 (Section 6.7)

Please refer the script `./eval_7.sh` to run with specified configurations.


### Experiment 8 (Section 6.8)

Please refer the script `./eval_8_lookup.sh` to reproduce the polytope repair time of Lookup-based approach.  


# Troubleshooting and Frequently Asked Questions

## Why do I see `bash: ./run.py: Permission denied`?

It is likely becuase `zip` does not preserve permissions of files. Please run
the following command to grant the execution permission to `./run.py`. Sorry for
the inconvenience.

```
chmod +x ./run.py
```

## Why do I see `gurobipy.GurobiError: Model too large for size-limited license`?

This is because the Gurobi academic license is missing and Gurobi is using a
trial license shipped with the `gurobipy` package. Please follow the "Setup
Gurobi License" section to acquire one and put it (or paste its content to)
under `/opt/gurobi/gurobi.lic`. To verify the license, the command

```
cat /opt/gurobi/gurobi.lic
```

should print a license like

```
# Gurobi WLS license file
# Your credentials are private and should not be shared or copied to public repositories.
# Visit https://license.gurobi.com/manager/doc/overview for more information.
WLSACCESSID=<WLSACCESSID>
WLSSECRET=<WLSSECRET>
LICENSEID=<LICENSEID>
```

And you should be able to see the following lines in the console output of experiments.

```
Set parameter WLSAccessID
Set parameter WLSSecret
Set parameter LicenseID to value <LICENSEID>
Academic license - for non-commercial use only - registered to <username or email>
```

Also, after running any experiment with your license, you should be able to
login https://license.gurobi.com/manager/keys and see activities of the
corresponding license.

## Why do I see "Gurobi license expired"?

There are few possible reasons:

1. It might because you haven't put your academic license
   `/opt/gurobi/gurobi.lic` and the trial license has expired. In this case,
   please follow the "Setup Gurobi License" section to install your license.

2. It might because your Gurobi WLS license is expired. You could login
   https://license.gurobi.com/manager/licenses and check the status of your
   Gurobi WLS license. If it is expired, check `extend` to extend it.

3. It might because the Gurobi server haven't update the expiration date of your
   license if you just registered one or extended it. In this case, please wait
   for a few minutes.

## Why do I see an `ModuleNotFoundError: No module named ...` exception?

It is because the python virtual environment (venv) is deactivate in the shell.
Our docker image installs a python 3.9.7 venv
(located in `/aprnn_pldi23/external/python_venv/3.9.7`)
with all dependencies and activates it in `cat /root/.bashrc`.
If it is deactivated by mistake, you should see the command prompt like

```
root@9bb4670e674d:/aprnn_pldi23#
```

instead of

```
(3.9.7) root@9bb4670e674d:/aprnn_pldi23#
```

To activate the venv, please run the following command:

```
source /aprnn_pldi23/external/python_venv/3.9.7/bin/activate
```

You could verify it by checking if `which python3` prints `/aprnn_pldi23/external/python_venv/3.9.7/bin/python3`.

If activating the venv does not resolve the issue, it's likely because the
installed venv is corrupted. Please exit the current docker container and run
`./docker_run.sh` to create a new container.

## Why do I see `FileNotFoundError: [Errno 2] No such file or directory: '/aprnn_pldi23/data/...'`?

Please follow the "Download and Extract Datasets" section and download the
imagenet datasets to run experiment 2.

## Why do I see `(torun)` in the result?

`(torun)` indicates the corresponding experiment to produce this result hasn't
been run yet, or was interrupted, or failed. Please run the command again to
produce the missing entries.

## Why do I see `N/A` in the result?

`N/A` only appears in the repair time (T) entries when running with option
`--use_artifact`. This is expected because the option `--use_artifact` is
intended to evaluate the drawdown and generalization metrics on authors'
artifact (repaired DNNs) post-repair. Thus, running with `--use_artifact` does
not involve the repair process and can not measure the repair time (T).

## Why the commands to run experiments finish faster at the second time?

Our scripts cache (partial) results and reuse them in later runs by default to
save time in case a sub-task failed or was interrupted. You could append the
`--rerun` option to discard the cached results and force a rerun.

## How could I view the cached (partial) results without actually running anything new experiments?

You could append the `--norun` option to view the cached (partial) results.
Missing results will be displayed as `(torun)`.

## Why the experiment is not using all the memory or CPU cores?

Please follow the "Using Docker" section and run `./docker_run.sh` with options
specifying the hardware resources. For example, the following command allows
docker to use up to 384GB memory, all CPU cores and all GPUs.

```
./docker_run.sh --memory=384g --cpus=$(nproc) --gpus=all
````
