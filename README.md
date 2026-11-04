# EpiSelect

[![arXiv](https://img.shields.io/badge/arXiv-2608.11746-b31b1b.svg)](https://arxiv.org/abs/2608.11746)

Code for [**Epiplexity Guided Data Selection and Generation for Out-of-Distribution
Generalization**](https://arxiv.org/abs/2608.11746) — epiplexity-maximizing data selection for
curriculum learning over mixed-domain pretraining corpora. EpiSelect fits a cross-domain scaling law
`L_m = beta_m (sum_k gamma_mk n_k)^-alpha_m + eps_m` and reweights batches toward the domains with the
largest marginal contribution to epiplexity, refitting every 1,000 steps. No proxy model required.
Adapted from the [ADO reference implementation](https://github.com/yidingjiang/ado).

## Setup

```bash
bash requirements.sh          # or: bash requirements.sh tpu
source .venv/bin/activate
export WANDB_API_KEY=...      # or write the key to .wandbkey
```

## Data

```bash
# all 31 domains as a SLURM array (one job per domain), synced to the bucket as each finishes
sbatch scripts/build_commonpile_tfds.slurm
# or by hand, e.g. a subset
python scripts/build_commonpile_tfds.py --domains github,pubmed --data_dir /scratch/$USER/commonpile_tfds_v3 --sync gs://commonpile-tfds/commonpile_tfds
# exact per-split token counts (token_counts.json next to each domain's shards; the loader requires them)
sbatch scripts/count_commonpile_tokens.slurm
# OOD eval token streams (c4, slim_pajama, fine_web)
sbatch scripts/build_ood_cache.slurm
```

Streams the 31 `common-pile/*_filtered` domains at pinned HF revisions, shuffles the stream with a
fixed seed and tokenizes with GPT-NeoX-20B until each domain reaches 4B tokens
(`src/tfds/common_pile_grouped.py`, builder v3.0.0): 78.0B tokens in total, 16 domains at the cap,
~154 GB of shards. Val is `train[:1%]`. Building a domain that is
already prepared is a no-op, so re-submitting the array only fills in what is missing. Token counts are
read from the shards, not from TFDS `num_bytes` (varint-serialized, ~2 bytes/token, which undercounts 2x). Set `HF_HOME`
to a shared cache; lm-eval task data is downloaded into it on first use.

## Train

Six experiments: `{124M, 1.3B} x {Natural, ADO, EpiSelect}` under `configs/experiment/`, plus two 124M ablations:
`ADO_noprior` (ADO without the empirical prior) and `EpiSelect_diag` (EpiSelect with gamma fixed to the identity,
i.e. a per-domain scaling law). The queue runs the ablations after the main methods of a size (`ABLATIONS_124M`).

Every run is identified by experiment and seed: checkpoints go to `<CKPT_BASE>/<size>/<method>/seed<N>`
and W&B (project `episelect-iclr26`) groups runs by `<size>-<method>` with one run per seed. Relaunching
the same experiment and seed resumes from the latest checkpoint.

```bash
# single host
python launch.py +experiment=124M/EpiSelect seed=0 \
  loader.data_dir=/path/to/commonpile_tfds \
  rundir=/path/to/checkpoints/124M/EpiSelect/seed0 \
  multihost=false

# TPU pod (spot v6e in us-east1-d). Needs .wandbkey in the repo root; setup ships the local tree to every worker
./scripts/tpu_setup.sh node1 us-east1-d
./scripts/tpu_launch.sh 124M/EpiSelect 0 gs://commonpile-tfds/commonpile_tfds gs://commonpile-tfds/checkpoints
# the whole sweep, one run at a time, skipping runs that already have a DONE marker
./scripts/tpu_queue.sh gs://commonpile-tfds/commonpile_tfds gs://commonpile-tfds/checkpoints node1 us-east1-d
# unattended on spot: request the node, set it up, run the queue, re-request after preemption, until done
./scripts/tpu_watchdog.sh node1 us-east1-d gs://commonpile-tfds/commonpile_tfds gs://commonpile-tfds/checkpoints v6e-32
```

Every run ends with a checkpoint at the last step and a `DONE` file; a run that stops early (spot
preemption) resumes from its latest checkpoint and saved loader state when relaunched.

## Evaluate

`configs/eval.yaml` lists the three methods under `ckpt_base/<size>/<method>` and evaluates every seed
on the Common Pile val split, three OOD token streams (C4, SlimPajama, FineWeb) and 13 LM Eval Harness
tasks. The paper reports 124M seeds 2, 3, 4 and 1.3B seeds 0, 1, 2; Table 1 averages `paper10`
(ARC = mean of arc_challenge and arc_easy, BBQ, CSQA, HellaSwag, LAMBADA, LogiQA2, OBQA, PIQA, SciQ, WinoGrande).

```bash
python -m src.eval size=124M seeds='[2,3,4]' output=eval_results/124M.json   # one entry per label+seed, saved after every task
python -m src.eval size=1.3B seeds='[0,1,2]' output=eval_results/1.3B.json
python scripts/aggregate_eval.py eval_results/1.3B.json --out eval_results/1.3B_agg.json   # mean ± std tables and group averages
```

## Citation

```bibtex
@article{su2026epiplexity,
  title   = {Epiplexity Guided Data Selection and Generation for Out-of-Distribution Generalization},
  author  = {Su, Ellen and Potapczynski, Andres and Qiu, Shikai and Hughes, Edward and Wilson, Andrew Gordon},
  journal = {arXiv preprint arXiv:2608.11746},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.11746},
}
```
