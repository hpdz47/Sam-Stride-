# Runbook — run the host-guest screening MAS on the GPU cluster

This runs the **screening decision phase + interpreter** end-to-end with the
real vLLM/Qwen backend (the first live test of the M2/M3 work). It reports the
deterministic verdicts and the Qwen interpreter's narration. It does **not** run
the full plan/code/report pipeline.

## 0. Prerequisite — rebuild the container (ONE TIME, important)

The spectral tools need `numpy scipy nmrglue dtw-python`. These were added to the
`Dockerfile`, but the cluster pulls the prebuilt image
`docker://sts2102/meng_project:latest`, so the image must be **rebuilt and
pushed** before the deps are present. From a machine with Docker + your DockerHub
login:

```bash
docker build -t sts2102/meng_project:latest .
docker push sts2102/meng_project:latest
# then on the cluster, force a fresh pull:
rm -f /nobackup/$USER/containers/bioagent.sif
```

Quick check the deps are in the image (after pull):
```bash
singularity exec /nobackup/$USER/containers/bioagent.sif \
  python3 -c "import numpy,scipy,nmrglue,dtw; print('spectral deps OK')"
```
> Fallback without rebuild: `pip install --target=$PROJECT/pydeps numpy scipy
> nmrglue dtw-python`, then add `--bind $PROJECT/pydeps:/pydeps` and
> `--env PYTHONPATH=/pydeps` to the singularity call. Rebuilding is cleaner.

## 1. Stage the data

Copy the ground-truth bundle so that `/inputs/DATA/...` and `/inputs/RAW-NMR/...`
resolve inside the container — i.e. put the CONTENTS of the cooper-group `data/`
folder into `$PROJECT/Inputs`:

```bash
PROJECT=/nobackup/$USER/projects/BioAgent
mkdir -p $PROJECT/Inputs
cp -r /path/to/cooper-group-uol-robotics/data/* $PROJECT/Inputs/
# sanity:
ls $PROJECT/Inputs/DATA/INPUT/SUPRAMOL-SCREENING-SM-NMR.json
ls $PROJECT/Inputs/RAW-NMR/NMR/SUPRAMOL-SCREENING/DATA/NMR | head
```

## 2. Configure `.env`

`$PROJECT/.env` must define at least:
```
API_KEY=<any-string-vllm-uses-it-as-a-token>
LLM_Model_Reasoning=<hf-id-or-local-path-of-your-Qwen-reasoning-model>
DOCKER_USERNAME=<dockerhub-user>      # only needed for the pull
DOCKER_TOKEN=<dockerhub-token>
```
Only the Reasoning model is used for screening. (8xH200: one H200 is plenty; to
shard a larger model add `--tensor-parallel-size N` to the Reasoning branch in
`Config/vLLM_Manager.py`.)

## 3. Submit

```bash
cd $PROJECT
sbatch batch_job_screening.sh
```

Optional workflow selection via env (screening is the default):
```bash
SCREEN_WORKFLOW=SUPRAMOL-HOST-GUEST sbatch batch_job_screening.sh
# choices: SUPRAMOL-SCREENING (default) | SUPRAMOL-REPLICATION | SUPRAMOL-HOST-GUEST
# SCREEN_USE_PROCESSED=0 to exercise the portable raw-FID path instead of 1r
```

## 4. Results

- stdout (SLURM log): verdict table + interpreter narration.
- `$PROJECT/Outputs/Results/SUPRAMOL-SCREENING-SCREENING-VERDICTS.json`
- `$PROJECT/Outputs/Results/SUPRAMOL-SCREENING-SCREENING-INTERPRETATION.txt`
- vLLM server log: `$PROJECT/Outputs/logs/vllm_server_reasoning.log`

## What to expect

The deterministic verdicts must match the validated CPU results (18/18 for
SUPRAMOL-SCREENING). The interpreter narration is where the Qwen model is
actually exercised — check that it explains the plate outcome and lists the
passing samples (positions 1 and 7 pass both NMR and MS) **without changing any
verdict**. If the narration contradicts a verdict, that is a prompt-tuning
finding, not a tool error (the booleans are ground-truth-stable).

## Troubleshooting

- `ModuleNotFoundError: nmrglue/scipy/dtw` → step 0 not done (container lacks deps).
- `FileNotFoundError .../SUPRAMOL-SCREENING-SM-NMR.json` → data not staged at step 1.
- vLLM never becomes healthy → check `logs/vllm_server_reasoning.log`; usually a
  model id/path or GPU-memory issue (lower `--gpu-memory-utilization`).
