# Slurm templates

The templates intentionally omit institution-specific account numbers,
filesystem paths, and environment-module commands.

Example full verbatim launch:

```bash
mkdir -p logs
sbatch --export=ALL,PROJECT_ROOT="$PWD",MANIFEST=/path/to/dual.jsonl,OUTPUT_DIR=/path/to/full,TRAINING_POLICY=verbatim,OPTIMIZATION=full,VENV_ACTIVATE=/path/to/venv/bin/activate slurm/train_four_gpu.sbatch
```

Example dual-policy LoRA launch: change `TRAINING_POLICY=dual` and
`OPTIMIZATION=lora`.
