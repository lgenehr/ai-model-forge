# GigaVerbo-v2 Integration for V2

This integration is isolated to `hybrid-moe-1b_v2/`. V1 code and behavior are untouched.

## Decision Summary

- Chosen PT recipe: `GigaVerbo-v2 educational subset + GigaVerbo-v2 Synth`.
- Training ingestion: local materialized memmaps only in the hot path.
- Streaming role: optional offline acquisition/warm-cache step in `prepare_gigaverbo_v2.py`.
- Curriculum: three V2-only stages defined in `data_config_v2.yaml`.

## Why This Mix

The Tucano 2 paper reports:

- `GigaVerbo-v2` default is the main corpus, but only about 37% of it is educational.
- `GigaVerbo-v2 Synth` complements missing domains.
- The `Edu+Synth` ablation is the best aggregate mixture, beating `Non-Edu`.

For this repo, that translates to:

- Favor the educational subset instead of the raw default subset in the main PT path.
- Add `Synth` later in the curriculum rather than from step zero.
- Keep a smaller amount of the existing PT corpus for breadth instead of replacing it outright.

## Prepare Data

Educational subset:

```bash
~/.pyenv/versions/venv-hybrid-mamba-bitnet/bin/python prepare_gigaverbo_v2.py prepare \
  --subset edu \
  --raw_dir /home/lgene/meu_modelo_temp/ai-model-forge/datasets/gigaverbo_v2/raw/gigaverbo_v2_edu \
  --output_dir /home/lgene/meu_modelo_temp/ai-model-forge/datasets/gigaverbo_v2/tokenized/gigaverbo_v2_edu \
  --append_eot
```

Synthetic subset:

```bash
~/.pyenv/versions/venv-hybrid-mamba-bitnet/bin/python prepare_gigaverbo_v2.py prepare \
  --subset synth \
  --raw_dir /home/lgene/meu_modelo_temp/ai-model-forge/datasets/gigaverbo_v2/raw/gigaverbo_v2_synth \
  --output_dir /home/lgene/meu_modelo_temp/ai-model-forge/datasets/gigaverbo_v2/tokenized/gigaverbo_v2_synth \
  --append_eot
```

If you already have local JSONL exports, replace HF acquisition with:

```bash
~/.pyenv/versions/venv-hybrid-mamba-bitnet/bin/python prepare_gigaverbo_v2.py prepare \
  --subset edu \
  --local_jsonl_glob "/path/to/gigaverbo/*.jsonl" \
  --raw_dir /home/lgene/meu_modelo_temp/ai-model-forge/datasets/gigaverbo_v2/raw/gigaverbo_v2_edu \
  --output_dir /home/lgene/meu_modelo_temp/ai-model-forge/datasets/gigaverbo_v2/tokenized/gigaverbo_v2_edu \
  --append_eot
```

## Train V2

```bash
~/.pyenv/versions/venv-hybrid-mamba-bitnet/bin/python train_v2.py \
  --data_config_v2 ./data_config_v2.yaml \
  --lr_schedule wsd
```

## Notes

- `train_v2.py` rebuilds the train loader automatically at stage boundaries.
- Validation uses the stable V2 validation mix from `data_config_v2.yaml`.
- Missing V2 artifacts fail fast instead of silently falling back to online tokenization.
