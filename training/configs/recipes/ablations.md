# Ablation recipes (Table 2, bottom)

Start from `stm_main.conf` and override:

| Variant   | Overrides |
|-----------|-----------|
| w/o ID loss | `APPLY_MASK_ON_ORIG_LOSS=False` |
| w/o augmentation | `AUG_CONFIG_PATH=$STM_HOME/training/configs/augmentation/no-aug.yaml` |
| w/o both | both of the above |
| StM (10K / 20K data) | `FILTERING_METHOD=<column>` after marking a random subset `True` in `metadata.csv` |

`bash training/train.sh <conf>` accepts any of the variables listed in `training/train.sh`.
