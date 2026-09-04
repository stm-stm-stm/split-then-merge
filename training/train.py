#!/usr/bin/env python
"""Train the StM Composer.  Usually launched through ``training/train.sh <config>``
(accelerate + FSDP), but can be called directly:

    python -m accelerate.commands.launch --config_file training/configs/accelerate/fsdp2_1node_4gpu.yaml \
        training/train.py --model_path THUDM/CogVideoX-5b-I2V --model_name cogvideox-i2v ...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stm.models import get_model_cls  # noqa: E402
from stm.schemas import Args  # noqa: E402


def main():
    args = Args.parse_args()
    trainer_cls = get_model_cls(args.model_name, args.training_type)
    trainer = trainer_cls(args)
    trainer.fit()


if __name__ == "__main__":
    main()
