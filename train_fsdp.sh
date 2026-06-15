#!/bin/bash
accelerate launch \
    --config_file examples/accelerate/fsdp_config_offload.yaml \
    src/train.py train_dpo.yaml
