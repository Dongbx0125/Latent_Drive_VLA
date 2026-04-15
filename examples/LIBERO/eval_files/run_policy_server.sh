#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
# === Paths (adapted for this cluster) ===
STARVLA_DIR=/code/starVLA
LIBERO_HOME=/code/LIBERO
STARVLA_PYTHON=/opt/conda/envs/starVLA/bin/python
LIBERO_PYTHON=/opt/conda/envs/libero/bin/python

# === Checkpoint ===
CKPT=${STARVLA_DIR}/playground/Checkpoints/Qwen2.5-VL-GR00T-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt

export star_vla_python=${STARVLA_PYTHON}
your_ckpt=${CKPT}   
gpu_id=0
port=6694
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
