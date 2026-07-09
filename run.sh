#!/usr/bin/env bash
set -euo pipefail

TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl lerobot-eval \
       --output_dir=./eval_logs/pi05_scene1_three_tasks_full \
       --env.type=scene1_libero \
       --env.scene_task=pick_red_car,pick_magnifying_glass,pick_gray_box \
       --env.episode_length=280 \
       --env.control_mode=relative \
       --env.max_parallel_tasks=1 \
       --eval.batch_size=1 \
       --eval.n_episodes=10 \
       --policy.path=lerobot/pi05_libero_finetuned \
       --policy.device=cuda:0 \
       --policy.n_action_steps=10 \
       --policy.compile_model=false \
       --policy.gradient_checkpointing=false
