```bash

python scripts/csv_to_npz.py --input_file source/whole_body_tracking/whole_body_tracking/motions/walk1_subject1.csv \
--input_fps 30 --output_name walk1_subject1 --headless

python scripts/replay_npz.py --registry_name=xu147266-shanghai-jiaotong-university-org/wandb-registry-motions/walk1_subject1

python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-Wo-State-Estimation-v0 --num_envs=2 \
--wandb_path=xu147266-shanghai-jiaotong-university/dance1/1y6uyz7k
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-Wo-State-Estimation-v0 --num_envs=2 \
--wandb_path=xu147266-shanghai-jiaotong-university/walk1_subject1/ntd0lnui



python deploy/deploy_mujoco/mujoco_deploy.py


# 翻墙
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
```