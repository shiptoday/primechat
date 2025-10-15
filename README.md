# primechat
fork of karpathy's nanochat. using it to learn, experiment and try some ideas. 
- i saw karpathy's nanochat and i knew i had to play with it. 
- i've never implemented an llm myself before, i've never used gpu clouds, so i had a lot to learn. drank coffee and got ready to dive in for a lot of hours.
- i bought $100 of credits and asked chatgpt on details on the best seetup. 


my setup:
- runpod.io for gpu cloud
- pod details:
    8x H200 SXM (1128 GB VRAM)
    2008 GB RAM • 192 vCPU
    Total Disk: 80 GB
    $28.72/hr
    pod template: pytorch 2.8.0 (runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404)
- macos as my local machine, using cursor to ssh into runpod
- the first thing i did was to make a test run of the model, training a model with depth of 4, it took like 45min and i got familiar with the process. after understanding a couple things, i started to brainstorm a few ideas on what to tweak: pretraining dataset, hyperparameters, improve the ui, etc. 


# Replicate (step by stepy commands)

git clone https://github.com/shiptoday/primechat.git
cd nanochat
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
source .venv/bin/activate
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
python -m nanochat.dataset -n 240
python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval
curl -L -o eval_bundle.zip https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
unzip -q eval_bundle.zip
rm eval_bundle.zip
mv eval_bundle "$HOME/.cache/nanochat"
wandb login
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=20 \
  --device_batch_size=64 \
  --total_batch_size=1048576 \
  --run=primechat_v1


# Things to play around with
- Modify the mid-training dataset (Smol smaltalk)
- Improve the ui: 
  - model picker
  - stats for nerds: t/s, ttft, total tokens 
  - copy message
- Hyperparameters?
  - depth: 20 default?
  - training horizon
  - Device batch
  - total batch size
  - Vocab size?

increase total batch size to 1,048,576

increasee device batch size to 56





![nanochat logo](dev/nanochat.png)


```bash
bash speedrun.sh
```

Alternatively, since the script runs for 4 hours, I like to launch it like this inside a new screen session `speedrun` (and also log output to `speedrun.log`):

```bash
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

## License

MIT
