# 🌐 Hivemind — Decentralized Heterogeneous Multi-Node Training

> **Pool laptops, gaming PCs, and cloud GPUs into a single async training swarm.**
> More nodes = more throughput. Heterogeneous hardware is a feature, not a bug.

## Why Hivemind?

Standard DDP (Distributed DataParallel) requires:
- Homogeneous GPUs (same model, same memory)
- Tight synchronization (all ranks wait for the slowest)
- Low-latency interconnects (NVLink / InfiniBand)
- All nodes available simultaneously

**Hivemind removes every one of these constraints.** Each peer trains at its own speed and asynchronously averages parameters with a random subset of the swarm. A 4090 doing 3 steps while a laptop GPU does 1 is **expected and correct** — faster peers contribute proportionally more.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Peer A        │    │   Peer B        │    │   Peer C        │
│   RTX 4090      │    │   RTX 3060      │    │   MacBook M3    │
│   8 GB batch    │    │   4 GB batch    │    │   2 GB batch    │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Hivemind DHT        │
                    │  (decentralized)      │
                    │  Async parameter      │
                    │  averaging            │
                    └───────────────────────┘
```

Each peer:
1. Has the **full model** (same architecture via `ModelConfig.from_target_size()`)
2. Reads its **own data shard**
3. Runs local forward/backward/step at its own pace
4. Fires an **async all-reduce** after each optimizer step
5. Continues training immediately (doesn't block on the all-reduce)
6. Absorbs averaged parameters when the all-reduce completes

## Quick Start

### 1. Install

```bash
pip install -r hivemind/requirements-hivemind.txt
```

### 2. Prepare data (same as the base framework)

> **Run this once on any single machine.**  The packed `.bin` files are fully
> portable — copy them to other peers or put them on a shared network drive.
> You do NOT need to run data preparation on every machine.

```bash
# Tokenizer
python tokenizer_train.py --data-dir ./data --output-dir ./tokenizer

# Pack pretrain data
python data/pack_pretrain.py --data-dir ./data --tokenizer ./tokenizer --cache-dir ./packed
```

### 3. Start the swarm

**Machine A** (bootstrap node — a beefy desktop or cloud instance):

```bash
bash hivemind/run.sh bootstrap --model-size 300M --data-dir ./packed \
  --batch-size 8 --grad-accum 4 --checkpoint-dir ./hivemind_ckpts
```

**Machine B** (another GPU machine):

```bash
bash hivemind/run.sh worker 192.168.1.100:5678 --model-size 300M \
  --data-dir ./packed --batch-size 4 --checkpoint-dir ./hivemind_ckpts_b
```

**Machine C** (laptop with limited GPU):

```bash
bash hivemind/run.sh worker 192.168.1.100:5678 --model-size 300M \
  --data-dir ./packed --batch-size 2 --dtype fp32 \
  --checkpoint-dir ./hivemind_ckpts_c
```

Each peer saves its own checkpoints. To produce a merged evaluation checkpoint:

```bash
bash hivemind/run.sh average 192.168.1.100:5678 ./averaged_model
```

## Files

| File | Purpose |
|------|---------|
| `hivemind_utils.py` | Shared utilities: peer setup, `build_hivemind_optimizer()`, checkpoint helpers |
| `train_pretrain_hivemind.py` | Decentralized **pretraining** (main training entry point) |
| `train_sft_hivemind.py` | **Supervised fine-tuning** with Hivemind (supports LoRA/DoRA) |
| `train_grpo_hivemind.py` | **GRPO RL post-training** with Hivemind (policy-only averaging) |
| `train_dpo_hivemind.py` | **DPO preference optimization** with Hivemind (policy-only averaging) |
| `run.sh` | Convenience launcher for all training modes |
| `requirements-hivemind.txt` | Additional dependencies |

## Training Pipeline

The full post-training pipeline matches the base framework:

```
Pretrain  ──→  SFT  ──→  GRPO or DPO
(train_pretrain   (train_sft    (train_grpo_hivemind.py
 _hivemind.py)     _hivemind.py)  or train_dpo_hivemind.py)
```

### Pretrain

```bash
# Bootstrap
bash hivemind/run.sh bootstrap --model-size 300M --data-dir ./packed \
  --batch-size 8 --grad-accum 4 --checkpoint-dir ./hivemind_ckpts

# Worker
bash hivemind/run.sh worker 192.168.1.100:5678 --model-size 300M \
  --data-dir ./packed --batch-size 4 --checkpoint-dir ./hivemind_ckpts_b
```

### SFT

```bash
# Bootstrap
bash hivemind/run.sh sft-bootstrap --model-size 300M \
  --data-dir ./sft_packed --lora-rank 64 --checkpoint-dir ./sft_ckpts

# Worker
bash hivemind/run.sh sft-worker 192.168.1.100:5678 --model-size 300M \
  --data-dir ./sft_packed --lora-rank 64 --checkpoint-dir ./sft_ckpts_b
```

### GRPO (Reinforcement Learning)

```bash
# Bootstrap
bash hivemind/run.sh grpo-bootstrap \
  --checkpoint ./sft_ckpts/latest.pt \
  --data-dir ./grpo_packed --tokenizer ./tokenizer \
  --out-dir ./grpo_ckpts --lora-rank 64

# Worker
bash hivemind/run.sh grpo-worker 192.168.1.100:5678 \
  --checkpoint ./sft_ckpts/latest.pt \
  --data-dir ./grpo_packed --tokenizer ./tokenizer \
  --out-dir ./grpo_ckpts_b --batch-size 2
```

### DPO (Preference Optimization)

```bash
# Bootstrap
bash hivemind/run.sh dpo-bootstrap \
  --checkpoint ./sft_ckpts/latest.pt \
  --data-dir ./dpo_packed --tokenizer ./tokenizer \
  --out-dir ./dpo_ckpts --lora-rank 64

# Worker
bash hivemind/run.sh dpo-worker 192.168.1.100:5678 \
  --checkpoint ./sft_ckpts/latest.pt \
  --data-dir ./dpo_packed --tokenizer ./tokenizer \
  --out-dir ./dpo_ckpts_b --batch-size 2
```

## Real-World Heterogeneous Setup

Below are concrete examples for three very different machines pooling their
compute on the same training run.  Adjust IPs, ports, and paths to match your
network.

> **Important**: All peers must share access to the same packed data directory
> (NFS / network drive or a copy on each machine).  Each peer has its **own**
> checkpoint directory so it can save and resume independently.

### Machine A — Desktop (32 GB RAM, RTX 4090)

This machine is the fastest — make it the **bootstrap** peer and use the most
aggressive batch sizes.  Its Gradio 4090 can run BF16 and handle large batches,
so it drives most of the training progress.

```bash
# ── Pretrain ────────────────────────────────────────────────────────
PORT=5678 bash hivemind/run.sh bootstrap --model-size 300M \
  --data-dir /mnt/nfs/packed \
  --batch-size 16 --grad-accum 2 --dtype bf16 \
  --checkpoint-dir /mnt/nfs/ckpts_a/pretrain

# ── SFT ──────────────────────────────────────────────────────────────
PORT=5678 bash hivemind/run.sh sft-bootstrap --model-size 300M \
  --data-dir /mnt/nfs/sft_packed \
  --batch-size 8 --grad-accum 4 --dtype bf16 \
  --lora-rank 64 --checkpoint-dir /mnt/nfs/ckpts_a/sft

# ── GRPO ──────────────────────────────────────────────────────────────
PORT=5678 bash hivemind/run.sh grpo-bootstrap \
  --checkpoint /mnt/nfs/ckpts_a/sft/latest.pt \
  --data-dir /mnt/nfs/grpo_packed \
  --tokenizer /mnt/nfs/tokenizer \
  --batch-size 8 --num-generations 8 --max-new-tokens 512 \
  --lora-rank 64 --out-dir /mnt/nfs/ckpts_a/grpo

# ── DPO ───────────────────────────────────────────────────────────────
PORT=5678 bash hivemind/run.sh dpo-bootstrap \
  --checkpoint /mnt/nfs/ckpts_a/sft/latest.pt \
  --data-dir /mnt/nfs/dpo_packed \
  --tokenizer /mnt/nfs/tokenizer \
  --batch-size 8 --beta 0.1 \
  --lora-rank 64 --out-dir /mnt/nfs/ckpts_a/dpo
```

### Machine B — Older Desktop (16 GB RAM, RTX 3050)

This machine has a smaller GPU with 4–6 GB VRAM.  Use smaller batch sizes,
fewer GRPO generations, and FP32 (or mixed-precision if supported).  Point
`--initial-peers` to Machine A's IP.

```bash
# ── Pretrain ────────────────────────────────────────────────────────
bash hivemind/run.sh worker 192.168.1.100:5678 --model-size 300M \
  --data-dir /mnt/nfs/packed \
  --batch-size 4 --grad-accum 4 --dtype bf16 \
  --checkpoint-dir /mnt/nfs/ckpts_b/pretrain

# ── SFT ──────────────────────────────────────────────────────────────
bash hivemind/run.sh sft-worker 192.168.1.100:5678 --model-size 300M \
  --data-dir /mnt/nfs/sft_packed \
  --batch-size 4 --grad-accum 8 --dtype bf16 \
  --lora-rank 32 --checkpoint-dir /mnt/nfs/ckpts_b/sft

# ── GRPO ──────────────────────────────────────────────────────────────
bash hivemind/run.sh grpo-worker 192.168.1.100:5678 \
  --checkpoint /mnt/nfs/ckpts_a/sft/latest.pt \
  --data-dir /mnt/nfs/grpo_packed \
  --tokenizer /mnt/nfs/tokenizer \
  --batch-size 4 --num-generations 4 --max-new-tokens 256 \
  --lora-rank 32 --out-dir /mnt/nfs/ckpts_b/grpo

# ── DPO ───────────────────────────────────────────────────────────────
bash hivemind/run.sh dpo-worker 192.168.1.100:5678 \
  --checkpoint /mnt/nfs/ckpts_a/sft/latest.pt \
  --data-dir /mnt/nfs/dpo_packed \
  --tokenizer /mnt/nfs/tokenizer \
  --batch-size 4 --beta 0.1 \
  --lora-rank 32 --out-dir /mnt/nfs/ckpts_b/dpo
```

### Machine C — Laptop (8 GB RAM, CPU only)

The humblest peer contributes whatever it can — every local step still counts
toward the global model.  Use `--dtype fp32`, `--batch-size 1`, and tiny
checkpoint intervals so you don't lose work if the laptop goes to sleep.

```bash
# ── Pretrain ────────────────────────────────────────────────────────
bash hivemind/run.sh worker 192.168.1.100:5678 --model-size 300M \
  --data-dir /mnt/nfs/packed \
  --batch-size 1 --grad-accum 1 --dtype fp32 \
  --checkpoint-dir /home/user/ckpts_c/pretrain

# ── SFT ──────────────────────────────────────────────────────────────
bash hivemind/run.sh sft-worker 192.168.1.100:5678 --model-size 300M \
  --data-dir /mnt/nfs/sft_packed \
  --batch-size 1 --grad-accum 1 --dtype fp32 \
  --lora-rank 8 --checkpoint-dir /home/user/ckpts_c/sft

# ── GRPO ──────────────────────────────────────────────────────────────
bash hivemind/run.sh grpo-worker 192.168.1.100:5678 \
  --checkpoint /mnt/nfs/ckpts_a/sft/latest.pt \
  --data-dir /mnt/nfs/grpo_packed \
  --tokenizer /mnt/nfs/tokenizer \
  --batch-size 1 --num-generations 2 --max-new-tokens 128 \
  --lora-rank 8 --out-dir /home/user/ckpts_c/grpo

# ── DPO ───────────────────────────────────────────────────────────────
bash hivemind/run.sh dpo-worker 192.168.1.100:5678 \
  --checkpoint /mnt/nfs/ckpts_a/sft/latest.pt \
  --data-dir /mnt/nfs/dpo_packed \
  --tokenizer /mnt/nfs/tokenizer \
  --batch-size 1 --beta 0.1 \
  --lora-rank 8 --out-dir /home/user/ckpts_c/dpo
```

### What to expect

| Machine | Est. relative throughput | Contributes |
|---------|------------------------|-------------|
| A (4090, 32 GB) | ~100% (baseline) | ~75% of all gradient updates |
| B (3050, 16 GB) | ~40% | ~20% of all gradient updates |
| C (CPU, 8 GB) | ~1–2% | ~5% of all gradient updates |

Even the slow CPU laptop adds value — its parameter updates are applied and
averaged just like the GPU peers', and the async all-reduce means nobody waits
for it.  The 4090 will do 10–15 local steps per CPU step, which is exactly
how Hivemind is designed to work.

### Tips for CPU peers

- Use `--save-every` every 10–20 steps so partial progress is preserved if the
  laptop goes to sleep.
- The CPU peer may lag behind by thousands of steps — this is fine.  Hivemind's
  `load_state_from_peers()` call on startup catches it up to the current swarm
  average.
- GRPO rollout generation is especially slow on CPU.  Reduce `--num-generations`
  to 1 or 2 and `--max-new-tokens` to 128.

## CLI Reference

### Shared Hivemind arguments (all training scripts)

```
--hivemind                     Enable decentralized training mode
--initial-peers ""             Bootstrap: empty = start new swarm
--initial-peers "ip:port"      Worker: point to any live peer
--host 0.0.0.0                 Interface to bind the P2P peer
--port 5678                    Port (0 = random for workers)
--peer-id ""                   Optional human-readable peer name
--target-group-size 8          Averaging fan-out (more = stabler but slower)
--averaging-period 1           All-reduce every N local steps
--average-parameters           Average parameters (not gradients) — more stable
--checkpoint-average-rounds 3  Rounds for final checkpoint averaging
--average-checkpoints          After training, produce merged checkpoint
```

### Architecture arguments (same as base framework)

```
--model-size 300M              Auto-detect architecture (supports 10M → 1T)
--batch-size N                 Per-peer batch size (varies by peer!)
--grad-accum N                 Gradient accumulation
--lr / --min-lr               Learning rate (auto-scaled by model size)
```

### Architecture variant arguments (all training scripts)

```
--arch dense|jamba             Architecture type (default: dense)
--layer-type sequential|parallel  Layer computation order (default: sequential)
--sliding-window-size N        Local sliding-window attention (0 = disabled)
--num-mtp-heads N              Multi-token prediction heads (0 = disabled)
--mtp-discount F               Discount factor for MTP loss (default 0.5)
--mod-alpha F                  Mixture-of-Depth threshold (0 = disabled)
--use-mla                      Enable Multi-head Latent Attention (DeepSeek-style)
--kv-lora-rank N               KV compression rank for MLA (default: hidden_size//4)
--jamba-interval N             Attention layer every N layers in Jamba (default 4)
```

## Multi-Model Training (GRPO / DPO)

GRPO and DPO involve **two** models — a **policy** (trainable) and a **reference** (frozen). Only the policy's optimizer should be wrapped in `DecentralizedOptimizer`:

```
┌─────────────────────────────────────────────┐
│  Peer                                       │
│                                             │
│  ┌──────────┐    ┌──────────────┐           │
│  │  Policy   │    │  Reference   │           │
│  │  Model    │    │  Model       │           │
│  │  (train)  │    │  (frozen)    │           │
│  └─────┬────┘    └──────────────┘           │
│        │                                    │
│  ┌─────▼──────┐                              │
│  │ Decentralized │  ← async all-reduce       │
│  │ Optimizer   │    with other peers         │
│  └─────────────┘                              │
└─────────────────────────────────────────────┘
```

Key points:
- **Policy model**: trainable parameters are averaged across peers via `DecentralizedOptimizer`
- **Reference model**: fully local and frozen — **not** averaged (each peer keeps its own copy)
- **Data sharding**: each peer sees a different subset of prompts/preference-pairs via endpoint-hash sharding
- **Checkpoints**: save the inner (unwrapped) optimizer state so checkpoints are portable across Hivemind/non-Hivemind runs

## Heterogeneous Training Tips

### Batch sizes
Each peer can use a **different** `--batch-size`. A 4090 with 24 GB can do 8, a laptop with 4 GB can do 2. Hivemind doesn't care — the per-step all-reduce averages the resulting parameter updates regardless.

### CPU-only training
Hivemind works on CPU too (just much slower). The async averaging means even a CPU-only peer contributes:
```bash
python hivemind/train_pretrain_hivemind.py --hivemind \
  --initial-peers "192.168.1.100:5678" \
  --model-size 300M --dtype fp32 --batch-size 1 \
  --data-dir ./packed --checkpoint-dir ./ckpts_cpu
```

### Network
- **LAN**: direct connection works
- **WAN / different networks**: use a mesh VPN like [Tailscale](https://tailscale.com) or [ZeroTier](https://zerotier.com). All peers appear on a virtual LAN and Hivemind works transparently.

### Firewall
The bootstrap peer needs port `--port` (default random, but set it explicitly for bootstrap) reachable from workers.

### Data access
Every peer needs access to the data files. Options:
1. **Shared NFS / network drive** — simplest
2. **Copy data to each machine** — works, no network dependency
3. **Different shards** — each peer can have a unique subset; Hivemind only averages parameters

### Resuming
Each peer resumes from its **own** checkpoint directory:
```bash
python hivemind/train_grpo_hivemind.py --hivemind \
  --initial-peers "192.168.1.100:5678" \
  --resume ./grpo_ckpts/grpo_step0000050.pt
```

## How It Works Under the Hood

### `DecentralizedOptimizer`

The core is Hivemind's `DecentralizedOptimizer`, which wraps any `torch.optim.Optimizer`:

```python
base_opt = torch.optim.AdamW(model.parameters(), lr=lr)
hopt = DecentralizedOptimizer(
    params=model.parameters(),
    opt=base_opt,
    peer=peer,
    target_group_size=8,
    averaging_period=1,
)
# In the training loop — use hopt.step() instead of base_opt.step()
hopt.step()   # local update + async all-reduce
```

Each call to `step()`:
1. Applies the local optimizer's step (gradient update)
2. Triggers an **asynchronous all-reduce** with `target_group_size` random peers
3. Returns immediately — the training loop continues
4. When the all-reduce finishes, averaged parameters are written into the model

### Multi-model support (GRPO/DPO)

For GRPO and DPO, only the policy model's parameters should be wrapped:

```python
# Policy optimizer — wrapped in Hivemind for async averaging
policy_opt = torch.optim.AdamW(policy_model.parameters(), lr=lr)
hopt = DecentralizedOptimizer(
    params=policy_model.parameters(),
    opt=policy_opt,
    peer=peer,
    target_group_size=8,
    averaging_period=1,
)

# Reference model — stays local, not averaged
ref_model = build_reference("two", config, checkpoint_path, device)
# ref_model is frozen and used only for KL / log-ratio computation
```

### Data Sharding

Each peer deterministically derives a shard index from its endpoint hash, ensuring different peers see different data. The total number of shards is estimated from `target_group_size` + visible peers.

### Checkpoint Averaging

After training (or periodically), `average_checkpoints_via_hivemind()` performs several rounds of parameter-only all-reduce across the swarm, producing a merged state dict that typically has better quality than any single peer's checkpoint.

## Comparison: DDP vs Hivemind

| Aspect | DDP | Hivemind |
|--------|-----|----------|
| GPU homogeneity | Required | Not required |
| Sync barrier | Every step | None (async) |
| Network | NVLink / InfiniBand | TCP / Internet |
| Batch size | Same per GPU | Can vary per peer |
| Speed | Limited by slowest peer | Faster peers contribute more |
| Checkpoint | One per group | One per peer (independently) |
| Complexity | Simple (torchrun) | Slightly more (swarm setup) |
| Fault tolerance | Low (one fails = all fail) | High (peers join/leave freely) |
| Multi-model (GRPO/DPO) | All models in DDP | Only policy averaged |
