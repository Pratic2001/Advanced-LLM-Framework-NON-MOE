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
| `train_pretrain_hivemind.py` | Decentralized pretraining (main training entry point) |
| `train_sft_hivemind.py` | Supervised fine-tuning with Hivemind (supports LoRA/DoRA) |
| `run.sh` | Convenience launcher for bootstrap/worker modes |
| `requirements-hivemind.txt` | Additional dependencies |

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
Every peer needs access to the packed data files. Options:
1. **Shared NFS / network drive** — simplest
2. **Copy data to each machine** — works, no network dependency
3. **Different shards** — each peer can have a unique subset; Hivemind only averages parameters

### Resuming
Each peer resumes from its **own** checkpoint directory:
```bash
python hivemind/train_pretrain_hivemind.py --hivemind \
  --initial-peers "192.168.1.100:5678" \
  --resume ./hivemind_ckpts_c
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

## Extending to GRPO / DPO

The same `DecentralizedOptimizer` pattern can wrap GRPO and DPO optimizers.
The key difference is that GRPO and DPO involve multiple models (policy,
reference, reward) — only the policy model's parameters should be averaged
across peers:

```python
# Example pattern for GRPO with Hivemind
policy_opt = torch.optim.AdamW(policy_model.parameters(), lr=lr)
hopt = build_hivemind_optimizer(policy_model, policy_opt, peer, ...)
# Reference model and reward model stay local (not averaged)
```

See `train_sft_hivemind.py` for a complete LoRA/DoRA example that can
serve as the template for GRPO/DPO adaptation.
