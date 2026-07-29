#!/usr/bin/env bash
# ===========================================================================
# run.sh — Hivemind multi-peer launcher examples
# ===========================================================================
# This script demonstrates how to start a Hivemind swarm.  It does NOT
# launch all peers automatically — you run it separately on each machine
# (or in separate terminals) with the appropriate role.
#
# Usage
# -----
#   # 1. Bootstrap peer (first machine)
#   bash hivemind/run.sh bootstrap --model-size 300M --data-dir ./packed
#
#   # 2. Worker peers (other machines, different terminals)
#   bash hivemind/run.sh worker 192.168.1.100:5678 --model-size 300M --data-dir ./packed
#
#   # 3. List all options
#   bash hivemind/run.sh help
#
# Tips
# ----
#   - All peers must use the SAME --model-size (same architecture).
#   - All peers must have access to the SAME data (nfs / same .bin files
#     at --data-dir) OR different shards — Hivemind averages parameters,
#     not data.
#   - The bootstrap peer's IP must be reachable from workers (no NAT, or
#     use a VPN like Tailscale / ZeroTier).
#   - Use --help to see ALL training arguments.
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

MODE="${1:-help}"
shift || true

case "$MODE" in
  bootstrap)
    echo "=== Starting Hivemind BOOTSTRAP peer ==="
    echo "    Other peers will connect to this node."
    echo ""

    PORT="${PORT:-5678}"
    exec python "$REPO_DIR/hivemind/train_pretrain_hivemind.py" \
      --hivemind \
      --initial-peers "" \
      --port "$PORT" \
      "$@"
    ;;

  worker)
    BOOTSTRAP_ADDR="${1}"
    shift || {
      echo "Usage: run.sh worker <bootstrap-ip:port> [args...]"
      exit 1
    }
    echo "=== Starting Hivemind WORKER peer ==="
    echo "    Connecting to bootstrap peer at $BOOTSTRAP_ADDR"
    echo ""

    exec python "$REPO_DIR/hivemind/train_pretrain_hivemind.py" \
      --hivemind \
      --initial-peers "$BOOTSTRAP_ADDR" \
      --port 0 \
      "$@"
    ;;

  sft-bootstrap)
    echo "=== Starting Hivemind SFT BOOTSTRAP peer ==="
    echo ""

    PORT="${PORT:-5678}"
    exec python "$REPO_DIR/hivemind/train_sft_hivemind.py" \
      --hivemind \
      --initial-peers "" \
      --port "$PORT" \
      "$@"
    ;;

  sft-worker)
    BOOTSTRAP_ADDR="${1}"
    shift || {
      echo "Usage: run.sh sft-worker <bootstrap-ip:port> [args...]"
      exit 1
    }
    exec python "$REPO_DIR/hivemind/train_sft_hivemind.py" \
      --hivemind \
      --initial-peers "$BOOTSTRAP_ADDR" \
      --port 0 \
      "$@"
    ;;

  grpo-bootstrap)
    echo "=== Starting Hivemind GRPO BOOTSTRAP peer ==="
    echo ""

    PORT="${PORT:-5678}"
    exec python "$REPO_DIR/hivemind/train_grpo_hivemind.py" \
      --hivemind \
      --initial-peers "" \
      --port "$PORT" \
      "$@"
    ;;

  grpo-worker)
    BOOTSTRAP_ADDR="${1}"
    shift || {
      echo "Usage: run.sh grpo-worker <bootstrap-ip:port> [args...]"
      exit 1
    }
    echo "=== Starting Hivemind GRPO WORKER peer ==="
    echo "    Connecting to bootstrap peer at $BOOTSTRAP_ADDR"
    echo ""

    exec python "$REPO_DIR/hivemind/train_grpo_hivemind.py" \
      --hivemind \
      --initial-peers "$BOOTSTRAP_ADDR" \
      --port 0 \
      "$@"
    ;;

  dpo-bootstrap)
    echo "=== Starting Hivemind DPO BOOTSTRAP peer ==="
    echo ""

    PORT="${PORT:-5678}"
    exec python "$REPO_DIR/hivemind/train_dpo_hivemind.py" \
      --hivemind \
      --initial-peers "" \
      --port "$PORT" \
      "$@"
    ;;

  dpo-worker)
    BOOTSTRAP_ADDR="${1}"
    shift || {
      echo "Usage: run.sh dpo-worker <bootstrap-ip:port> [args...]"
      exit 1
    }
    echo "=== Starting Hivemind DPO WORKER peer ==="
    echo "    Connecting to bootstrap peer at $BOOTSTRAP_ADDR"
    echo ""

    exec python "$REPO_DIR/hivemind/train_dpo_hivemind.py" \
      --hivemind \
      --initial-peers "$BOOTSTRAP_ADDR" \
      --port 0 \
      "$@"
    ;;

  average)
    # Run checkpoint averaging across the swarm (no training, just merge)
    echo "=== Averaging checkpoints across Hivemind swarm ==="
    exec python "$REPO_DIR/hivemind/train_pretrain_hivemind.py" \
      --hivemind \
      --initial-peers "$1" \
      --average-checkpoints \
      --checkpoint-dir "${2:-./averaged_checkpoint}" \
      --num-steps 0
    ;;

  help|--help|-h)
    echo "============================================================================"
    echo " Hivemind Multi-Peer Launcher"
    echo "============================================================================"
    echo ""
    echo " Commands:"
    echo "   pretrain / training:"
    echo "   bootstrap              Start as pretrain bootstrap peer (first node)"
    echo "   worker <ip:port>       Join as pretrain worker peer"
    echo "   average <ip:port> [dir]  Average checkpoints across swarm"
    echo ""
    echo "   SFT:"
    echo "   sft-bootstrap          Start SFT bootstrap peer"
    echo "   sft-worker <ip:port>   Join SFT worker peer"
    echo ""
    echo "   GRPO (reinforcement learning):"
    echo "   grpo-bootstrap         Start GRPO bootstrap peer"
    echo "   grpo-worker <ip:port>  Join GRPO worker peer"
    echo ""
    echo "   DPO (preference optimization):"
    echo "   dpo-bootstrap          Start DPO bootstrap peer"
    echo "   dpo-worker <ip:port>   Join DPO worker peer"
    echo ""
    echo " Environment variables:"
    echo "   PORT          Peer port (default 5678 for bootstrap)"
    echo ""
    echo " Examples:"
    echo "   # Machine A (pretrain bootstrap):"
    echo "     PORT=5678 bash hivemind/run.sh bootstrap --model-size 300M \\"
    echo "       --data-dir ./packed --batch-size 4"
    echo ""
    echo "   # Machine B (pretrain worker, points to A at 192.168.1.5):"
    echo "     bash hivemind/run.sh worker 192.168.1.5:5678 --model-size 300M \\"
    echo "       --data-dir ./packed --batch-size 2"
    echo ""
    echo "   # SFT swarm (bootstrap):"
    echo "     bash hivemind/run.sh sft-bootstrap --model-size 300M \\"
    echo "       --data-dir ./sft_packed --lora-rank 64"
    echo ""
    echo "   # GRPO swarm (worker):"
    echo "     bash hivemind/run.sh grpo-worker 192.168.1.5:5678 \\"
    echo "       --checkpoint ./sft_ckpts/latest.pt --data-dir ./grpo_packed \\"
    echo "       --tokenizer ./tokenizer --batch-size 2"
    echo ""
    echo "   # DPO swarm (worker):"
    echo "     bash hivemind/run.sh dpo-worker 192.168.1.5:5678 \\"
    echo "       --checkpoint ./sft_ckpts/latest.pt --data-dir ./dpo_packed \\"
    echo "       --tokenizer ./tokenizer --batch-size 2"
    echo ""
    echo "   # Machine C (laptop, CPU only, small batch):"
    echo "     bash hivemind/run.sh worker 192.168.1.5:5678 --model-size 300M \\"
    echo "       --data-dir ./packed --batch-size 1 --dtype fp32"
    echo ""
    echo " Notes:"
    echo "   - All peers in the same swarm must use the same --model-size."
    echo "   - Each peer can have a different --batch-size / --grad-accum."
    echo "   - Data must be accessible by all peers (shared NFS or copy)."
    echo "   - Use a VPN (Tailscale, ZeroTier) if machines are on different networks."
    echo "============================================================================"
    ;;

  *)
    echo "Unknown mode: $MODE"
    echo "Usage: run.sh <bootstrap|worker|sft-bootstrap|sft-worker|"
    echo "              grpo-bootstrap|grpo-worker|dpo-bootstrap|dpo-worker|"
    echo "              average|help> [args...]"
    exit 1
    ;;
esac
