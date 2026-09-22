#!/usr/bin/env bash
# T2Image entrypoint — routes to sub-commands
set -euo pipefail

CMD="${1:-help}"
shift || true

case "$CMD" in
  help)
    echo "T2Image Phase-A CLI"
    echo ""
    echo "Commands:"
    echo "  train-pixart        Launch 8-card PixArt-Σ training (FSDP)"
    echo "  train-flux          Launch 8-card Flux training"
    echo "  infer-pixart        Run PixArt-Σ inference on prompts file"
    echo "  infer-flux          Run Flux inference on prompts file"
    echo "  eval-fid            Compute FID-10K"
    echo "  eval-clip           Compute CLIP score"
    echo "  eval-pick           Compute PickScore / ImageReward"
    echo "  eval-hps            Compute HPSv2.1"
    echo "  eval-all            Run all evaluations"
    echo "  data-fingerprint    Compute dataset fingerprint"
    echo "  download-laion      Download LAION-Aesthetics via img2dataset"
    echo "  clean-data          Recaption + CLIP score + NSFW filtering"
    echo "  bash                Drop into shell"
    ;;

  train-pixart)
    torchrun --nproc_per_node=8 \
        /workspace/src/train/main.py \
        --config /workspace/configs/pixart_sigma.yaml "$@"
    ;;

  train-flux)
    torchrun --nproc_per_node=8 \
        /workspace/src/train/main.py \
        --config /workspace/configs/flux1_dev.yaml "$@"
    ;;

  infer-pixart)
    python /workspace/src/train/infer_pixart.py "$@"
    ;;

  infer-flux)
    python /workspace/src/train/infer_flux.py "$@"
    ;;

  eval-fid)
    python /workspace/src/eval/compute_fid.py "$@"
    ;;

  eval-clip)
    python /workspace/src/eval/compute_clip.py "$@"
    ;;

  eval-pick)
    python /workspace/src/eval/compute_pick.py "$@"
    ;;

  eval-hps)
    python /workspace/src/eval/compute_hps.py "$@"
    ;;

  eval-all)
    python /workspace/src/eval/compute_fid.py "$@"
    python /workspace/src/eval/compute_clip.py "$@"
    python /workspace/src/eval/compute_pick.py "$@"
    ;;

  data-fingerprint)
    python /workspace/scripts/compute_data_fingerprint.py "$@"
    ;;

  download-laion)
    bash /workspace/scripts/download_datasets.sh laion "$@"
    ;;

  clean-data)
    python /workspace/scripts/clean_data.py "$@"
    ;;

  bash)
    exec /bin/bash
    ;;

  *)
    echo "Unknown command: $CMD"; exit 1 ;;
esac
