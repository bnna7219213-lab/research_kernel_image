#!/usr/bin/env bash
# ============================================================
# T2Image Phase-A — dataset downloader
# Usage: ./download_datasets.sh <laion|coco|all> [--max N]
# Note: LAION is huge (billions). Use --max for dev samples.
# ============================================================
set -euo pipefail

DATASET="${1:?Usage: $0 <laion|coco|all> [--max N]}"
shift || true
MAX=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max) MAX="$2"; shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

DATA_ROOT="${T2IMAGE_DATA_ROOT:-/data/t2image}"
RAW_DIR="${DATA_ROOT}/raw"
SPEC_DIR="${DATA_ROOT}/_spec"
mkdir -p "${RAW_DIR}/laion_aesthetics" "${SPEC_DIR}"

echo ">>> T2Image dataset download | dataset=${DATASET} max=${MAX:-all}"
echo ">>> DATA_ROOT = ${DATA_ROOT}"

# ============================================================
# LAION-Aesthetics V2 (via img2dataset — fast parallel)
# ============================================================
download_laion() {
  echo ""
  echo "=== LAION-Aesthetics V2 (img2dataset) ==="
  echo "Note: this dataset contains billions of images. Use --max N for dev."

  # img2dataset reads a parquet index and downloads images in parallel
  # Reference: https://github.com/rom1504/img2dataset
  if ! command -v img2dataset >/dev/null 2>&1; then
    echo ">> Installing img2dataset..."
    python -m pip install --quiet img2dataset
  fi

  # Simplified dev pipeline: pick 1 image from the list
  # In production: decompress + shard the full parquet
  echo ">> Running img2dataset on LAION-Aesthetics (sample mode)..."
  cd "${RAW_DIR}/laion_aesthetics" || exit 1
  if [[ ${MAX} -gt 0 ]]; then
    img2dataset \
        "https://the-eye.eu/public/AI/cah/laion-aesthetics-V2/data/v1_0_with_nsfw.parquet" \
        input_format="parquet" \
        url_col="URL" \
        caption_col="TEXT" \
        output_format="files" \
        output_folder="$(pwd)" \
        thread_count=16 \
        number_sample_per_shard=1000 \
        incremental="incremental" \
        save_additional_columns='["similarity","hash","punsafe","pwatermark","AESTHETIC_SCORE"]' \
        max_shard_id=0 \
        max_sample=${MAX} \
        2>&1 | tail -10
  else
    echo ">> LAION download skipped (specify --max N to download sample)."
  fi
  cd -
  COUNT=$(find "${RAW_DIR}/laion_aesthetics" -name "*.jpg" 2>/dev/null | wc -l)
  echo ">> LAION done. Files: ${COUNT}"
}

# ============================================================
# COCO 2014 val (≈1GB, for FID reference)
# ============================================================
download_coco() {
  echo ""
  echo "=== COCO 2014 val (FID reference) ==="
  COCO_URL="http://images.cocodataset.org/zips/val2014.zip"
  COCO_ZIP="${RAW_DIR}/val2014.zip"

  if [[ ! -f "${COCO_ZIP}" ]]; then
    echo ">> Downloading COCO val2014 (~1GB)..."
    wget -q --show-progress -O "${COCO_ZIP}" "${COCO_URL}" || {
      curl -Lo "${COCO_ZIP}" "${COCO_URL}"
    }
  fi

  if [[ ! -d "${DATA_ROOT}/coco/val2014" ]]; then
    echo ">> Unzipping..."
    unzip -q -o "${COCO_ZIP}" -d "${DATA_ROOT}/coco"
  fi

  # Create captions CSV for CLIP / PickScore
  echo ">> Generating captions CSV (run separately via COCO API script)..."
  COUNT=$(find "${DATA_ROOT}/coco/val2014" -name "*.jpg" | wc -l)
  echo ">> COCO done. Val images: ${COUNT}"
}

# ---------- dispatch ----------
case "${DATASET}" in
  laion)  download_laion ;;
  coco)   download_coco ;;
  all)    download_laion; download_coco ;;
  *)      echo "Unknown dataset: ${DATASET}"; exit 1 ;;
esac
echo ""
echo "=== Done. Data root: ${DATA_ROOT} ==="
