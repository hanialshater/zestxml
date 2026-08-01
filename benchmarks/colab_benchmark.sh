#!/bin/bash
# Benchmark the PyTorch port against the C++ reference on a real GZXML dataset.
# Intended for a GPU box (Colab A100); works on CPU too, just slowly.
#
#   bash benchmarks/colab_benchmark.sh                 # GZ-Eurlex-4.3K, both implementations
#   bash benchmarks/colab_benchmark.sh GZ-Eurlex-4.3K torch    # skip the C++ baseline
#
# Everything it writes lives under GZXML-Datasets/ and Results/.
set -u

DATASET=${1:-GZ-Eurlex-4.3K}
WHICH=${2:-both}          # both | torch | cpp
DATA_DIR=GZXML-Datasets/${DATASET}

# hyper-parameters from run_eurlex.sh, applied identically to both implementations
ARGS="-bs_count 120 -bs_alpha 0.02 -bs_direct_wt 0.8 -shortyK 150
      -bilinear_classifier_cost 5 -bilinear_normalize 0 -num_thread 0"

case "$DATASET" in
  GZ-Eurlex-4.3K) DRIVE_ID=1j27bQZol6gOQ7AATawShcF4jXJr3Venb ;;
  *)              DRIVE_ID="" ;;
esac

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31m!! %s\033[0m\n' "$*"; exit 1; }

say "environment"
python3 -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(),
      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu only')" || fail "torch missing"
nproc | sed 's/^/cpu cores: /'

# ---------------------------------------------------------------- dataset
if [ ! -d "$DATA_DIR" ]; then
  [ -n "$DRIVE_ID" ] || fail "no download id known for $DATASET; put it in $DATA_DIR yourself"
  say "downloading $DATASET"
  pip install -q gdown
  mkdir -p GZXML-Datasets && (
    cd GZXML-Datasets &&
    gdown "https://drive.google.com/uc?id=${DRIVE_ID}" -O "${DATASET}.tar.gz" &&
    tar -xzf "${DATASET}.tar.gz" && rm -f "${DATASET}.tar.gz"
  ) || fail "download failed -- fetch it by hand into $DATA_DIR"
fi
[ -f "$DATA_DIR/trn_X_Xf.txt" ] || fail "$DATA_DIR does not look like a GZXML dataset"
say "dataset"
head -1 "$DATA_DIR/trn_X_Xf.txt" | awk '{print "  train points x point features:", $1, "x", $2}'
head -1 "$DATA_DIR/tst_X_Xf.txt" | awk '{print "  test points               :", $1}'
head -1 "$DATA_DIR/Y_Yf.txt"     | awk '{print "  labels x label features   :", $1, "x", $2}'

data_args="-trn_X_Xf $DATA_DIR/trn_X_Xf.txt -tst_X_Xf $DATA_DIR/tst_X_Xf.txt
           -Y_Yf $DATA_DIR/Y_Yf.txt -trn_X_Y $DATA_DIR/trn_X_Y.txt -tst_X_Y $DATA_DIR/tst_X_Y.txt
           -Xf $DATA_DIR/Xf.txt -Yf $DATA_DIR/Yf.txt"

# ---------------------------------------------------------------- C++ baseline
if [ "$WHICH" != "torch" ]; then
  say "building the C++ reference"
  make >/dev/null 2>&1 || fail "make failed"
  RES=Results/${DATASET}-cpp
  rm -rf "$RES"; mkdir -p "$RES/model"
  say "C++ train"
  t0=$(date +%s)
  ./run $data_args -res_dir "$RES" -model_dir "$RES/model" -type train $ARGS 2>&1 \
    | grep -E "acc :|recall of|nnz of|STAT" || fail "C++ train failed"
  t1=$(date +%s)
  say "C++ predict"
  ./run $data_args -res_dir "$RES" -model_dir "$RES/model" -type predict $ARGS 2>&1 \
    | grep -E "recall of|nnz of|STAT" || fail "C++ predict failed"
  t2=$(date +%s)
  echo "cpp_train_s=$((t1 - t0)) cpp_predict_s=$((t2 - t1))" > "$RES/timing.txt"
  cat "$RES/timing.txt"
fi

# ---------------------------------------------------------------- PyTorch
if [ "$WHICH" != "cpp" ]; then
  RES=Results/${DATASET}-torch
  rm -rf "$RES"; mkdir -p "$RES/model"
  # A100 has room for bigger working blocks than the conservative defaults
  BIG="-max_elems 67108864 -dense_elems 33554432 -device auto"
  say "PyTorch train"
  t0=$(date +%s)
  python3 run_torch.py $data_args -res_dir "$RES" -model_dir "$RES/model" -type train $ARGS $BIG 2>&1 \
    | grep -E "acc :|recall of|nnz of|STAT|epoch|finished" || fail "torch train failed"
  t1=$(date +%s)
  say "PyTorch predict"
  python3 run_torch.py $data_args -res_dir "$RES" -model_dir "$RES/model" -type predict $ARGS $BIG 2>&1 \
    | grep -E "recall of|nnz of|STAT|finished" || fail "torch predict failed"
  t2=$(date +%s)
  echo "torch_train_s=$((t1 - t0)) torch_predict_s=$((t2 - t1))" > "$RES/timing.txt"
  cat "$RES/timing.txt"
fi

# ---------------------------------------------------------------- metrics
say "metrics"
for impl in cpp torch; do
  f=Results/${DATASET}-${impl}/score_mat.bin
  [ -f "$f" ] && python3 tools/eval_xc.py "$f" "$DATA_DIR"
done

say "timings"
cat Results/${DATASET}-*/timing.txt 2>/dev/null

say "done -- Results/${DATASET}-{cpp,torch}/"
