#!/bin/bash
# End-to-end: zero-shot Reuters dataset -> lexical baseline vs sparse semantic
# atoms vs RQ-style codes, evaluated with seen/unseen splits.
# Usage: ./semantic_atoms/run_experiment.sh   (from repo root; needs `make` done)
set -e
HP="-bilinear_classifier_cost 5 -bs_count 60 -bs_direct_wt 0.8 -bs_alpha 0.02 -shortyK 30"

python3 semantic_atoms/build_reuters.py
python3 semantic_atoms/build_atoms.py GZXML-Datasets/GZ-Reuters-90 \
        --n-atoms 512 --k 16 --atom-wt 0.3 --out GZXML-Datasets/GZ-Reuters-90-atoms
( cd semantic_atoms && python3 build_rq_codes.py ../GZXML-Datasets/GZ-Reuters-90 \
        --out ../GZXML-Datasets/GZ-Reuters-90-rqcodes )

for ds in GZ-Reuters-90 GZ-Reuters-90-atoms GZ-Reuters-90-rqcodes; do
    ./run.sh $ds train   $HP
    ./run.sh $ds predict $HP
done

python3 semantic_atoms/evaluate.py GZ-Reuters-90 GZ-Reuters-90-atoms GZ-Reuters-90-rqcodes
