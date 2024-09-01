#!/bin/bash

echo "----------Training Starge--------------"
python ../demo_classification.py -net "mla" \
                    -d "seam" \
                    -dn "cuda" \
                    -v "small"\
                    -loss "wdice"\
                    -cp "unfrozen"\
                    -l 1e-5
echo "----------Training Over----------------"

echo "---------------Evaluate----------------"
python ../evaluate_classification.py -net "mla" \
                    -d "seam" \
                    -dn "cuda" \
                    -v "small"\
                    -loss "wdice"\
                    -cp "unfrozen"
echo "Done"
