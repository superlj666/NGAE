#!/bin/bash

python bo.py \
  --data-name="asia_200k" \
  --save-appendix="NGAE" \
  --checkpoint=100 \
  --vis-2d \
  --res-dir="vis_results/" \

  #--save-appendix="SVAE" \
