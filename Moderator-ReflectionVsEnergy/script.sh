#!/bin/bash

set -euo pipefail

# Source environment once
source ~/.bashrc

# Loop over requested energies
for E in 0.015 0.02 0.03 0.05 0.075 0.1 0.15 0.2 0.3 0.5 0.75 1 1.5 2 2.5; do  

  echo
  echo $E

  # Clean per-energy scratch data
  rm -f data-dir/*

  # Run g4bl and processing in parallel over T=1..14
  seq 1 14 | xargs -n1 -P14 -I{} g4bl Moderator.g4bl KE=$E T={} > /dev/null
  echo "Processing 1..."
  seq 1 14 | xargs -n1 -P14 -I{} python3 ModeratorProcess1.py {} > /dev/null

  echo "Processing 2..."

  # Append energy tag and aggregated results
  echo "$E" >> out.txt
  python3 ModeratorProcess2.py --E $E >> out.txt
  echo >> out.txt
done