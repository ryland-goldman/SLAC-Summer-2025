#!/bin/bash

cd data-amd
for i in {1..16}; do /home/ubuntu/G4beamline-3.08/bin/g4bl LBand.g4bl T="$i" TIncA="$1" TIncB="$2" TIncC="$3" TIncD="$4" TIncE="$5" Grad=15 G="$6" & done; wait
cd ..
cat LBandOut.$6.*.txt > LBandOut.$6.txt