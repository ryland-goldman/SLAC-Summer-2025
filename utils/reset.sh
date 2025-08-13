#!/bin/sh
# This script will configure the EC2 instance to start running
# g4beamline's directory is /home/ubuntu/G4beamline-3.08/bin/g4bl

rm *.txt
rm *.dat
rm *.csv
rm *.out
rm *.py
rm *.g4bl
rm *.zip
nano downloadedscript.py
./start-ramdisk.sh
cd /tmp/ramdisk
nano GridModerator.g4bl
nano s.sh
chmod +x s.sh
ulimit -n 100000
screen