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
nano CombinedWithPhotonTrack5000.g4bl # change nEvents=1000
nano s.sh
chmod +x s.sh
ulimit -n 100000
screen

# while true; do ./s.sh; done

# zip -r out.zip Out*.dat

# #!/bin/bash
# for i in {0..191}; do python3 downloadedscript.py "$i" & done; wait