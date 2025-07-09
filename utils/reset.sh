rm *.txt
rm *.dat
rm *.csv
rm *.out
rm *.py
rm *.g4bl
rm *.zip
nano downloadedscript.py
nano Combined.g4bl
./start-ramdisk.sh
cd /tmp/ramdisk
ulimit -n 100000
screen

while true; do python downloadedscript.py; done