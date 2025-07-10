#!/bin/bash

for i in {0..14}
do
    python3 batch_threaded_withphotontrack.py "$i" &
done

wait