#!/bin/bash
# Read a string with spaces using for loop

#source /venv/dl/bin/activate


CMD="python -m prlab.cli run --json_conf config/nsclc.json --run to-del "

epochs=200

$CMD --epochs $epochs
