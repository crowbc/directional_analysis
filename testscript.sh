#!/bin/bash
for i in {1..5}; do python3 directional_analysis.py it_num"$i".txt "$i"; done
