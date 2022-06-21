#!/bin/bash
nohup python3 inference_tta_split.py --mode private --checkpoint 06-06-10-04-43 06-06-10-03-00 06-06-09-29-55 06-06-07-46-11 06-06-05-29-11 --topk 1 1 1 1 1 > infer.txt &