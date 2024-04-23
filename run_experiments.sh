#!/bin/bash
python main.py -m vgg -n 10000 -b 20 --log_file /users/hshivam/CS744-Colocation/v100_exp_raw_data/vgg/train.log -t training --enable_perf_log
python main.py -m vgg -n 10000 -b 20 --log_file /users/hshivam/CS744-Colocation/v100_exp_raw_data/vgg/infer.log -t inference --enable_perf_log
python main.py -m cyclegan -n 1000 -b 1 --log_file /users/hshivam/CS744-Colocation/v100_exp_raw_data/cyclegan/train.log -t training --enable_perf_log
python main.py -m cyclegan -n 1000 -b 1 --log_file /users/hshivam/CS744-Colocation/v100_exp_raw_data/cyclegan/infer.log -t inference --enable_perf_log
python main.py -m lstm -n 10000 -b 20 --log_file /users/hshivam/CS744-Colocation/v100_exp_raw_data/lstm/train.log -t training --enable_perf_log
python main.py -m lstm -n 10000 -b 20 --log_file /users/hshivam/CS744-Colocation/v100_exp_raw_data/lstm/infer.log -t inference --enable_perf_log

