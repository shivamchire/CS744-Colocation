#!/bin/bash
#python main.py -m lstm -b 20 --num_steps 10000 -t inference -log_file /users/hshivam/CS744-Colocation/v100_lat_exp2/1_lstm.log
#python main.py -m lstm,s2s -b 20,32 --num_steps 20000,1000 -t inference,training -log_file /users/hshivam/CS744-Colocation/v100_lat_exp2/2_lstm.log,/users/hshivam/CS744-Colocation/v100_lat_exp2/2_s2s.log
python main.py -m s2s,lstm -b 32,20 --num_steps 1000,50000 -t training,inference -log_file /users/hshivam/CS744-Colocation/v100_lat_exp2/2_s2s.log,/users/hshivam/CS744-Colocation/v100_lat_exp2/2_lstm.log
#python main.py -m lstm,s2s,resnet50 -b 20,32,32 --num_steps 10000,1000,1000 -t inference,training,training -log_file /users/hshivam/CS744-Colocation/v100_lat_exp2/3_lstm.log,/users/hshivam/CS744-Colocation/v100_lat_exp2/3_s2s.log,/users/hshivam/CS744-Colocation/v100_lat_exp2/3_resnet50.log
#python main.py -m lstm,s2s,resnet50,cyclegan -b 20,32,32,1 --num_steps 10000,1000,1000,1000 -t inference,training,training,training -log_file /users/hshivam/CS744-Colocation/v100_lat_exp2/4_lstm.log,/users/hshivam/CS744-Colocation/v100_lat_exp2/4_s2s.log,/users/hshivam/CS744-Colocation/v100_lat_exp2/4_resnet50.log,/users/hshivam/CS744-Colocation/v100_lat_exp2/4_cyclegan.log

#python parse_log.py -l /users/hshivam/CS744-Colocation/v100_lat_exp2/1_lstm.log -o /users/hshivam/CS744-Colocation/v100_lat_exp2/1_out.log
python parse_log.py -l /users/hshivam/CS744-Colocation/v100_lat_exp2/2_lstm.log,/users/hshivam/CS744-Colocation/v100_lat_exp2/2_s2s.log -o /users/hshivam/CS744-Colocation/v100_lat_exp2/2_out.log
#python parse_log.py -l /users/hshivam/CS744-Colocation/v100_lat_exp2/3_lstm.log,/users/hshivam/CS744-Colocation/v100_lat_exp2/3_s2s.log,/users/hshivam/CS744-Colocation/v100_lat_exp2/3_resnet50.log -o /users/hshivam/CS744-Colocation/v100_lat_exp2/3_out.log
#python parse_log.py -l /users/hshivam/CS744-Colocation/v100_lat_exp2/4_lstm.log,/users/hshivam/CS744-Colocation/v100_lat_exp2/4_s2s.log,/users/hshivam/CS744-Colocation/v100_lat_exp2/4_resnet50.log,/users/hshivam/CS744-Colocation/v100_lat_exp2/4_cyclegan.log -o /users/hshivam/CS744-Colocation/v100_lat_exp2/4_out.log
