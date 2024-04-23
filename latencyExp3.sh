#!/bin/bash
h=/users/hshivam/CS744-Colocation/v100_lat_exp7
python main.py -m resnet50 -b 32 --num_steps 1000 -t inference -log_file $h/1_i.log
python main.py -m resnet50,resnet50 -b 32,32 --num_steps 1000,1000 -t inference,training -log_file $h/2_i.log,$h/2_t.log
python main.py -m resnet50,resnet50,resnet50 -b 32,32,32 --num_steps 1000,1000,1000 -t inference,training,training -log_file $h/3_i.log,$h/3_t1.log,$h/3_t2.log
python main.py -m resnet50,resnet50,resnet50,resnet50 -b 32,32,32,32 --num_steps 1000,1000,1000,1000 -t inference,training,training,training -log_file $h/4_i.log,$h/4_t1.log,$h/4_t2.log,$h/4_t3.log
python main.py -m resnet50,resnet50,resnet50,resnet50,resnet50 -b 32,32,32,32,32 --num_steps 1000,1000,1000,1000,1000 -t inference,training,training,training,training -log_file $h/5_i.log,$h/5_t1.log,$h/5_t2.log,$h/5_t3.log,$h/5_t4.log


python parse_log.py -l $h/1_i.log -o $h/1_out.log
python parse_log.py -l $h/2_i.log,$h/2_t.log -o $h/2_out.log
python parse_log.py -l $h/3_i.log,$h/3_t1.log,$h/3_t2.log -o $h/3_out.log
python parse_log.py -l $h/4_i.log,$h/4_t1.log,$h/4_t2.log,$h/4_t3.log -o $h/4_out.log
python parse_log.py -l $h/5_i.log,$h/5_t1.log,$h/5_t2.log,$h/5_t3.log,$h/5_t4.log -o $h/5_out.log
