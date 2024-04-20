#!/bin/bash

# Define arrays
model=('cyclegan' 'lstm' 'resnet50' 's2s')
outer_job_type=('training' 'training' 'training' 'training' )
inner_job_type=('inference' 'inference' 'inference' 'inference' )
batch_size=(1 20 32 32 )
num_steps=(50 50 50 50 )
outer_log_file=()
inner_log_file=()

# Create log file arrays
for (( i=0; i<${#model[@]}; i++ )); do
    outer_log_file+=("outer_${model[i]}_${outer_job_type[i]}.log")
    inner_log_file+=("inner_${model[i]}_${inner_job_type[i]}.log")
done

# Iterate over models
for (( i=0; i<${#model[@]}; i++ )); do
    for (( j=0; j<${#model[@]}; j++ )); do
        echo "*****************"
        echo "Running ${model[i]} ${outer_job_type[i]} and ${model[j]} ${inner_job_type[j]}"
        echo "*****************"
        python3 main.py --MPS 1 -c 1 -m "${model[i]},${model[j]}" -b "${batch_size[i]},${batch_size[j]}" -n "${num_steps[i]},${num_steps[j]}" -log_file "${outer_log_file[i]},${inner_log_file[j]}"  -t "${outer_job_type[i]},${inner_job_type[j]}"
        python3 parse_log.py -l "${model[i]}/${outer_log_file[i]},${model[j]}/${inner_log_file[j]}" --output "outer_${model[i]}_${outer_job_type[i]}_inner_${model[j]}_${inner_job_type[j]}.json"
    done
done