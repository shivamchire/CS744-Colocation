#!/bin/bash

# Define arrays
echo "starting the script"
outer_model=('dqn')
inner_model=('dqn' 'cyclegan' 'lstm' 'resnet50' 's2s')
outer_job_type=('inference')
inner_job_type=('inference' 'inference' 'inference' 'inference' 'inference' )
outer_batch_size=(32)
inner_batch_size=(1 1 20 32 32)
outer_num_steps=(10000)
inner_num_steps=(10000 5000 5000 5000 5000)
outer_log_file=()
inner_log_file=()

echo "creating log files"
# Create log file arrays
for (( i=0; i<${#outer_model[@]}; i++ )); do
    for (( j=0; j<${#inner_model[@]}; j++ )); do	    
        outer_log_file+=("outer_${outer_model[i]}_${outer_job_type[i]}.log")
        inner_log_file+=("inner_${inner_model[j]}_${inner_job_type[j]}.log")
    done
done

echo "iterating over models"
# Iterate over models
for (( i=0; i<${#outer_model[@]}; i++ )); do
    for (( j=0; j<${#inner_model[@]}; j++ )); do
        echo "*****************"
        echo "Running ${outer_model[i]} ${outer_job_type[i]} and ${inner_model[j]} ${inner_job_type[j]}"
        echo "*****************"
        python3 main.py --MPS 1 -c 1 -m "${outer_model[i]},${inner_model[j]}" -b "${outer_batch_size[i]},${inner_batch_size[j]}" -n "${outer_num_steps[i]},${inner_num_steps[j]}" -log_file "${outer_log_file[i]},${inner_log_file[j]}"  -t "${outer_job_type[i]},${inner_job_type[j]}"
        python3 parse_log.py -l "${outer_model[i]}/${outer_log_file[i]},${inner_model[j]}/${inner_log_file[j]}" --output "outer_${outer_model[i]}_${outer_job_type[i]}_inner_${inner_model[j]}_${inner_job_type[j]}.json"
    done
done

echo "Completed Experiments"
