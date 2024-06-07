cd ../

MAX_JOBS=1
SEED=(1 2 3 4 5)
DVIECE_ID=0

PYTHON_INTERPRETER="/home/wuk9/miniconda3/envs/poison/bin/python3"
SCRIPT_PATH="/home/wuk9/poison_attack/coarsen_FGA.py"

running_jobs=0

for seed in "${SEED[@]}"; do
    for method in "variation_neighborhoods_degree" "variation_neighborhoods" "variation_edges_degree"; do
        for dataset in "Cora" "Pubmed" "Flickr" "Polblogs"; do
            nohup $PYTHON_INTERPRETER $SCRIPT_PATH --device_id $DVIECE_ID --seed $seed --dataset $dataset --coarsening_method $method 2>&1 &
            ((running_jobs++))
            if [[ $running_jobs -eq $MAX_JOBS ]]; then
                wait
                ((running_jobs--))
            fi
        done
    done
done

wait
rm nohup.out
echo "All scripts are done."