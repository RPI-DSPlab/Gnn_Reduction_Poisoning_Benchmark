cd ../

MAX_JOBS=5
SEED=(1 2 3 4 5)
DVIECE_ID=1

running_jobs=0

for seed in "${SEED[@]}"; do
    for method in "variation_neighborhoods_degree" "variation_neighborhoods" "variation_edges_degree"; do
        for dataset in "Cora" "Pubmed" "Flickr" "Polblogs"; do
            nohup python3 coarsen_FGA.py --device_id $DVIECE_ID --seed $seed --dataset $dataset --coarsening_method $method & > ./log/log_${dataset}_${method}.txt 2>&1 &

            ((running_jobs++))
            if [[ $running_jobs -eq $MAX_JOBS ]]; then
                wait
                ((running_jobs--))
            fi
        done
    done
done

wait
echo "All scripts are done."