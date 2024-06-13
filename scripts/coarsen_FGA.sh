cd ../

MAX_JOBS=3
SEED=1
DVIECE_ID=0

PYTHON_INTERPRETER="/home/wuk9/miniconda3/envs/poison/bin/python3"
SCRIPT_PATH="/home/wuk9/poison_attack/coarsen_FGA.py"

running_jobs=0


for dataset in "Cora" "Pubmed"; do
    nohup $PYTHON_INTERPRETER $SCRIPT_PATH --device_id $DVIECE_ID --seed $SEED --dataset $dataset --technique sparsification 2>&1 &
    ((running_jobs++))
done

wait
rm nohup.out
echo "All scripts are done."