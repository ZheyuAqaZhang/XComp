# 1. bash scripts/baselines/ver4.5_baseline_extreme_true/1.sh
# 2. bash scripts/baselines/ver4.5_baseline_extreme_true/2.sh

max_retries=10

for ((i=1; i<=max_retries; i++)); do
    bash scripts/baselines/verify_data/ab_same_length.sh && break
    echo "Retrying ($i/$max_retries)..."
    ps | grep python | awk '{print $1}' | xargs kill -9
    sleep 300
    if [[ $i -eq $max_retries ]]; then
        echo "failed after $max_retries attempts."
        exit 1
    fi
done
