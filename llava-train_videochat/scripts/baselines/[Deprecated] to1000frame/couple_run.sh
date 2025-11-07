# 1. bash scripts/baselines/ver4.5_baseline_extreme_true/1.sh
# 2. bash scripts/baselines/ver4.5_baseline_extreme_true/2.sh

max_retries=10

# Execute 1.sh with retries
for ((i=1; i<=max_retries; i++)); do
    bash scripts/baselines/to1000frame/2.sh && break
    echo "Retrying 1.sh ($i/$max_retries)..."
    if [[ $i -eq $max_retries ]]; then
        echo "1.sh failed after $max_retries attempts."
        exit 1
    fi
done
