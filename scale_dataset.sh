#!/usr/bin/env bash

# Configuration
CSV_FILE="scale_dataset.fake.csv" # Path to your CSV file
FINETUNE_CMD="./tools/finetune.py"

# Dataset flags as requested
DATASET_LIST=(
    "SMALL_DATASET"
    "MEDIUM_DATASET"
)

# Check if CSV file exists
if [[ ! -f "$CSV_FILE" ]]; then
    echo "Error: CSV file '$CSV_FILE' not found."
    exit 1
fi
if [ -z "$POLYBENCH_DIR" ]; then
    echo "Please set POLYBENCH_DIR environment variable to point to the PolyBench directory."
    exit 1
fi
if [ ! -d "$POLYBENCH_DIR" ]; then
    echo "POLYBENCH_DIR ($POLYBENCH_DIR) is not a valid directory."
    exit 1
fi
if [ -z "$PLUTO_BIN" ]; then
    echo "Please set PLUTO_BIN environment variable to point to the Pluto binary."
    exit 1
fi
if [ ! -x "$PLUTO_BIN" ]; then
    echo "PLUTO_BIN ($PLUTO_BIN) is not a valid executable."
    exit 1
fi
if [ -z "$CC" ]; then
    echo "CC is not set, default is gcc"
    CC="gcc"
fi
if [ -z "$ENV_FILE" ]; then
    echo "ENV_FILE is not set, default env/omp32.env"
    ENV_FILE="./env/omp32.env"
fi

PLUTO_FLAGS="--tile --parallel --diamond-tile --nounroll --prevector"
PLUTO_VEC_PRAGMA="#pragma GCC ivdep"
CFLAGS="-march=native -O3 -fopenmp"

printf "benchmark,source,npar,p0,p1,p2,p3"
for dataset in "${DATASET_LIST[@]}"; do
    printf ",%s" "$dataset"
done
printf "\n"

tail -n +2 "$CSV_FILE" | while IFS=',' read -r benchmark source npar p0 p1 p2 p3; do

    # Trim whitespace from variables (important for CSV parsing)
    benchmark=$(echo "$benchmark" | xargs)
    source=$(echo "$source" | xargs)
    npar=$(echo "$npar" | xargs)
    p0=$(echo "$p0" | xargs)
    p1=$(echo "$p1" | xargs)
    p2=$(echo "$p2" | xargs)
    p3=$(echo "$p3" | xargs)

    schedule="unknown"
    if [[ "$source" == *"static"* ]]; then
        schedule="static"
    elif [[ "$source" == *"dynamic"* ]]; then
        schedule="dynamic"
    fi
    is_pluto=false
    if [[ "$source" == *"pluto"* ]]; then
        is_pluto=true
    fi

    # Skip empty lines or malformed rows
    if [[ -z "$benchmark" || -z "$source" ]]; then
        continue
    fi

    src_file="${POLYBENCH_DIR}/${benchmark}/${source}"
    if [[ "$is_pluto" == true ]]; then
        src_file="${POLYBENCH_DIR}/${benchmark}/$(basename "$benchmark").c"
    fi

    # echo "Processing: $benchmark ($source) with npar=$npar"

    printf "%s,%s,%s,%s,%s,%s,%s" "$benchmark" "$source" "$npar" "$p0" "$p1" "$p2" "$p3"

    # Loop through each dataset flag
    for dataset in "${DATASET_LIST[@]}"; do

        # Construct the arguments for the finetune script
        # We assume the finetune script accepts:
        # 1. The source file
        # 2. The npar value
        # 3. The tile parameters (P0, P1, P2, P3)
        # 4. The dataset flag

        # Build the list of tile parameters, ignoring empty ones
        tile_params=()
        if [[ -n "$p0" ]]; then tile_params+=("$p0"); fi
        if [[ -n "$p1" ]]; then tile_params+=("$p1"); fi
        if [[ -n "$p2" ]]; then tile_params+=("$p2"); fi
        if [[ -n "$p3" ]]; then tile_params+=("$p3"); fi

        # Create a unique identifier for logging
        log_file="${benchmark//\//_}_${dataset}.log"

        EXTRA_FLAGS="-lm -D${dataset} -DPOLYBENCH_TIME"

        # Construct the command
        # Example: ./finetune.sh covariance.pluto.static.c 3 16 64 16 -DMEDIUM_DATASET
        cmd=(
            python3
            "$FINETUNE_CMD"
            "${POLYBENCH_DIR}/utilities/polybench.c"
            "${src_file}"
            -I "${POLYBENCH_DIR}/utilities"
            --log-file "${log_file}"
            --compiler-bin "${CC}"
            --compiler-cflags="${CFLAGS}"
            --env "$ENV_FILE"
            --compiler-extra-flags="${EXTRA_FLAGS}"
            --timeout 5
            --perf-nrun 40
            --perf-nmedianrun 20
        )
        if [[ "$is_pluto" == true ]]; then
            cmd+=(
                --pluto "${PLUTO_BIN}"
                --pluto-flags="${PLUTO_FLAGS}"
                --pluto-custom-vec-pragma="${PLUTO_VEC_PRAGMA}"
            )
        fi
        if [[ "$schedule" != "unknown" ]]; then
            cmd+=(--force-omp-schedule "$schedule")
        fi

        i=0
        for param in "${tile_params[@]}"; do
            cmd+=("--param")
            if [[ "$is_pluto" == true ]]; then
                cmd+=("T$i")
            else
                cmd+=("DIV$i")
            fi
            cmd+=("{$param}")
            ((i++))
        done

        # Print to console
        # echo "${cmd[*]}"

        # Uncomment the next line to actually execute the commands
        "${cmd[@]}" >/dev/null 2>&1

        TRIMMED_MEAN=$(awk '
            /Best run \(trimmed mean\):/ {
                found = 1
                next
            }
            found == 1 {
                # Skip the header line (the one with "Trimmed Mean" in the text)
                if ($0 ~ /Trimmed Mean/) {
                    next
                }
                # Now print the last field ($NF) which is the actual value
                print $NF
                found = 0
            }
        ' "$log_file")

        rm -f "${log_file}"

        printf ",%s" "$TRIMMED_MEAN"

    done

    printf "\n"
done

echo "Done. All commands logged to $OUTPUT_LOG."
