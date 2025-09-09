#!/usr/bin/env bash

if [[ "$#" -ne 1 ]]; then
    echo "Usage: source load_env.sh <env_file>"
    return 1
fi
ENV_FILE="$1"
if [ ! -f "$ENV_FILE" ]; then
    echo "Environment file $ENV_FILE does not exist."
    return 1
fi
# read env file and export variables inside it

# Read the file and export each variable
while IFS='=' read -r key value; do
    # Skip empty lines and comments
    [[ -z "$key" || "$key" =~ ^# ]] && continue

    # Export the variable
    export "$key=$value"
    echo "export $key=$value"
done <"$ENV_FILE"
