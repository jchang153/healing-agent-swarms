#!/bin/sh
# Rebuild CheatBench task data into data/cheatbench/<task>/ from the pinned commit in
# vendor/cheatbench/SOURCE_COMMIT, using CheatBench's own offline builder.
# Usage: vendor/cheatbench/fetch.sh [prime_factorization|subset_sum]   (default: prime_factorization)
set -e
cd "$(dirname "$0")/../.."
TASK=${1:-prime_factorization}
COMMIT=$(cat vendor/cheatbench/SOURCE_COMMIT)
TMP=$(mktemp -d)
git clone -q https://github.com/centerforaisafety/cheatbench "$TMP"
git -C "$TMP" checkout -q "$COMMIT"
python3 "$TMP/tasks/$TASK/build.py"
mkdir -p "data/cheatbench/$TASK"
rm -rf "data/cheatbench/$TASK"/*
cp -R "$TMP/tasks/$TASK/data.jsonl" "$TMP/tasks/$TASK/environment" "data/cheatbench/$TASK/"
rm -rf "$TMP"
echo "ok: data/cheatbench/$TASK"
