#!/usr/bin/env bash
# Download the Criteo Display Advertising Challenge dataset into data/.
#
#   ./download_data.sh sample   # 100k-row official sample (~9 MB)  -> data/dac_sample.txt
#   ./download_data.sh full     # full dataset (~4.6 GB compressed) -> data/train.txt
#
# Source: the original Kaggle/Criteo release (dac.tar.gz), mirrored ungated on
# Hugging Face by the Microsoft Recommenders team. Row order is the original
# temporal order — do not shuffle; the whole protocol depends on it.
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-sample}"
BASE_URL="https://huggingface.co/datasets/Recommenders/criteo/resolve/main"
mkdir -p data

# extract $2 from tar $1, normalize location to data/$2
extract() {
    # GNU tar supports --wildcards/--warning=no-unknown-keyword; macOS bsdtar
    # doesn't, but listing+filtering member names works on both.
    member=$(tar -tzf "$1" | grep "$2\$" | head -1)
    tar -xzf "$1" -C data "$member"
    found=$(find data -name "$2" | head -1)
    if [ "$found" != "data/$2" ]; then
        mv "$found" "data/$2"
    fi
    rm -f "$1"
    echo "OK: data/$2 ($(wc -l < "data/$2") rows)"
}

if [ "$MODE" = "sample" ]; then
    if [ -f data/dac_sample.txt ]; then
        echo "data/dac_sample.txt already exists, skipping download"
        exit 0
    fi
    curl -L --fail -C - -o data/dac_sample.tar.gz "$BASE_URL/dac_sample.tar.gz"
    extract data/dac_sample.tar.gz dac_sample.txt

elif [ "$MODE" = "full" ]; then
    if [ -f data/train.txt ]; then
        echo "data/train.txt already exists, skipping download"
        exit 0
    fi
    curl -L --fail -C - -o data/dac.tar.gz "$BASE_URL/dac.tar.gz"
    extract data/dac.tar.gz train.txt

else
    echo "usage: $0 [sample|full]" >&2
    exit 1
fi
