#!/bin/bash
# Download CAMELYON16 from the public AWS Open Data bucket into $DATA_ROOT and verify MD5s.
# Run on a Fir login node (compute nodes have no internet), inside tmux/screen:
#   DATA_ROOT=/project/def-<PI>/$USER/camelyon16 bash scripts/download_camelyon16.sh
# Resumable: re-run after an interruption; verified files are skipped, partial files are continued.
# Fetches images/ (~701 GiB), annotations/, evaluation/. Set INCLUDE_MASKS=1 to also fetch masks/ (~8.8 GiB).

set -euo pipefail

: "${DATA_ROOT:?set DATA_ROOT to the target directory}"
BASE_URL=${BASE_URL:-https://camelyon-dataset.s3.amazonaws.com/CAMELYON16}
JOBS=${JOBS:-4}
INCLUDE_MASKS=${INCLUDE_MASKS:-0}

mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

for f in checksums.md5 README.md license.txt; do
    curl -fsSL --retry 5 -o "$f" "$BASE_URL/$f"
done

folders="images|annotations|evaluation"
[ "$INCLUDE_MASKS" = 1 ] && folders="$folders|masks"
# checksums.md5 lines look like: "<md5> *images/tumor_001.tif"
sed 's/ \*/  /' checksums.md5 | grep -E "  ($folders)/" > selected.md5
touch verified.txt

fetch() {
    local md5=$1 path=$2
    grep -qxF "$path" verified.txt && return 0
    mkdir -p "$(dirname "$path")"
    # A complete file left from an interrupted run only needs verifying, not re-requesting.
    if ! { [ -f "$path" ] && echo "$md5  $path" | md5sum -c --status; }; then
        curl -fsSL --retry 10 --retry-delay 30 -C - -o "$path" "$BASE_URL/$path" \
            || { echo "FAIL $path (download error; partial file kept for resume)" >&2; return 1; }
    fi
    if echo "$md5  $path" | md5sum -c --status; then
        echo "$path" >> verified.txt
        echo "ok   $path"
    else
        echo "FAIL $path (md5 mismatch; deleting so the next run re-downloads)" >&2
        rm -f "$path"
        return 1
    fi
}
export -f fetch
export BASE_URL

status=0
xargs -P "$JOBS" -L 1 bash -c 'fetch "$0" "$1"' < selected.md5 || status=$?

echo "verified $(sort -u verified.txt | wc -l) / $(wc -l < selected.md5) files"
exit $status
