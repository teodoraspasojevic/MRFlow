#!/usr/bin/env bash
# Retry wrapper for `forithmus submit` -- the ~5.3GB image upload to GCS has shown recurring
# transient "Connection error: The read operation timed out" failures on this host (2 of 3 tries
# for the mod1-rep1 baseline, then all 3 tries for mod10-rep10 in the cfg-sweep batch), unrelated
# to the submission's contents. Retries with backoff; a real (non-transient) failure surfaces on
# the last attempt's exit code.
set -uo pipefail

TARBALL="${1:?usage: submit_with_retry.sh <tarball> <reuse_weights_id> <desc>}"
REUSE_WEIGHTS="${2:?usage: submit_with_retry.sh <tarball> <reuse_weights_id> <desc>}"
DESC="${3:?usage: submit_with_retry.sh <tarball> <reuse_weights_id> <desc>}"
PHASE="0edb5ef1-2030-4556-ab59-b26f94d3646a"
TIER="gpu-a100-80"
BUDGET=320
MAX_ATTEMPTS=6

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    echo "=== attempt $attempt/$MAX_ATTEMPTS: $(basename "$TARBALL") ==="
    if forithmus submit "$TARBALL" \
        --reuse-weights "$REUSE_WEIGHTS" \
        --phase "$PHASE" \
        --tier "$TIER" \
        --time-budget "$BUDGET" \
        --desc "$DESC"; then
        echo "=== SUCCESS: $(basename "$TARBALL") ==="
        exit 0
    fi
    backoff=$((attempt * 20))
    echo "=== attempt $attempt failed, retrying in ${backoff}s ==="
    sleep "$backoff"
done
echo "=== FAILED after $MAX_ATTEMPTS attempts: $(basename "$TARBALL") ==="
exit 1
