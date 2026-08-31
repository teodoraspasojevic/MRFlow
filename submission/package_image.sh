#!/usr/bin/env bash
# Convert a locally-built Docker image into the classic (non-OCI) tarball format Forithmus's
# validator requires, and verify the result before spending an upload on a rejected image.
#
# Ported verbatim from R2V-MR-Generation/submission/helpers/package_image.sh (the conversion is
# generic -- `docker save` on this host produces an OCI-format tarball by default, which the
# validator rejects with "Image config blobs/sha256/<hash>... not found in tarball").
#
#     ./submission/package_image.sh mrflow-vlm3d-challenge:latest
#     ./submission/package_image.sh mrflow-vlm3d-challenge:latest mrflow_submission.tar.gz
set -euo pipefail

IMAGE="${1:?usage: package_image.sh <image:tag> [out.tar.gz]}"
OUT="${2:-mrflow_submission.tar.gz}"

command -v skopeo >/dev/null 2>&1 || {
    echo "FATAL: skopeo not found. Install it first: sudo apt-get install -y skopeo" >&2
    exit 1
}

TMP_TAR="$(mktemp --suffix=.tar)"
trap 'rm -f "$TMP_TAR"' EXIT

echo "converting $IMAGE (docker-daemon, likely OCI format) -> classic docker-archive ..."
skopeo copy "docker-daemon:${IMAGE}" "docker-archive:${TMP_TAR}"

echo "verifying classic format (no index.json / oci-layout) ..."
if tar -tf "$TMP_TAR" | grep -qE '(^|/)(index\.json|oci-layout)$'; then
    echo "FATAL: $TMP_TAR still has OCI markers (index.json / oci-layout) -- conversion did not take" >&2
    exit 1
fi
CONFIG_ENTRY=$(tar -xf "$TMP_TAR" manifest.json -O | python3 -c 'import json, sys; print(json.load(sys.stdin)[0]["Config"])')
case "$CONFIG_ENTRY" in
    *.json) echo "OK: classic format (Config=$CONFIG_ENTRY)" ;;
    *)      echo "FATAL: Config entry '$CONFIG_ENTRY' is not <hash>.json -- not classic format" >&2
            exit 1 ;;
esac

echo "compressing -> $OUT"
gzip -c "$TMP_TAR" > "$OUT"
echo
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
