#!/usr/bin/env bash
# Convert a locally-built Docker image into the classic (non-OCI) tarball format Forithmus's
# validator requires, and verify the result before spending an upload on a rejected image.
#
# Adapted from R2V-MR-Generation/submission/helpers/package_image.sh: `docker save` on this host
# (Docker 29.x, containerd-backed buildkit) produces an OCI-format tarball by default, which the
# validator rejects with "Image config blobs/sha256/<hash>... not found in tarball" -- skopeo does
# the OCI -> classic conversion. Ported version used `skopeo copy docker-daemon:...` to pull the
# image straight from the daemon, but skopeo vendors its own (older) Docker API client here and
# that transport fails outright: "client version 1.41 is too old. Minimum supported API version is
# 1.44" -- unrelated to the `docker` CLI itself, which negotiates fine (confirmed: `docker build`/
# `images`/`run` all work throughout this session). Fixed by getting the OCI tar via `docker save`
# (the real docker CLI/daemon protocol) first, then handing skopeo a **file** (`oci-archive:`) to
# convert -- no daemon RPC in that step, so skopeo's stale client version never comes up.
#
#     ./submission/package_image.sh mrflow-vlm3d-challenge:latest
#     ./submission/package_image.sh mrflow-vlm3d-challenge:latest mrflow_submission.tar.gz
#
# Needs ~2x the image size in scratch space (an uncompressed OCI tar + an uncompressed classic
# tar, briefly side by side) -- confirmed the hard way: this host's `/` has only ~50GB free, `mktemp`
# defaults there, and 9 of these run at once for a cfg sweep. PKG_TMPDIR below defaults to `/data`
# (Docker's own storage disk, terabytes free) instead of `/tmp`; override it if `/data` isn't
# right on your host, but never leave it on `/` for a multi-image sweep.
#
# Redirecting where the .tar *files* land isn't enough on its own -- confirmed the hard way, twice:
# skopeo (via containers/image) unpacks the OCI archive into its *own* scratch directory first, and
# that directory ignores $TMPDIR entirely -- it comes from `SystemContext.BigFilesTemporaryDir`,
# which the CLI only exposes as the global `--tmpdir` flag (`skopeo --help`), not an env var. Take
# one (redirecting mktemp) and take two (exporting TMPDIR) both left it defaulting to /var/tmp,
# which filled `/` to 100% under 9 concurrent conversions and had to be killed mid-run. `--tmpdir`
# below is the flag that actually reaches that code path -- confirmed against skopeo 1.13.3's
# `--help` output, not guessed.
set -euo pipefail

IMAGE="${1:?usage: package_image.sh <image:tag> [out.tar.gz]}"
OUT="${2:-mrflow_submission.tar.gz}"
PKG_TMPDIR="${PKG_TMPDIR:-/data}"

command -v skopeo >/dev/null 2>&1 || {
    echo "FATAL: skopeo not found. Install it first: sudo apt-get install -y skopeo" >&2
    exit 1
}
[ -d "$PKG_TMPDIR" ] && [ -w "$PKG_TMPDIR" ] || {
    echo "FATAL: PKG_TMPDIR=$PKG_TMPDIR doesn't exist or isn't writable" >&2
    exit 1
}
export TMPDIR="$PKG_TMPDIR"

OCI_TAR="$(mktemp --tmpdir="$PKG_TMPDIR" --suffix=.oci.tar)"
TMP_TAR="$(mktemp --tmpdir="$PKG_TMPDIR" --suffix=.tar)"
trap 'rm -f "$OCI_TAR" "$TMP_TAR"' EXIT

echo "docker save $IMAGE -> $OCI_TAR (OCI format) ..."
docker save "$IMAGE" -o "$OCI_TAR"

echo "converting $OCI_TAR (OCI, file-only -- no daemon RPC) -> classic docker-archive ..."
skopeo --tmpdir="$PKG_TMPDIR" copy "oci-archive:${OCI_TAR}" "docker-archive:${TMP_TAR}:${IMAGE}"

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
