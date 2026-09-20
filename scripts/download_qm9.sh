#!/usr/bin/env bash
# Download and unpack QM9 (dsgdb9nsd): 134k CHONF molecules with optimized
# geometries and B3LYP/6-31G(2df,p) harmonic frequencies.
# Ramakrishnan et al., Sci. Data 1, 140022 (2014).
#
# Produces qm9_data/dsgdb9nsd/dsgdb9nsd/dsgdb9nsd_*.xyz
#
# Figshare answers the first request to a large file with HTTP 202
# ("preparing") and a zero-length body, so we poll until it returns a real,
# bzip2-valid archive. If your network cannot reach Figshare, download the
# archive by hand (a browser works) and place it at
#   qm9_data/dsgdb9nsd.xyz.tar.bz2
# then re-run this script; it will skip downloading and just verify + unpack.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$HERE/../qm9_data"
mkdir -p "$DATA"
cd "$DATA"

ARCH=dsgdb9nsd.xyz.tar.bz2
URLS=(
  "https://springernature.figshare.com/ndownloader/files/3195389"
  "https://figshare.com/ndownloader/files/3195389"
)
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

valid_archive() { [ -s "$1" ] && bzip2 -t "$1" 2>/dev/null; }

if ! valid_archive "$ARCH"; then
  ok=0
  for URL in "${URLS[@]}"; do
    echo "Downloading QM9 (~82 MB) from $URL"
    echo "(Figshare may reply 202 while preparing; polling, up to ~5 min) ..."
    for i in $(seq 1 30); do
      code=$(curl -sL -A "$UA" -w '%{http_code}' -o "$ARCH.tmp" "$URL" || echo 000)
      sz=$(stat -c%s "$ARCH.tmp" 2>/dev/null || echo 0)
      if [ "$code" = "200" ] && [ "$sz" -gt 1000000 ] && bzip2 -t "$ARCH.tmp" 2>/dev/null; then
        mv "$ARCH.tmp" "$ARCH"; ok=1; break
      fi
      echo "  attempt $i: HTTP $code, $sz bytes; retry in 10 s ..."
      rm -f "$ARCH.tmp"; sleep 10
    done
    [ "$ok" = 1 ] && break
  done
  if [ "$ok" != 1 ]; then
    echo "ERROR: could not fetch a valid archive from Figshare." >&2
    echo "Download $ARCH manually from ${URLS[0]} and place it in:" >&2
    echo "  $DATA" >&2
    echo "then re-run: bash scripts/download_qm9.sh" >&2
    exit 1
  fi
fi

echo "Archive OK ($(du -h "$ARCH" | cut -f1)). Extracting ..."
tar xjf "$ARCH"

N=$(find . -name 'dsgdb9nsd_*.xyz' | wc -l)
echo "Found $N .xyz files. Samples:"
find . -name 'dsgdb9nsd_*.xyz' | sort | head -3
if [ "$N" -lt 100000 ]; then
  echo "WARNING: expected ~133,885 xyz files but found $N." >&2
  exit 1
fi
echo "Next: python scripts/build_records_aug.py"
