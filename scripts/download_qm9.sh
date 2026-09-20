#!/usr/bin/env bash
# Download and unpack QM9 (dsgdb9nsd): 134k CHONF molecules with optimized
# geometries and B3LYP/6-31G(2df,p) harmonic frequencies.
# Ramakrishnan et al., Sci. Data 1, 140022 (2014).
#
# Produces repo/qm9_data/dsgdb9nsd/dsgdb9nsd/dsgdb9nsd_*.xyz
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$HERE/../qm9_data"
mkdir -p "$DATA"
cd "$DATA"

URL="https://springernature.figshare.com/ndownloader/files/3195389"

if [ ! -f dsgdb9nsd.xyz.tar.bz2 ]; then
  echo "Downloading QM9 (~82 MB) from $URL ..."
  curl -L --retry 3 "$URL" -o dsgdb9nsd.xyz.tar.bz2
fi

echo "Extracting ..."
tar xjf dsgdb9nsd.xyz.tar.bz2

echo "Done. Sample xyz files:"
find . -name 'dsgdb9nsd_*.xyz' | head -3
echo "Next: python scripts/build_records_aug.py"
