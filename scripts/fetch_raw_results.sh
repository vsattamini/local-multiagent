#!/usr/bin/env bash
#
# Restore the raw experiment-result directories from the committed archives/
# (.tar.gz). Verifies each archive against archives/MANIFEST.txt, then extracts
# it into the repo root. On a fresh clone the archives are already present (they
# are committed to git), so this needs no network access.
#
# Usage:
#   scripts/fetch_raw_results.sh            # verify + extract all
#   scripts/fetch_raw_results.sh <name>     # just one, e.g. results_phase3
#
set -euo pipefail

cd "$(dirname "$0")/.."

if [ $# -ge 1 ]; then
  targets=("archives/${1}.tar.gz")
else
  targets=(archives/*.tar.gz)
fi

sha_tool() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }

for f in "${targets[@]}"; do
  [ -f "$f" ] || { echo "missing: $f" >&2; exit 1; }
  base=$(basename "$f")
  if [ -f archives/MANIFEST.txt ]; then
    want=$(awk -F'\t' -v b="$base" '$1==b {print $3}' archives/MANIFEST.txt)
    got=$(sha_tool "$f")
    if [ -n "$want" ] && [ "$want" != "$got" ]; then
      echo "CHECKSUM MISMATCH for $base (want $want, got $got)" >&2
      exit 1
    fi
  fi
  echo "==> extracting $f"
  tar -xzf "$f"
done

echo "done."
