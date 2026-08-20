#!/usr/bin/env bash
#
# Compress the large raw experiment-result directories into per-dir .tar.gz
# archives under archives/, and regenerate archives/MANIFEST.txt with sizes +
# sha256 sums.
#
# The extracted result dirs are gitignored; the compressed archives/ are
# committed to git and are the source of truth (see RAW_DATA.md). Re-run this
# whenever the raw result dirs change, then commit the updated archives.
#
# Usage:
#   scripts/archive_raw_results.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."

DIRS=(
  results_multiseed
  results_phase3
  results_humaneval_agentic
  results_terminalbench
  results_context_shuffle
)

mkdir -p archives
MANIFEST="archives/MANIFEST.txt"
: > "$MANIFEST"

sha_tool() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }

echo "# Raw results archives" >> "$MANIFEST"
echo "# archive<TAB>size<TAB>sha256<TAB>source_dir" >> "$MANIFEST"

for d in "${DIRS[@]}"; do
  if [ ! -d "$d" ]; then
    echo "skip (missing): $d"
    continue
  fi
  out="archives/${d}.tar.gz"
  echo "==> archiving $d -> $out"
  tar -czf "$out" "$d"
  size=$(du -h "$out" | awk '{print $1}')
  sum=$(sha_tool "$out")
  printf '%s\t%s\t%s\t%s\n' "$(basename "$out")" "$size" "$sum" "$d" >> "$MANIFEST"
  echo "    $size  $sum"
done

echo
echo "Manifest written to $MANIFEST:"
cat "$MANIFEST"
echo
echo "Now commit the updated archives/ if they changed."
