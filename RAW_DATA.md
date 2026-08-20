# Raw experiment data

The raw per-seed experiment outputs (the extracted `results_*` directories) are
**gitignored**. Instead, they are committed as compressed `.tar.gz` archives
under `archives/`, which are the source of truth.

Directories treated as raw data:

| Directory | Uncompressed | Contents |
|---|---|---|
| `results_multiseed/` | ~311M | Multi-seed sweep outputs (JSON/JSONL per seed) + `.log` |
| `results_phase3/` | ~484M | Phase-3 population-N sweep outputs |
| `results_humaneval_agentic/` | ~24M | HumanEval agentic runs |
| `results_terminalbench/` | ~164K | Terminal-bench runs |
| `results_context_shuffle/` | ~12K | Context-shuffle probe |

Compressed, all of the above total roughly **~31M** (text compresses ~25×).

The tracked `results/` directory (analysis summaries such as
`results/glmm_analysis.json`) **is** committed — it is not part of the raw dump.

## Storage location

Committed to git under `archives/` (one `.tar.gz` per directory above).
Per-archive sizes and `sha256` checksums are recorded in
`archives/MANIFEST.txt`.

## Workflow

Rebuild the archives (+ manifest) after the raw result dirs change, then commit
the updated `archives/`:

```bash
scripts/archive_raw_results.sh
```

Restore the extracted dirs on a fresh clone (verifies checksums, extracts —
no network needed, archives are already in the repo):

```bash
scripts/fetch_raw_results.sh                  # everything
scripts/fetch_raw_results.sh results_phase3   # just one dir
```
