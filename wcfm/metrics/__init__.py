"""Online metrics: collectors with a declared cadence, append-only JSONL, and no collectives
inside `compute()`, so a data-dependent branch cannot hang a job."""
