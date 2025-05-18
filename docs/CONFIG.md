# Configuration & Weight Format

All symbol generation logic is centralised in `simulator/config.py`.  The *only* place where probabilities, multipliers or pay-table entries should be edited is this module.

## 1.  Structure

```text
simulator/config.py
├── SYMBOLS                       # canonical Symbol objects
├── PAYTABLE                      # pay multipliers per cluster size
├── SYMBOL_GENERATION_WEIGHTS_BG  # dict[str, float]
├── SYMBOL_GENERATION_WEIGHTS_FS  # dict[str, float]
├── BG_SYMBOL_NAMES / BG_WEIGHTS  # numpy views for vector maths
├── FS_SYMBOL_NAMES / FS_WEIGHTS  #   ‘‘
└── validate_config()             # runtime assertions
```

### BG vs FS arrays
The numpy arrays are *views* derived from the authoritative dictionaries.  This means that updating a value inside the dict will **not** update the numpy array automatically.  In tooling that mutates weights (e.g. the forthcoming optimizer) you should regenerate the arrays *after* modifying the dicts.

## 2.  Validation rules
`validate_config()` asserts a few invariants at import-time so that CI fails fast:

1. Sum of BG weights > 0
2. Sum of FS weights > 0
3. The symbol sets for BG and FS are identical – the engine assumes equal length vectors for fast vectorised operations.

If any rule fails a `ValueError` is raised during module import, causing the entire test-suite to abort.

## 3.  Persisting a weight snapshot
Use `config.save_weights("weights_YYYYMMDD.json")` to snapshot the current probability tables.  The helper writes a minimal JSON containing BG & FS dicts plus an ISO-8601 timestamp.

```python
from simulator import config
config.save_weights("weights_20250101.json")
```

The optimiser (Epic 4) will be able to *resume* from such a blob via the `--resume` flag. 