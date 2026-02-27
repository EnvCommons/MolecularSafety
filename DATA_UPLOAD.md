# Data Upload Requirements for SafetyClassify

## Overview
This environment requires molecular safety classification data uploaded to OpenReward cloud storage.

## Directory Structure
```
/orwd_data/
└── data/
    ├── train.json (1000 tasks, ~400 KB)
    └── test.json (100 tasks, ~40 KB)
```

## Files Required
- **train.json**: 1000 safety classification tasks pooled from 3 TDC datasets (AMES, hERG_Karim, ClinTox)
- **test.json**: 100 safety classification tasks (same distribution)

## Data Generation
Run locally: `python prepare_data.py` (requires `pip install PyTDC pandas`)

## Upload Instructions
Upload the `data/` directory to your OpenReward namespace at https://openreward.ai.
