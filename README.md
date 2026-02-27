# SafetyClassify

Single-turn OpenReward environment for classifying molecular safety across multiple toxicity endpoints from SMILES notation.

## Task

Given a molecule's SMILES string and a safety endpoint, the agent predicts whether the molecule is positive (unsafe/active) or negative (safe/inactive) for that endpoint. One tool call per task.

## Data Source

All data comes from [Therapeutics Data Commons (TDC)](https://tdcommons.ai/single_pred_tasks/tox/), pooling 3 safety classification datasets:

| Dataset | Property | Classes | Molecules | Source |
|---------|----------|---------|-----------|--------|
| AMES | Mutagenicity | 0 = non-mutagenic, 1 = mutagenic | 7,255 | Hansen et al. |
| hERG_Karim | hERG Cardiotoxicity | 0 = non-blocker, 1 = blocker | 13,445 | Karim et al. |
| ClinTox | Clinical Trial Toxicity | 0 = non-toxic, 1 = toxic | 1,484 | Gayvert et al. |

Total pool: ~22,000 molecules. 1,100 sampled (1,000 train + 100 test), shuffled with `random_state=42`.

### Property Distribution in Splits

**Train (1,000 tasks):** hERG Cardiotoxicity (609), AMES Mutagenicity (334), Clinical Trial Toxicity (57).

**Test (100 tasks):** hERG Cardiotoxicity (56), AMES Mutagenicity (34), Clinical Trial Toxicity (10).

Distribution is proportional to dataset size. Overall class balance is ~48.5% positive.

## Reward Function

Binary reward:

```
reward = 1.0 if predicted == actual else 0.0
```

## Environment API

- **Splits:** `train` (1,000 tasks), `test` (100 tasks)
- **Tool:** `submit_prediction(prediction: int)` -- submit 0 (safe) or 1 (unsafe)
- **Prompt:** Provides SMILES string, safety endpoint name, and class label descriptions
- **Finished:** Always `True` after one tool call (single-turn)

## Files

```
safetyclassify/
├── safetyclassify.py   # Environment class (SafetyClassify)
├── server.py           # Server wrapper
├── test_agent.py       # OpenAI Responses API test harness
├── prepare_data.py     # TDC download + JSON generation script
├── requirements.txt    # openreward, pydantic
├── Dockerfile
├── DATA_UPLOAD.md      # Cloud storage upload instructions
└── data/
    ├── train.json      # 1,000 training tasks
    └── test.json       # 100 test tasks
```

## Local Development

```bash
# Generate data (requires PyTDC)
pip install PyTDC pandas
python prepare_data.py

# Run server
pip install -r requirements.txt
python server.py

# Test with agent
export OPENAI_API_KEY=...
python test_agent.py
```

## Docker

```bash
docker build -t safetyclassify:test .
docker run -p 8080:8080 safetyclassify:test
```
