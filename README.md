# MolecularSafety

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/MolecularSafety)

## Description

**MolecularSafety** is an environment for evaluating agents on molecular safety classification tasks. Given a molecule's SMILES string and a safety endpoint, the agent predicts whether the molecule is safe or unsafe. The dataset pools three toxicity classification datasets from [Therapeutics Data Commons (TDC)](https://tdcommons.ai/single_pred_tasks/tox/): AMES mutagenicity, hERG cardiotoxicity, and ClinTox clinical trial toxicity.

## Capabilities

- Classifying molecular safety across multiple toxicity endpoints
- Predicting AMES mutagenicity from molecular structure
- Predicting hERG cardiotoxicity (potassium channel blocking)
- Predicting clinical trial toxicity

## Compute Requirements

MolecularSafety does not require a sandbox. It has minimal compute requirements.

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (following the TDC dataset licenses).

## Tasks

There are two splits: train (1,000 tasks) and test (100 tasks), totaling 1,100 tasks. Tasks are pooled from three TDC safety datasets (~22,000 molecules total) with proportional sampling:

| Dataset | Property | Classes | Train | Test |
|---------|----------|---------|-------|------|
| [AMES](https://tdcommons.ai/single_pred_tasks/tox/#ames-mutagenicity) | Mutagenicity | 0 = non-mutagenic, 1 = mutagenic | 334 | 34 |
| [hERG_Karim](https://tdcommons.ai/single_pred_tasks/tox/#herg-karim-et-al) | hERG Cardiotoxicity | 0 = non-blocker, 1 = blocker | 609 | 56 |
| [ClinTox](https://tdcommons.ai/single_pred_tasks/tox/#clintox) | Clinical Trial Toxicity | 0 = non-toxic, 1 = toxic | 57 | 10 |

Overall class balance is ~48.5% positive (unsafe).

## Reward Structure

This is a sparse, verifiable reward environment with binary scoring. The agent calls `submit_prediction` once with a classification (0 = safe, 1 = unsafe).

- **Correct**: Reward **1.0**.
- **Incorrect**: Reward **0.0**.

We do not use LLM graders for this task.

## Data

Task data is pooled from three [TDC](https://tdcommons.ai/single_pred_tasks/tox/) toxicity datasets: AMES (Hansen et al.), hERG_Karim (Karim et al.), and ClinTox (Gayvert et al.). Data files are stored on the OpenReward platform.

## Tools

Agents are given a single tool:

- `submit_prediction`: Submit a safety classification (0 = safe/negative, 1 = unsafe/positive). Returns whether the prediction is correct. This tool can only be called once per task.

## Time Horizon

MolecularSafety is a single-turn environment. The agent receives a molecule's SMILES string and safety endpoint, and submits one classification. Each task requires exactly one tool call.

## Environment Difficulty

[Statistics on environment difficulty here]

## Other Environment Requirements

There are no further environment requirements; MolecularSafety works out of the box with the OpenReward endpoint.

## Safety

Agents in MolecularSafety are asked to classify molecules for toxicity across multiple safety endpoints. The environment does not present direct safety risks, as agents only provide classification predictions with no access to external systems.

However, this is a dual-use domain. Models trained for toxicity prediction capabilities could potentially be misused for designing harmful compounds in other contexts.

## Citations

```bibtex
@dataset{GRMolecularSafety,
  author    = {General Reasoning Inc. Team},
  title     = {MolecularSafety},
  year      = {2026},
  publisher = {OpenReward},
  url       = {https://openreward.ai/GeneralReasoning/MolecularSafety}
}

@article{huang2021therapeutics,
  title={Therapeutics Data Commons: Machine learning datasets and tasks for drug discovery and development},
  author={Huang, Kexin and Fu, Tianfan and Gao, Wenhao and Zhao, Yue and Roohani, Yusuf and Leskovec, Jure and Coley, Connor W and Xiao, Cao and Sun, Jimeng and Zitnik, Marinka},
  journal={Proceedings of NeurIPS Datasets and Benchmarks},
  year={2021}
}
```
