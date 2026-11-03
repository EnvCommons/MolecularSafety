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

[MIT](https://opensource.org/licenses/MIT). The underlying TDC datasets are [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Tasks

There are 1,000 tasks in the `train` split and 100 in `test`. Tasks are pooled from three TDC safety datasets (~22,000 molecules total) with proportional sampling:

| Dataset | Property | Classes | Train | Test |
|---------|----------|---------|-------|------|
| [AMES](https://tdcommons.ai/single_pred_tasks/tox/#ames-mutagenicity) | Mutagenicity | 0 = non-mutagenic, 1 = mutagenic | 334 | 34 |
| [hERG_Karim](https://tdcommons.ai/single_pred_tasks/tox/#herg-karim-et-al) | hERG Cardiotoxicity | 0 = non-blocker, 1 = blocker | 609 | 56 |
| [ClinTox](https://tdcommons.ai/single_pred_tasks/tox/#clintox) | Clinical Trial Toxicity | 0 = non-toxic, 1 = toxic | 57 | 10 |

Overall class balance is ~48.5% positive (unsafe).

## Reward Structure

This is a sparse, verifiable reward environment with binary scoring. The agent replies with an ordinary message stating a digit (0 = safe, 1 = unsafe); the environment extracts the last standalone 0 or 1 in the message and grades it by exact match against the ground truth. A message with no such digit scores 0.

- **Correct**: Reward **1.0**.
- **Incorrect**: Reward **0.0**.

We do not use LLM graders for this task.

## Data

Task data is pooled from three [TDC](https://tdcommons.ai/single_pred_tasks/tox/) toxicity datasets: AMES (Xu et al.), hERG_Karim (Karim et al.), and ClinTox (Gayvert et al.). Data files are stored on the OpenReward platform.

## Tools

Agents are given no visible tools. The environment uses a hidden terminal tool (`submit_prediction`): the agent's final plain-text message ends the rollout and is graded by extracting the classification digit (0 or 1).

## Time Horizon

MolecularSafety is a single-turn environment. The agent receives a molecule's SMILES string and safety endpoint, and replies with a single message stating its classification digit.

## Other Environment Requirements

There are no further environment requirements; MolecularSafety needs no secrets.

## Safety

Agents in MolecularSafety are asked to classify molecules for toxicity across multiple safety endpoints. The environment does not present direct safety risks, as agents only provide classification predictions with no access to external systems.

However, this is a dual-use domain. Models trained for toxicity prediction capabilities could potentially be misused for designing harmful compounds in other contexts.

## Citations

```bibtex
@inproceedings{huang2021tdc,
  title     = {Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development},
  author    = {Huang, Kexin and Fu, Tianfan and Gao, Wenhao and Zhao, Yue and Roohani, Yusuf and Leskovec, Jure and Coley, Connor W. and Xiao, Cao and Sun, Jimeng and Zitnik, Marinka},
  booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks},
  year      = {2021},
  eprint    = {2102.09548},
  archivePrefix = {arXiv}
}

@article{xu2012ames,
  title   = {In silico Prediction of Chemical Ames Mutagenicity},
  author  = {Xu, Congying and Cheng, Feixiong and Chen, Lei and Du, Zheng and Li, Weihua and Liu, Guixia and Lee, Philip W. and Tang, Yun},
  journal = {Journal of Chemical Information and Modeling},
  volume  = {52},
  number  = {11},
  pages   = {2840--2847},
  year    = {2012},
  doi     = {10.1021/ci300400a}
}

@article{karim2021cardiotox,
  title   = {CardioTox net: a robust predictor for hERG channel blockade based on deep learning meta-feature ensembles},
  author  = {Karim, Abdul and Lee, Matthew and Balle, Thomas and Sattar, Abdul},
  journal = {Journal of Cheminformatics},
  volume  = {13},
  pages   = {60},
  year    = {2021},
  doi     = {10.1186/s13321-021-00541-z}
}

@article{gayvert2016clintox,
  title   = {A Data-Driven Approach to Predicting Successes and Failures of Clinical Trials},
  author  = {Gayvert, Kaitlyn M. and Madhukar, Neel S. and Elemento, Olivier},
  journal = {Cell Chemical Biology},
  volume  = {23},
  number  = {10},
  pages   = {1294--1301},
  year    = {2016},
  doi     = {10.1016/j.chembiol.2016.07.023}
}
```
