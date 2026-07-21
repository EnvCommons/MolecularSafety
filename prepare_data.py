"""
Download and prepare safety classification datasets from TDC.

Pools 3 safety endpoints (AMES, hERG_Karim, ClinTox) into 1000 train + 100 test tasks.
Run once locally: python prepare_data.py
"""

import json
from pathlib import Path

import pandas as pd
from tdc.single_pred import Tox

DATASETS = [
    {
        "name": "AMES",
        "property": "AMES Mutagenicity",
        "class_labels": "0 = non-mutagenic, 1 = mutagenic",
        "description": "whether this molecule is mutagenic (AMES test positive, class 1) or non-mutagenic (class 0)",
    },
    {
        "name": "hERG_Karim",
        "property": "hERG Cardiotoxicity",
        "class_labels": "0 = non-blocker, 1 = blocker",
        "description": "whether this molecule is an hERG potassium channel blocker (class 1) or non-blocker (class 0)",
    },
    {
        "name": "ClinTox",
        "property": "Clinical Trial Toxicity",
        "class_labels": "0 = non-toxic, 1 = toxic",
        "description": "whether this molecule showed toxicity in clinical trials (class 1) or not (class 0)",
    },
]

TRAIN_SIZE = 1000
TEST_SIZE = 100
TOTAL = TRAIN_SIZE + TEST_SIZE


def make_question(smiles: str, ds_info: dict) -> str:
    return (
        f"You are a molecular safety assessment expert.\n\n"
        f"Given the molecule with SMILES notation: {smiles}\n\n"
        f"Predict {ds_info['description']}.\n\n"
        f"Classes: {ds_info['class_labels']}\n\n"
        f"Reply with your final answer as an ordinary message. State the digit 0 or 1."
    )


def main():
    all_rows = []

    for ds_info in DATASETS:
        print(f"Downloading {ds_info['name']}...")
        data = Tox(name=ds_info["name"])
        df = data.get_data()
        df = df.dropna(subset=["Drug", "Y"])
        df = df.drop_duplicates(subset=["Drug"])

        # Ensure binary classification
        df["Y"] = df["Y"].astype(int)
        df = df[df["Y"].isin([0, 1])]

        df["property_name"] = ds_info["property"]
        df["class_labels"] = ds_info["class_labels"]
        df["ds_info_key"] = ds_info["name"]

        all_rows.append(df)
        print(f"  -> {len(df)} molecules (class distribution: {df['Y'].value_counts().to_dict()})")

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nTotal pooled molecules: {len(combined)}")

    selected = combined.head(TOTAL)

    ds_info_map = {d["name"]: d for d in DATASETS}

    tasks = []
    for idx, row in selected.iterrows():
        split = "test" if idx < TEST_SIZE else "train"
        ds_key = row["ds_info_key"]
        ds = ds_info_map[ds_key]
        task = {
            "task_id": f"safety_{split}_{idx}",
            "smiles": row["Drug"],
            "property_name": row["property_name"],
            "class_labels": row["class_labels"],
            "answer": int(row["Y"]),
            "question": make_question(row["Drug"], ds),
        }
        tasks.append(task)

    test_tasks = [t for t in tasks if "test" in t["task_id"]]
    train_tasks = [t for t in tasks if "train" in t["task_id"]]

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    with open(data_dir / "test.json", "w") as f:
        json.dump(test_tasks, f, indent=2)
    with open(data_dir / "train.json", "w") as f:
        json.dump(train_tasks, f, indent=2)

    print(f"\nSaved {len(train_tasks)} train tasks and {len(test_tasks)} test tasks")


if __name__ == "__main__":
    main()
