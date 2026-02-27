"""
SafetyClassify - Molecular Safety Classification Environment

Single-turn environment where agents classify molecules as safe or unsafe
across multiple safety endpoints: AMES mutagenicity, hERG cardiotoxicity,
and clinical trial toxicity (ClinTox).

Data source: TDC AMES, hERG_Karim, ClinTox datasets (pooled).
Binary reward: 1.0 for correct classification, 0.0 for incorrect.
"""

import json
import os
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from openreward.environments import (
    Environment,
    JSONObject,
    Split,
    TextBlock,
    ToolOutput,
    tool,
)

if os.path.exists("/orwd_data"):
    ENV_PATH = Path("/orwd_data")
else:
    ENV_PATH = Path(__file__).parent


def load_all_tasks() -> dict[str, list[dict]]:
    data_dir = ENV_PATH / "data"
    all_tasks = {}
    for split in ["train", "test"]:
        json_file = data_dir / f"{split}.json"
        if json_file.exists():
            with open(json_file, "r", encoding="utf-8") as f:
                all_tasks[split] = json.load(f)
        else:
            print(f"Warning: {json_file} not found")
            all_tasks[split] = []
    return all_tasks


ALL_TASKS = load_all_tasks()

ANSWERS = {
    task["task_id"]: {"value": task["answer"]}
    for split_tasks in ALL_TASKS.values()
    for task in split_tasks
}

print(f"Loaded {len(ANSWERS)} SafetyClassify tasks")


class SafetyClassifyTaskSpec(BaseModel):
    task_id: str
    smiles: str
    property_name: str
    class_labels: str
    question: str


class SubmitClassificationInput(BaseModel):
    prediction: int = Field(
        ..., description="Your predicted class: 0 (negative/safe) or 1 (positive/unsafe)"
    )


class SafetyClassify(Environment):
    """
    Molecular safety classification environment.

    Agents classify molecules as safe or unsafe across multiple
    safety endpoints (AMES, hERG, ClinTox).
    Binary reward: 1.0 for correct, 0.0 for incorrect.
    """

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.validated = SafetyClassifyTaskSpec.model_validate(task_spec)

        if self.validated.task_id not in ANSWERS:
            raise ValueError(f"Task {self.validated.task_id} not found in ANSWERS")

        self.answer = ANSWERS[self.validated.task_id]

    @classmethod
    def list_splits(cls) -> list[Split]:
        return [
            Split(name="train", type="train"),
            Split(name="test", type="test"),
        ]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split not in ALL_TASKS:
            return []
        return [
            {k: v for k, v in task.items() if k != "answer"}
            for task in ALL_TASKS[split]
        ]

    async def get_prompt(self) -> List[TextBlock]:
        return [TextBlock(text=self.validated.question)]

    @tool
    async def submit_prediction(self, params: SubmitClassificationInput) -> ToolOutput:
        """Submit your safety classification for the molecule (0 = safe, 1 = unsafe)."""
        predicted = params.prediction
        actual = self.answer["value"]
        correct = predicted == actual
        reward = 1.0 if correct else 0.0

        if correct:
            feedback = (
                f"Correct! The molecule is {'positive/unsafe' if actual == 1 else 'negative/safe'} "
                f"for {self.validated.property_name}.\n"
                f"Reward: {reward:.1f}"
            )
        else:
            feedback = (
                f"Incorrect. You predicted {'positive/unsafe' if predicted == 1 else 'negative/safe'}, "
                f"but the molecule is {'positive/unsafe' if actual == 1 else 'negative/safe'} "
                f"for {self.validated.property_name}.\n"
                f"Reward: {reward:.1f}"
            )

        return ToolOutput(
            blocks=[TextBlock(text=feedback)],
            metadata={
                "task_id": self.validated.task_id,
                "smiles": self.validated.smiles,
                "property_name": self.validated.property_name,
                "predicted": predicted,
                "actual": actual,
                "correct": correct,
            },
            reward=reward,
            finished=True,
        )
