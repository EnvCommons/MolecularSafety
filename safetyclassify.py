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
import re
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from openreward.environments import (
    Environment,
    JSONObject,
    Split,
    TextBlock,
    ToolOutput,
    terminal,
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
    prediction: str = Field(
        ..., description="Your final message. Include the digit 0 (negative/safe) or 1 (positive/unsafe)."
    )


_DIGIT_RE = re.compile(r"(?<![\d.])([01])(?!\.?\d)")


def _extract_prediction(text: str) -> Optional[int]:
    """Extract 0 or 1 from a free-form assistant message.

    Matches a standalone 0/1 (not adjacent to other digits) and returns the
    LAST such mention. Returns None if no standalone digit is found — chosen
    over a permissive default because scoring an unrelated digit as a real
    prediction would silently mislabel refusals and off-topic replies.
    """
    matches = _DIGIT_RE.findall(text)
    return int(matches[-1]) if matches else None


_OLD_SUBMIT_LINE = "Submit your prediction as 0 or 1 using the submit_prediction tool."
_NEW_SUBMIT_LINE = (
    "Reply with your final answer as an ordinary message. State the digit 0 or 1."
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
        question = self.validated.question.replace(_OLD_SUBMIT_LINE, _NEW_SUBMIT_LINE)
        return [TextBlock(text=question)]

    @terminal
    @tool
    async def submit_prediction(self, params: SubmitClassificationInput) -> ToolOutput:
        """Grade the assistant's final message as a safety classification (0 or 1)."""
        predicted = _extract_prediction(params.prediction)
        actual = self.answer["value"]
        correct = predicted is not None and predicted == actual
        reward = 1.0 if correct else 0.0

        if predicted is None:
            feedback = (
                f"No 0/1 prediction found in your message. "
                f"The molecule is {'positive/unsafe' if actual == 1 else 'negative/safe'} "
                f"for {self.validated.property_name}.\n"
                f"Reward: {reward:.1f}"
            )
        elif correct:
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
