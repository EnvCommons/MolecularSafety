"""Unit tests for SafetyClassify grading.

The dataset is not in the repo, so these tests register small fixture tasks in
the module's answer table. Run with:
    uv run --no-project --with-requirements requirements.txt --with pytest \
        --with pytest-asyncio python -m pytest tests.py
"""

import json

import pytest

import safetyclassify
from safetyclassify import SafetyClassify, SubmitClassificationInput

LABEL_WORDS = {0: "negative/safe", 1: "positive/unsafe"}

FIXTURE_TASKS = [
    {
        "task_id": "fixture_ames_pos",
        "smiles": "Nc1ccc2cc3ccccc3cc2c1",
        "property_name": "AMES Mutagenicity",
        "class_labels": "0 = non-mutagenic, 1 = mutagenic",
        "question": "Is this molecule mutagenic? Nc1ccc2cc3ccccc3cc2c1",
        "answer": 1,
    },
    {
        "task_id": "fixture_herg_neg",
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "property_name": "hERG Cardiotoxicity",
        "class_labels": "0 = non-blocker, 1 = blocker",
        "question": "Is this molecule an hERG blocker? CC(=O)Oc1ccccc1C(=O)O",
        "answer": 0,
    },
    {
        "task_id": "fixture_clintox_pos",
        "smiles": "O=C(O)CCCCC(=O)O",
        "property_name": "Clinical Trial Toxicity",
        "class_labels": "0 = non-toxic, 1 = toxic",
        "question": "Did this molecule show clinical toxicity? O=C(O)CCCCC(=O)O",
        "answer": 1,
    },
]

RESULT_METADATA_KEYS = {"task_id", "smiles", "property_name", "predicted", "correct"}


@pytest.fixture(autouse=True)
def fixture_answers(monkeypatch):
    answers = dict(safetyclassify.ANSWERS)
    answers.update({t["task_id"]: {"value": t["answer"]} for t in FIXTURE_TASKS})
    monkeypatch.setattr(safetyclassify, "ANSWERS", answers)


def make_env(task: dict) -> SafetyClassify:
    return SafetyClassify(task_spec={k: v for k, v in task.items() if k != "answer"})


async def grade(task: dict, prediction: str):
    env = make_env(task)
    return await env.submit_prediction(SubmitClassificationInput(prediction=prediction))


@pytest.mark.asyncio
@pytest.mark.parametrize("task", FIXTURE_TASKS, ids=lambda t: t["task_id"])
async def test_gold_prediction_scores_one(task):
    result = await grade(task, f"My final answer is {task['answer']}.")
    assert result.reward == 1.0
    assert result.finished is True
    assert result.metadata["correct"] is True
    assert result.metadata["predicted"] == task["answer"]


@pytest.mark.asyncio
@pytest.mark.parametrize("task", FIXTURE_TASKS, ids=lambda t: t["task_id"])
async def test_wrong_prediction_scores_zero(task):
    result = await grade(task, f"My final answer is {1 - task['answer']}.")
    assert result.reward == 0.0
    assert result.finished is True
    assert result.metadata["correct"] is False
    assert result.metadata["predicted"] == 1 - task["answer"]


@pytest.mark.asyncio
@pytest.mark.parametrize("task", FIXTURE_TASKS, ids=lambda t: t["task_id"])
async def test_missing_prediction_scores_zero(task):
    result = await grade(task, "I cannot tell.")
    assert result.reward == 0.0
    assert result.metadata["predicted"] is None
    assert result.metadata["correct"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("task", FIXTURE_TASKS, ids=lambda t: t["task_id"])
@pytest.mark.parametrize("kind", ["gold", "wrong", "missing"])
async def test_result_does_not_reveal_label(task, kind):
    """Text and metadata carry the verdict and the agent's own prediction only."""
    prediction = {
        "gold": f"Answer: {task['answer']}",
        "wrong": f"Answer: {1 - task['answer']}",
        "missing": "No idea.",
    }[kind]
    result = await grade(task, prediction)

    assert set(result.metadata) == RESULT_METADATA_KEYS
    text = "\n".join(b.text for b in result.blocks)
    assert "molecule is" not in text
    if kind != "gold":
        # The label's wording may only appear when it is the agent's own correct
        # prediction being echoed back.
        assert LABEL_WORDS[task["answer"]] not in text
        assert LABEL_WORDS[task["answer"]] not in json.dumps(result.metadata)


@pytest.mark.asyncio
async def test_repeat_submission_is_penalised_and_not_regraded():
    task = FIXTURE_TASKS[0]
    env = make_env(task)
    first = await env.submit_prediction(SubmitClassificationInput(prediction=str(task["answer"])))
    second = await env.submit_prediction(SubmitClassificationInput(prediction=str(task["answer"])))
    assert first.reward == 1.0
    assert second.reward == safetyclassify.REPEAT_SUBMISSION_PENALTY
    assert second.metadata == {"already_submitted": True, "submission_count": 1}
