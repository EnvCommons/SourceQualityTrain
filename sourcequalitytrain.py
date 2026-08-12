"""
SourceQualityTrain Environment - Systematic review study exclusion QA with web search

A training environment with 1,000 QA pairs about why studies were excluded from
Cochrane systematic reviews. Agents research questions using web search and URL
fetching, then submit answers for LLM-based grading. Questions cover diverse
medical domains including oncology, cardiology, neurology, infectious disease,
respiratory, musculoskeletal, gastroenterology, obstetrics, pediatrics, and more.
"""

import asyncio
import json
import re
import openai
from pydantic import BaseModel, Field
from typing import Dict, List


from openreward.environments import Environment, JSONObject, Server, TextBlock, ToolOutput, terminal, tool
from openreward.toolsets import WebToolset

from constants import SOURCEQUALITYTRAIN_JSONL


# Grader prompt template for LLM-based answer evaluation
# Based on LABBench2's STRUCTURED_EVALUATION_PROMPT for semantic equivalence checking
GRADER_PROMPT_TEMPLATE = """You are a helpful assistant that evaluates the correctness of an answer.

Consider the question, the expected correct answer, and the submitted answer.
Your task is to determine if the submitted answer is correct.

Be rigorous but reasonable in your evaluation:
- Accept answers that are semantically equivalent, even if phrased slightly differently
- Accept expanded forms of abbreviations (e.g., "Not RCT" matches "Not a randomised controlled trial")
- Accept answers that clearly capture the same exclusion reason even if worded differently
- Minor differences in punctuation, capitalization, or article usage should not affect correctness

First provide your reasoning, then provide your final answer. Your answer MUST be one of: "correct", "incorrect", or "unsure".

Use the following format:
<reasoning>
Your explanation of the evaluation here.
</reasoning>
<answer>correct/incorrect/unsure</answer>

## QUESTION ##
{question}

## EXPECTED ANSWER ##
{correct_answer}

## SUBMITTED ANSWER ##
{answer}

## EVALUATION ##"""


# Pydantic schemas for type safety
class SourceQualityTaskSpec(BaseModel):
    """Task specification for SourceQualityTrain environment"""
    id: str
    question: str
    answer: str
    review_url: str
    excluded_study_ref: str
    excluded_study_doi: str
    research_question: str
    exclusion_domain: str


class SubmitAnswerParams(BaseModel):
    """Parameters for the terminal grading tool. The assistant's final message
    becomes `answer` — include the precise exclusion reason there."""
    answer: str = Field(
        ...,
        description="The precise reason why the study was excluded from the systematic review"
    )


def load_sourcequalitytrain_data() -> Dict[str, List[Dict]]:
    """
    Load SourceQualityTrain JSONL dataset.

    Returns:
        Dict with "train" split containing list of task dicts

    Raises:
        FileNotFoundError: If JSONL file not found at expected path
    """
    print(f"Loading SourceQualityTrain data from: {SOURCEQUALITYTRAIN_JSONL}")

    if not SOURCEQUALITYTRAIN_JSONL.exists():
        raise FileNotFoundError(
            f"SourceQualityTrain JSONL not found at {SOURCEQUALITYTRAIN_JSONL}. "
            f"Please ensure the dataset file exists."
        )

    tasks = []
    with open(SOURCEQUALITYTRAIN_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                tasks.append({
                    "id": data["id"],
                    "question": data["question"],
                    "answer": data["answer"],
                    "review_url": data["review_url"],
                    "excluded_study_ref": data["excluded_study_ref"],
                    "excluded_study_doi": data.get("excluded_study_doi", ""),
                    "research_question": data["research_question"],
                    "exclusion_domain": data["exclusion_domain"],
                })
            except Exception as e:
                print(f"Warning: Failed to parse line: {e}")
                continue

    print(f"Successfully loaded {len(tasks)} tasks")
    return {"train": tasks}


# Load dataset once at module level
ALL_DATA = load_sourcequalitytrain_data()


class SourceQualityTrain(Environment):
    """
    SourceQualityTrain environment: Systematic review exclusion QA with web search
    and LLM grading.

    Agent workflow:
    1. Receives a question about why a study was excluded from a systematic review
    2. Uses web_search tool to find the relevant systematic review
    3. Uses web_fetch tool to read the review's excluded studies table
    4. Submits the exclusion reason with explanation for LLM-based grading
    5. Receives reward (1.0 correct, 0.0 incorrect) and feedback
    """

    # web_search / web_fetch come from the SDK rather than being hand-rolled here.
    # Which provider answers is process configuration (OPENREWARD_SEARCH_BACKEND,
    # default "backsearch"), so changing search provider needs no change here.
    #
    # The toolset owns the error split too: an unfetchable page stays tool output
    # the agent can act on, while a missing key or exhausted quota raises so the
    # rollout ends with a blank reward rather than a score that reads as a bad answer.
    toolsets = [WebToolset]

    # Search hits keep their snippets, as the prompt promises. Off in the SDK by
    # default, which would force a fetch per candidate just to triage results.
    web_include_snippets = True

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        """
        Initialize SourceQualityTrain environment instance.

        Args:
            task_spec: Task specification with question, answer, review_url, etc.
            secrets: Must contain "openai_api_key" for grading; search credentials
                (api_key / tavily_api_key) are forwarded to the search backend

        Raises:
            ValueError: If required API keys missing or task_spec invalid
        """
        super().__init__(task_spec)
        self.config = SourceQualityTaskSpec.model_validate(task_spec)

        # Require OpenAI API key for grader - fail fast if missing
        openai_api_key = secrets.get("openai_api_key")
        if not openai_api_key:
            raise ValueError(
                "openai_api_key required in secrets parameter for LLM grading. "
                "Pass secrets={'openai_api_key': 'sk-...'} when creating session."
            )

        # Read live by WebToolset on every tool call, so the search backend takes its
        # credentials from the session rather than the server process. The configured
        # backend picks the key it needs: `api_key` for backsearch, `tavily_api_key`
        # for tavily. No up-front check — which key is required depends on the backend.
        self.search_secrets = secrets

        self.openai_client = openai.AsyncClient(api_key=openai_api_key)

    @classmethod
    def list_splits(cls) -> list[str]:
        """Return available data splits"""
        return ["train"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        """
        List all tasks for a given split.

        Args:
            split: Data split name (only "train" available)

        Returns:
            List of task specifications

        Raises:
            ValueError: If split is unknown
        """
        if split != "train":
            raise ValueError(f"Unknown split: {split}. Available splits: train")

        return [
            {
                "id": task["id"],
                "question": task["question"],
                "answer": task["answer"],
                "review_url": task["review_url"],
                "excluded_study_ref": task["excluded_study_ref"],
                "excluded_study_doi": task["excluded_study_doi"],
                "research_question": task["research_question"],
                "exclusion_domain": task["exclusion_domain"],
            }
            for task in ALL_DATA["train"]
        ]

    def get_prompt(self) -> list[TextBlock]:
        """
        Generate prompt for the agent.

        Returns:
            List containing single TextBlock with question
        """
        text = (
            f"{self.config.question}\n\n"
            "Reply with the precise exclusion reason as an ordinary message when "
            "you're ready. Your whole reply is graded, so keep it focused — no "
            "preamble or closing remarks."
        )
        return [TextBlock(type="text", text=text)]

    async def _grade_answer(
        self,
        answer: str
    ) -> Dict:
        """
        Use LLM grader to evaluate answer correctness.

        Args:
            answer: Agent's submitted answer

        Returns:
            Dict with keys: is_correct, grading_response

        Note: Uses gpt-5-mini without temperature parameter (per CLAUDE.md)
        """
        grader_prompt = GRADER_PROMPT_TEMPLATE.format(
            question=self.config.question,
            correct_answer=self.config.answer,
            answer=answer
        )

        response = await self.openai_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": grader_prompt}],
        )

        grading_text = response.choices[0].message.content or ""

        # Parse verdict from <answer></answer> tags: "correct" = 1.0, "incorrect" or "unsure" = 0.0
        answer_match = re.search(r'<answer>\s*(correct|incorrect|unsure)\s*</answer>', grading_text, re.IGNORECASE)
        if answer_match:
            result_value = answer_match.group(1).strip().lower()
            is_correct = result_value == "correct"
        else:
            # Fallback: check for verdict in text
            lower_text = grading_text.lower()
            is_correct = "correct" in lower_text and "incorrect" not in lower_text and "unsure" not in lower_text

        return {
            "is_correct": is_correct,
            "grading_response": grading_text
        }

    @terminal
    @tool
    async def submit_answer(self, params: SubmitAnswerParams) -> ToolOutput:
        """Grade the assistant's final message against the reference exclusion reason."""
        grading_result = await self._grade_answer(params.answer)

        reward = 1.0 if grading_result["is_correct"] else 0.0
        result_status = "Correct" if grading_result["is_correct"] else "Incorrect"

        display_text = f"""{result_status}

Grading Analysis:
{grading_result['grading_response']}

Reward: {reward:.1f}

Expected Answer: {self.config.answer}
Your Answer: {params.answer}

Review: {self.config.review_url}"""

        return ToolOutput(
            blocks=[TextBlock(type="text", text=display_text)],
            metadata={
                "task_id": self.config.id,
                "is_correct": grading_result["is_correct"],
                "grading_response": grading_result["grading_response"],
                "submitted_answer": params.answer,
                "correct_answer": self.config.answer,
                "question": self.config.question,
                "review_url": self.config.review_url,
                "excluded_study_ref": self.config.excluded_study_ref,
                "exclusion_domain": self.config.exclusion_domain,
            },
            reward=reward,
            finished=True
        )


if __name__ == "__main__":
    Server([SourceQualityTrain]).run()
