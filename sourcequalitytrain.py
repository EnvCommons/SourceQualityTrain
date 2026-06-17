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

from tavily import AsyncTavilyClient

from openreward.environments import Environment, JSONObject, Server, TextBlock, ToolOutput, tool

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


class WebSearchInput(BaseModel):
    """Parameters for web_search tool"""
    query: str = Field(..., description="Search query to find systematic review or study information")


class FetchUrlInput(BaseModel):
    """Parameters for fetch_url tool"""
    url: str = Field(..., description="URL to fetch (e.g., PubMed Central systematic review page)")
    page: int = Field(default=1, description="Page number to retrieve (1-indexed). Each page contains ~10,000 characters.")


class SubmitAnswerParams(BaseModel):
    """Parameters for submit_answer tool"""
    explanation: str = Field(
        ...,
        description="Your reasoning showing how you found and verified the answer (2-4 sentences)"
    )
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
    3. Uses fetch_url tool to read the review's excluded studies table
    4. Submits the exclusion reason with explanation for LLM-based grading
    5. Receives reward (1.0 correct, 0.0 incorrect) and feedback
    """

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        """
        Initialize SourceQualityTrain environment instance.

        Args:
            task_spec: Task specification with question, answer, review_url, etc.
            secrets: Must contain "openai_api_key" for grading and "tavily_api_key" for search

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
                "Pass secrets={'openai_api_key': 'sk-...', 'tavily_api_key': 'tvly-...'} when creating session."
            )

        # Require Tavily API key for web search - fail fast if missing
        tavily_api_key = secrets.get("tavily_api_key")
        if not tavily_api_key:
            raise ValueError(
                "tavily_api_key required in secrets parameter for web search. "
                "Pass secrets={'openai_api_key': 'sk-...', 'tavily_api_key': 'tvly-...'} when creating session."
            )

        self.openai_client = openai.AsyncClient(api_key=openai_api_key)
        self.tavily_client = AsyncTavilyClient(api_key=tavily_api_key)

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
        return [TextBlock(type="text", text=self.config.question)]

    async def _tavily_with_retry(self, label: str, call, *, max_attempts: int = 4):
        """Call Tavily with exponential backoff, re-raising on persistent failure.

        A genuinely-down dependency (exhausted quota, auth error) exhausts the
        retries and re-raises, so the SDK marks the call ToolFailed and ends the
        rollout. `call` returns a fresh awaitable on each attempt.
        """
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                return await call()
            except Exception as e:
                last_exc = e
                if attempt < max_attempts - 1:
                    wait = min(2 ** attempt, 30)
                    print(f"TAVILY ERROR: {label} | {e} | retry in {wait}s (attempt {attempt + 1}/{max_attempts})")
                    await asyncio.sleep(wait)
        assert last_exc is not None
        raise last_exc

    @tool
    async def web_search(self, params: WebSearchInput) -> ToolOutput:
        """
        Search the web for systematic review or study information using Tavily.
        Returns search results with titles, URLs, and snippets.
        """
        response = await self._tavily_with_retry(
            f"search({params.query!r})",
            lambda: self.tavily_client.search(
                query=params.query,
                search_depth="basic",
                max_results=5,
            ),
        )

        results = response.get("results", [])
        if not results:
            return ToolOutput(
                blocks=[TextBlock(type="text", text="No search results found. Try a different query.")],
                metadata={"query": params.query, "results": []},
                reward=0.0,
                finished=False
            )

        display_parts = [f"Search results for: {params.query}\n"]
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            snippet = result.get("content", "")
            display_parts.append(f"{i}. {title}\n   URL: {url}\n   {snippet}\n")

        display_text = "\n".join(display_parts)

        return ToolOutput(
            blocks=[TextBlock(type="text", text=display_text)],
            metadata={
                "query": params.query,
                "results": results,
                "count": len(results)
            },
            reward=0.0,
            finished=False
        )

    @tool
    async def fetch_url(self, params: FetchUrlInput) -> ToolOutput:
        """
        Fetch and return the text content from a specific URL using Tavily's extract method.
        Use this to read systematic review pages from PubMed Central.
        Content is paginated - use the page parameter to retrieve additional pages.
        """
        PAGE_SIZE = 10000  # Characters per page

        # extract_depth="advanced" pulls more of the rendered page than the default
        # basic depth — needed for large/JS-heavy PMC articles that otherwise come
        # back empty or near-empty.
        response = await self._tavily_with_retry(
            f"extract({params.url!r})",
            lambda: self.tavily_client.extract(
                urls=[params.url],
                extract_depth="advanced",
                format="text",
            ),
        )

        results = response.get("results", [])
        if not results:
            # Tavily produced no result object at all — usually a fetch
            # failure (DNS/timeout/blocked) or an unsupported URL.
            return ToolOutput(
                blocks=[TextBlock(type="text", text=(
                    f"Could not fetch {params.url}: the extractor returned "
                    f"no result. The URL may be unreachable, blocked, or "
                    f"invalid. Try a different source or the article's PMC URL."
                ))],
                metadata={"url": params.url, "results": []},
                reward=0.0,
                finished=False
            )

        raw_content = results[0].get("raw_content", "") or ""
        if not raw_content.strip():
            # A result came back but with no usable text — typically a
            # JavaScript-gated page that renders content client-side, so the
            # extractor saw only an empty shell. Surface that explicitly so
            # the agent picks a different source instead of re-paging into an
            # opaque empty response.
            return ToolOutput(
                blocks=[TextBlock(type="text", text=(
                    f"No readable text could be extracted from {params.url}. "
                    f"The page appears to be JavaScript-gated or otherwise "
                    f"served no content to the extractor. Try the article's "
                    f"PubMed Central (PMC) full-text URL or a direct data/API "
                    f"endpoint instead."
                ))],
                metadata={"url": params.url, "results": results, "empty_content": True},
                reward=0.0,
                finished=False
            )

        total_length = len(raw_content)

        # Calculate pagination
        total_pages = max(1, (total_length + PAGE_SIZE - 1) // PAGE_SIZE)
        page = max(1, min(params.page, total_pages))

        # Extract the requested page
        start_idx = (page - 1) * PAGE_SIZE
        end_idx = min(start_idx + PAGE_SIZE, total_length)
        page_content = raw_content[start_idx:end_idx]

        # Build display text with pagination info
        if total_pages == 1:
            display_text = f"Content from {params.url}:\n\n{page_content}"
        else:
            display_text = f"Content from {params.url} (Page {page}/{total_pages}):\n\n{page_content}"
            if page < total_pages:
                display_text += f"\n\n[Use fetch_url with page={page + 1} to see more content]"

        return ToolOutput(
            blocks=[TextBlock(type="text", text=display_text)],
            metadata={
                "url": params.url,
                "page": page,
                "total_pages": total_pages,
                "total_length": total_length,
                "page_start": start_idx,
                "page_end": end_idx
            },
            reward=0.0,
            finished=False
        )

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

    @tool
    async def submit_answer(self, params: SubmitAnswerParams) -> ToolOutput:
        """
        Submit your final answer for why the study was excluded from the systematic review.

        This tool grades your answer using an LLM judge and returns a reward.
        The episode ends after calling this tool.

        Args:
            explanation: Your reasoning and how you found the answer (2-4 sentences)
            answer: The precise exclusion reason

        Returns:
            ToolOutput with grading result, reward, and feedback
        """
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
                "submitted_explanation": params.explanation,
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
