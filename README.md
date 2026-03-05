# SourceQualityTrain

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/SourceQualityTrain)

## Description

SourceQualityTrain is a training environment for systematic review source quality assessment, based on FutureHouse's [SourceQuality](https://github.com/Future-House/LAB-Bench) benchmark. Agents are given questions about why specific studies were excluded from systematic reviews, and must use web search to identify the verbatim exclusion reason.

## Capabilities

- Understanding systematic review methodology and exclusion criteria
- Locating Cochrane systematic reviews on PubMed Central
- Extracting specific information from excluded studies tables
- Multi-hop reasoning: finding the review, then locating the specific exclusion entry

## Compute Requirements

No sandbox or special compute requirements.

## License

[MIT](https://opensource.org/licenses/MIT)

## Tasks

There are 1,000 training tasks distributed across 10 medical domains:

| Domain | Count | Description |
|--------|-------|-------------|
| Cardiology | 100 | Heart disease, cardiovascular interventions |
| Gastroenterology/Hepatology | 100 | Digestive disorders, liver disease |
| Infectious Disease | 100 | Bacterial, viral, parasitic infections |
| Musculoskeletal/Rheumatology | 100 | Joint disorders, arthritis, bone disease |
| Neurology/Psychiatry | 100 | Brain disorders, mental health conditions |
| Obstetrics/Gynecology | 100 | Pregnancy, women's health |
| Oncology | 100 | Cancer treatment and prevention |
| Other | 100 | Miscellaneous medical specialties |
| Pediatrics/Neonatal | 100 | Child and newborn health |
| Respiratory | 100 | Lung disease, breathing disorders |

Each task asks why a specific study was excluded from a Cochrane systematic review. Questions include the study reference and the review's research question, requiring agents to locate the source and extract the verbatim exclusion reason.

## Reward Structure

Sparse, binary reward:
- **1.0** for correct answers (as judged by LLM grader)
- **0.0** for incorrect or unsure answers

Grading uses semantic equivalence checking: answers that capture the same exclusion reason are accepted, even if phrased differently (e.g., "Not RCT" matches "Not a randomised controlled trial"). The grader is based on LAB-Bench's structured evaluation methodology.

We do not use exact string matching. The LLM grader (gpt-5-mini) evaluates whether the submitted answer captures the core factual content of the expected answer.

## Data

Ground-truth data consists of QA pairs derived from systematic reviews on PubMed Central. Each task includes:
- A question referencing a specific excluded study
- The expected exclusion reason (verbatim from the review)
- The source review URL
- The excluded study reference and domain

Data is stored on the OpenReward platform.

## Tools

Agents have access to three tools:

| Tool | Description |
|------|-------------|
| `web_search` | Search the web using Tavily. Returns titles, URLs, and snippets. |
| `fetch_url` | Fetch full text content from a URL using Tavily extract. Supports pagination for long documents. |
| `submit_answer` | Submit a final answer with explanation. Triggers LLM grading and ends the episode. |

## Time Horizon

SourceQualityTrain is a multi-turn environment. Agents typically search for the relevant systematic review, fetch the review page from PubMed Central, locate the excluded studies table, and extract the specific exclusion reason.

## Environment Difficulty

[Statistics on environment difficulty here]

## Other Environment Requirements

This environment requires the following API keys passed via the `secrets` parameter:
- `openai_api_key`: For LLM-based answer grading
- `tavily_api_key`: For web search and URL content extraction

## Safety

SourceQualityTrain focuses on factual information retrieval from publicly available medical literature. The environment does not involve medical decision-making or patient data. All source reviews are publicly accessible.

## Citations

```bibtex
@article{laurent2024labbench,
  title     = {LAB-Bench: Measuring Capabilities of Language Models for Biology Research},
  author    = {Laurent, Jon M. and Janizek, Joseph D. and Ruzo, Michael and Hinks, Michaela M. and Hammerling, Michael J. and Narayanan, Siddharth and Ponnapati, Manvitha and White, Andrew D. and Rodriques, Samuel G.},
  year      = {2024},
  eprint    = {2407.10362},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}
```

```bibtex
@dataset{GRSourceQualityTrain,
  author    = {General Reasoning Inc. Team},
  title     = {SourceQualityTrain},
  year      = {2026},
  publisher = {OpenReward},
  url       = {https://openreward.ai/GeneralReasoning/SourceQualityTrain}
}
```
