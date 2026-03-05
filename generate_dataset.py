"""
SourceQualityTrain Dataset Generator

Generates 1,000 QA pairs about study exclusions from Cochrane systematic reviews by:
1. Searching PMC via Tavily for Cochrane reviews across medical domains
2. Extracting full text via Tavily extract
3. Using gpt-5.2 to extract the research question and excluded studies table
4. Forming questions from a fixed template + verbatim exclusion reasons
5. Writing to JSONL format

Usage:
    export OPENAI_API_KEY="sk-..."
    export TAVILY_API_KEY="tvly-..."
    python generate_dataset.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient

# ============= Configuration =============

TARGET_PER_DOMAIN = 100  # 100 per domain = 1000 total
OUTPUT_JSONL = Path(__file__).parent / "data" / "sourcequalitytrain.jsonl"
PROGRESS_JSONL = Path(__file__).parent / "sqt_progress.jsonl"
REFERENCE_FILE = Path(__file__).parent / "reference.txt"

DOMAINS = [
    "Oncology",
    "Cardiology",
    "Neurology/Psychiatry",
    "Infectious Disease",
    "Respiratory",
    "Musculoskeletal/Rheumatology",
    "Gastroenterology/Hepatology",
    "Obstetrics/Gynecology",
    "Pediatrics/Neonatal",
    "Other",
]

# Tavily search queries to find Cochrane reviews on PMC per domain
# Each query targets a specific medical subdomain to ensure diversity
DOMAIN_SEARCH_QUERIES = {
    "Oncology": [
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "Characteristics of excluded studies" cancer treatment',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" chemotherapy randomized',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" oncology radiotherapy',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" breast cancer surgery',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" lung cancer treatment',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" colorectal cancer screening',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" lymphoma therapy',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" prostate cancer',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" palliative care cancer',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" leukaemia treatment',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" melanoma immunotherapy',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" ovarian cancer',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" head neck cancer',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" pancreatic cancer',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" bladder cancer',
    ],
    "Cardiology": [
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "Characteristics of excluded studies" heart failure',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" atrial fibrillation anticoagulation',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" hypertension treatment',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" myocardial infarction',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" coronary artery disease',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" cardiac rehabilitation',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" statin cholesterol',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" stroke prevention',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" peripheral vascular disease',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" arrhythmia treatment',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" antiplatelet therapy',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" valve disease surgical',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" thromboembolism prophylaxis',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" aortic aneurysm',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" blood pressure lowering',
    ],
    "Neurology/Psychiatry": [
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "Characteristics of excluded studies" dementia cognitive',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" depression antidepressant',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" epilepsy seizure',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" schizophrenia antipsychotic',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" Parkinson disease',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" anxiety disorder',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" multiple sclerosis',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" migraine headache',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" bipolar disorder',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" ADHD attention deficit',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" PTSD trauma',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" autism spectrum',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" neuropathic pain',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" Alzheimer treatment',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" substance abuse dependence',
    ],
    "Infectious Disease": [
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "Characteristics of excluded studies" tuberculosis treatment',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" HIV antiretroviral',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" malaria prevention',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" hepatitis treatment',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" vaccine immunization',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" pneumonia antibiotic',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" urinary tract infection',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" sepsis critical care',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" fungal infection antifungal',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" parasitic infection',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" influenza treatment prevention',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" sexually transmitted infection',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" wound infection surgical',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" dengue treatment',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" meningitis treatment',
    ],
    "Respiratory": [
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "Characteristics of excluded studies" asthma inhaler',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" COPD pulmonary',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" cystic fibrosis',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" sleep apnoea CPAP',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" bronchiectasis',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" mechanical ventilation',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" pulmonary fibrosis',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" respiratory infection children',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" pneumothorax',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" oxygen therapy',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" smoking cessation',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" chronic cough',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" allergic rhinitis',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" pleural effusion',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" acute respiratory distress',
    ],
    "Musculoskeletal/Rheumatology": [
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "Characteristics of excluded studies" rheumatoid arthritis',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" osteoarthritis knee',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" back pain treatment',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" fracture management',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" osteoporosis treatment',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" gout treatment',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" fibromyalgia',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" shoulder pain rotator',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" carpal tunnel syndrome',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" hip replacement arthroplasty',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" ankylosing spondylitis',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" lupus SLE treatment',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" tendinopathy rehabilitation',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" neck pain cervical',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" physiotherapy exercise',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" joint replacement surgery',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" scoliosis spinal',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" muscle injury strain',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" plantar fasciitis foot',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" psoriatic arthritis',
    ],
    "Gastroenterology/Hepatology": [
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "Characteristics of excluded studies" inflammatory bowel disease',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" liver cirrhosis',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" gastroesophageal reflux',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" Crohn disease',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" ulcerative colitis',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" irritable bowel syndrome',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" peptic ulcer Helicobacter',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" gallstone cholecystectomy',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" pancreatitis treatment',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" celiac disease',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" constipation laxative',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" hepatitis B C treatment',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" appendicitis surgical',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" nausea vomiting antiemetic',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" fatty liver disease',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" gastrointestinal bleeding',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" endoscopy colonoscopy',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" colon rectal polyp',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" dyspepsia functional',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" hernia repair inguinal',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" liver transplant hepatic',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" Barrett esophagus',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" probiotics gut microbiome',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" parenteral nutrition bowel',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" portal hypertension varices',
    ],
    "Obstetrics/Gynecology": [
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "Characteristics of excluded studies" pregnancy labour',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" caesarean section delivery',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" preterm birth prevention',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" pre-eclampsia hypertension pregnancy',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" contraception fertility',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" postpartum haemorrhage',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" gestational diabetes',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" endometriosis treatment',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" cervical cancer screening',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" menopause hormone therapy',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" breastfeeding lactation',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" miscarriage recurrent pregnancy loss',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" polycystic ovary syndrome',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" uterine fibroids',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" induction labour',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" antenatal prenatal care',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" assisted reproduction IVF',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" ectopic pregnancy tubal',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" perineal tear episiotomy',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" ovarian cancer gynaecology',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" maternal fetal monitoring',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" pelvic floor incontinence',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" nausea pregnancy morning sickness',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" vaginal birth breech',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" pain relief childbirth epidural',
    ],
    "Pediatrics/Neonatal": [
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "Characteristics of excluded studies" neonatal premature infant',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" children fever treatment',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" childhood obesity',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" neonatal jaundice phototherapy',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" bronchiolitis children',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" otitis media ear children',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" childhood vaccination immunization',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" neonatal respiratory support',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" paediatric anaesthesia surgery',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" child asthma wheeze',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" tonsillectomy adenoidectomy children',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" diarrhoea rehydration children',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" cerebral palsy children',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" newborn screening neonatal',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" preterm nutrition feeding',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" ADHD attention deficit children',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" autism spectrum disorder children',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" croup laryngitis children',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" neonatal sepsis infection',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" pediatric diabetes insulin',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" neonatal hypoglycemia glucose',
    ],
    "Other": [
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "Characteristics of excluded studies" diabetes treatment',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" anaesthesia postoperative',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" wound care healing',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" kidney renal disease',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" oral health dental',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" eye ophthalmology glaucoma',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" skin dermatology eczema',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" thyroid endocrine',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" urological incontinence',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" hearing loss cochlear',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" nutrition dietary supplement',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" pain management analgesic',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" transplantation immunosuppression',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" burns treatment skin grafting',
        'site:pmc.ncbi.nlm.nih.gov "Cochrane" "excluded studies" rehabilitation physiotherapy recovery',
    ],
}

# Trivial exclusion reasons to filter out
TRIVIAL_REASONS = {
    "duplicate",
    "withdrawn",
    "retracted",
    "duplicate publication",
    "duplicate report",
    "same study",
    "no excluded studies table found",
    "no excluded studies found",
    "error",
    "n/a",
    "not applicable",
}

# Minimum exclusion reason length
MIN_REASON_LENGTH = 10

# Max excluded studies to take from one review (for diversity)
MAX_PER_REVIEW = 10


# ============= Pydantic Models =============

class ExcludedStudy(BaseModel):
    study_ref: str = Field(..., description="Study identifier as written in the review, e.g. 'Smith 2020'")
    exclusion_reason: str = Field(..., description="Verbatim exclusion reason from the review")


class ReviewExtraction(BaseModel):
    research_question: str = Field(..., description="The review's research question/objective")
    excluded_studies: list[ExcludedStudy] = Field(..., description="List of excluded studies")


class QAPair(BaseModel):
    question: str
    answer: str
    review_url: str
    excluded_study_ref: str
    excluded_study_doi: str
    research_question: str
    exclusion_domain: str


# ============= Reference Examples =============

def load_reference_examples() -> str:
    with open(REFERENCE_FILE, "r") as f:
        examples = json.load(f)

    parts = []
    for i, ex in enumerate(examples, 1):
        parts.append(f"""Example {i}:
Question: {ex['question']}
Answer: {ex['answer']}
Review: {ex['review_url']}
Study: {ex['excluded_study_ref']}""")

    return "\n\n".join(parts)


# ============= Core Pipeline =============

async def search_reviews_for_domain(
    tavily_client: AsyncTavilyClient,
    domain: str,
) -> list[dict]:
    """Search for Cochrane reviews on PMC in a specific medical domain using Tavily."""
    queries = DOMAIN_SEARCH_QUERIES.get(domain, [])
    all_results = []
    seen_urls = set()

    for query in queries:
        try:
            response = await tavily_client.search(
                query=query,
                search_depth="basic",
                max_results=5,
            )
            results = response.get("results", [])
            for r in results:
                url = r.get("url", "")
                # Only keep PMC article URLs
                if url and "pmc.ncbi.nlm.nih.gov/articles/" in url and url not in seen_urls:
                    # Normalize URL: strip trailing slash, anchors, etc.
                    base_url = url.split("#")[0].split("?")[0].rstrip("/")
                    if base_url not in seen_urls:
                        seen_urls.add(base_url)
                        all_results.append({
                            "url": base_url + "/",
                            "title": r.get("title", ""),
                            "snippet": r.get("content", ""),
                            "domain": domain,
                        })
            await asyncio.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"  Search error for '{query[:60]}...': {e}")
            continue

    return all_results


async def fetch_review_content(
    tavily_client: AsyncTavilyClient,
    url: str,
) -> str | None:
    """Fetch full text content from a Cochrane review on PMC."""
    try:
        response = await tavily_client.extract(urls=[url])
        results = response.get("results", [])
        if not results:
            return None
        raw_content = results[0].get("raw_content", "")
        if len(raw_content) < 500:
            return None
        # Allow more content than usual since excluded studies table is often near the end
        # For very long content, try to keep the end which has the excluded studies table
        if len(raw_content) > 40000:
            # Keep first 10k (abstract/intro) + last 30k (includes excluded studies)
            raw_content = raw_content[:10000] + "\n\n[...content truncated...]\n\n" + raw_content[-30000:]
        return raw_content
    except Exception as e:
        print(f"  Fetch error for {url}: {e}")
        return None


# ============= Extraction via gpt-5.2 =============

EXTRACTION_PROMPT = """You are extracting structured data from a Cochrane systematic review published on PubMed Central.

Extract the following:

1. **Research question**: The primary research objective from the abstract or objectives section. This should be a clear sentence describing what the review investigates (e.g., "To evaluate the effectiveness of cognitive stimulation for people with dementia on cognition and quality of life").

2. **Excluded studies**: From the "Characteristics of excluded studies" table/section, extract EVERY excluded study entry. Each entry has:
   - study_ref: The study identifier exactly as it appears (e.g., "Smith 2020", "Jones 2015a", "NCT00004315")
   - exclusion_reason: The verbatim reason for exclusion as stated in the review

IMPORTANT:
- Copy the exclusion reasons VERBATIM from the text
- Include ALL excluded studies listed, not just a sample
- The study_ref should match the exact format used in the review
- If you cannot find an excluded studies section, return {{"error": "no excluded studies table found"}}

Return a JSON object:
{{
  "research_question": "...",
  "excluded_studies": [
    {{"study_ref": "...", "exclusion_reason": "..."}},
    ...
  ]
}}

Review content:
{content}"""


async def extract_from_review(
    oai_client: AsyncOpenAI,
    content: str,
) -> ReviewExtraction | None:
    """Use gpt-5.2 to extract research question and excluded studies from review content."""
    prompt = EXTRACTION_PROMPT.format(content=content)

    try:
        response = await oai_client.chat.completions.create(
            model="gpt-5.2",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        result_text = response.choices[0].message.content or ""
        result = json.loads(result_text)

        if "error" in result:
            return None

        if "research_question" not in result or "excluded_studies" not in result:
            return None

        if not result["research_question"] or not result["excluded_studies"]:
            return None

        studies = []
        for entry in result["excluded_studies"]:
            if isinstance(entry, str):
                continue  # Skip malformed entries
            if not isinstance(entry, dict):
                continue
            ref = entry.get("study_ref", "").strip()
            reason = entry.get("exclusion_reason", "").strip()
            if ref and reason:
                studies.append(ExcludedStudy(study_ref=ref, exclusion_reason=reason))

        if not studies:
            return None

        return ReviewExtraction(
            research_question=result["research_question"].strip(),
            excluded_studies=studies,
        )
    except Exception as e:
        print(f"  Extraction error: {e}")
        return None


# ============= Question Formation =============

def format_question(study_ref: str, research_question: str) -> str:
    """Form the standardized question from template."""
    return (
        f"A panel of evidence-based medicine experts determined that the study "
        f"{study_ref} does not provide appropriate evidence to address the "
        f"following research question: {research_question} What was their "
        f"justification for excluding this study?"
    )


def is_trivial_reason(reason: str) -> bool:
    """Check if an exclusion reason is too trivial or is a meta-reference rather than an actual reason."""
    lower = reason.lower().strip().rstrip(".")
    if lower in TRIVIAL_REASONS:
        return True
    if len(reason) < MIN_REASON_LENGTH:
        return True
    # Filter meta-references, LLM artifacts, and non-reason text
    meta_patterns = [
        "supplementary table",
        "additional file",
        "online supplementary",
        "see table",
        "see appendix",
        "listed in",
        "shown in",
        "detailed in",
        "available in",
        "reasons for exclusion are",
        "does not contain",
        "provided review content",
        "no excluded studies",
        "could not find",
        "not found in",
        "table/section",
    ]
    for pattern in meta_patterns:
        if pattern in lower:
            return True
    return False


# ============= Domain Processing =============

async def process_domain(
    oai_client: AsyncOpenAI,
    tavily_client: AsyncTavilyClient,
    domain: str,
    target_count: int,
    existing_questions: set[str],
    existing_review_study_pairs: set[str],
    progress_path: Path = PROGRESS_JSONL,
) -> list[QAPair]:
    """Process a single domain: search for reviews, extract excluded studies, form QA pairs."""
    print(f"\n{'='*60}")
    print(f"Processing domain: {domain}")
    print(f"Target: {target_count} QA pairs")
    print(f"{'='*60}")

    # Step 1: Search for Cochrane reviews
    print(f"  Searching for Cochrane reviews...")
    review_results = await search_reviews_for_domain(tavily_client, domain)
    print(f"  Found {len(review_results)} candidate review URLs")

    qa_pairs = []
    processed_reviews = 0

    for review in review_results:
        if len(qa_pairs) >= target_count:
            break

        processed_reviews += 1
        url = review["url"]
        print(f"\n  [{processed_reviews}] Fetching review: {url[:80]}...")

        # Step 2: Fetch review content
        content = await fetch_review_content(tavily_client, url)
        if not content:
            print(f"    -> No content extracted, skipping")
            continue

        print(f"    -> Got {len(content)} chars of content")

        # Quick check: does the content seem to have excluded studies?
        if "excluded stud" not in content.lower() and "characteristics of excluded" not in content.lower():
            print(f"    -> No excluded studies section found in text, skipping")
            continue

        # Step 3: Extract using gpt-5.2
        extraction = await extract_from_review(oai_client, content)
        if not extraction:
            print(f"    -> Failed to extract data, skipping")
            continue

        print(f"    -> Research question: {extraction.research_question[:80]}...")
        print(f"    -> Found {len(extraction.excluded_studies)} excluded studies")

        # Step 4: Filter and form QA pairs from this review
        review_count = 0
        for study in extraction.excluded_studies:
            if len(qa_pairs) >= target_count:
                break
            if review_count >= MAX_PER_REVIEW:
                break

            # Filter trivial reasons
            if is_trivial_reason(study.exclusion_reason):
                continue

            # Dedup: check review+study combo
            pair_key = f"{url}|{study.study_ref}"
            if pair_key in existing_review_study_pairs:
                continue

            # Form question
            question = format_question(study.study_ref, extraction.research_question)

            # Dedup: check question text
            if question in existing_questions:
                continue

            # Create QA pair
            qa = QAPair(
                question=question,
                answer=study.exclusion_reason,
                review_url=url,
                excluded_study_ref=study.study_ref,
                excluded_study_doi="",  # DOI lookup is best-effort, skip for now
                research_question=extraction.research_question,
                exclusion_domain=domain,
            )

            existing_questions.add(question)
            existing_review_study_pairs.add(pair_key)
            qa_pairs.append(qa)
            save_one_qa(qa, progress_path)
            review_count += 1
            print(f"    -> QA #{len(qa_pairs)}: [{study.study_ref}] -> {study.exclusion_reason[:60]}")

        await asyncio.sleep(1)  # Rate limiting between reviews

    print(f"\n  Domain complete: {len(qa_pairs)}/{target_count} QA pairs generated")
    return qa_pairs


# ============= Progress Tracking =============

def save_one_qa(qa: QAPair, path: Path) -> None:
    """Append a single QA pair to the progress JSONL file."""
    record = {
        "question": qa.question,
        "answer": qa.answer,
        "review_url": qa.review_url,
        "excluded_study_ref": qa.excluded_study_ref,
        "excluded_study_doi": qa.excluded_study_doi,
        "research_question": qa.research_question,
        "exclusion_domain": qa.exclusion_domain,
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_progress(path: Path) -> list[QAPair]:
    """Load previous progress from JSONL."""
    if not path.exists():
        return []
    pairs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                pairs.append(QAPair(
                    question=data["question"],
                    answer=data["answer"],
                    review_url=data["review_url"],
                    excluded_study_ref=data["excluded_study_ref"],
                    excluded_study_doi=data.get("excluded_study_doi", ""),
                    research_question=data["research_question"],
                    exclusion_domain=data["exclusion_domain"],
                ))
            except Exception as e:
                print(f"Warning: Failed to parse progress line: {e}")
                continue
    return pairs


# ============= Main =============

async def main():
    # Validate API keys
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if not openai_api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        sys.exit(1)
    if not tavily_api_key:
        print("ERROR: Set TAVILY_API_KEY environment variable")
        sys.exit(1)

    oai_client = AsyncOpenAI(api_key=openai_api_key)
    tavily_client = AsyncTavilyClient(api_key=tavily_api_key)

    # Load any previous progress
    all_qa_pairs = load_progress(PROGRESS_JSONL)
    existing_questions = {qa.question for qa in all_qa_pairs}
    existing_review_study_pairs = {f"{qa.review_url}|{qa.excluded_study_ref}" for qa in all_qa_pairs}

    if all_qa_pairs:
        print(f"Loaded {len(all_qa_pairs)} existing QA pairs from progress file")
        domain_counts: dict[str, int] = {}
        for qa in all_qa_pairs:
            domain_counts[qa.exclusion_domain] = domain_counts.get(qa.exclusion_domain, 0) + 1
        for d, c in sorted(domain_counts.items()):
            print(f"  {d}: {c}")

    # Process each domain sequentially
    for domain in DOMAINS:
        # Count existing for this domain
        existing_count = sum(1 for qa in all_qa_pairs if qa.exclusion_domain == domain)
        remaining = TARGET_PER_DOMAIN - existing_count

        if remaining <= 0:
            print(f"\nSkipping {domain} (already have {existing_count}/{TARGET_PER_DOMAIN})")
            continue

        domain_pairs = await process_domain(
            oai_client, tavily_client, domain, remaining,
            existing_questions, existing_review_study_pairs,
        )
        all_qa_pairs.extend(domain_pairs)

    # Final output with IDs
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total QA pairs: {len(all_qa_pairs)}")

    # Assign IDs and write final output
    domain_counters: dict[str, int] = {}
    output_records = []

    for qa in all_qa_pairs:
        domain_counters[qa.exclusion_domain] = domain_counters.get(qa.exclusion_domain, 0) + 1
        idx = domain_counters[qa.exclusion_domain]
        record = {
            "id": f"sqt_train_{qa.exclusion_domain.lower().replace('/', '_')}_{idx:03d}",
            "question": qa.question,
            "answer": qa.answer,
            "review_url": qa.review_url,
            "excluded_study_ref": qa.excluded_study_ref,
            "excluded_study_doi": qa.excluded_study_doi,
            "research_question": qa.research_question,
            "exclusion_domain": qa.exclusion_domain,
        }
        output_records.append(record)

    # Write final output
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w") as f:
        for record in output_records:
            f.write(json.dumps(record) + "\n")

    print(f"\nSaved to {OUTPUT_JSONL}")

    # Print domain distribution
    print(f"\nDomain distribution:")
    for domain, count in sorted(domain_counters.items()):
        print(f"  {domain}: {count}")

    # Print answer length stats
    answer_lengths = [len(qa.answer) for qa in all_qa_pairs]
    if answer_lengths:
        print(f"\nAnswer length stats:")
        print(f"  Min: {min(answer_lengths)}")
        print(f"  Max: {max(answer_lengths)}")
        print(f"  Mean: {sum(answer_lengths)/len(answer_lengths):.1f}")

    # Print sample
    print(f"\nSample QA pairs:")
    for qa in all_qa_pairs[:3]:
        print(f"\n  Q: {qa.question[:120]}...")
        print(f"  A: {qa.answer}")
        print(f"  Review: {qa.review_url}")
        print(f"  Study: {qa.excluded_study_ref}")
        print(f"  Domain: {qa.exclusion_domain}")


if __name__ == "__main__":
    asyncio.run(main())
