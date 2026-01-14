"""
Multimodal RAG Evaluation Dataset Generation Prompts
=====================================================

A collection of prompts for generating high-quality Question-Answer datasets
from multimodal technical documents. Designed for RAG system evaluation with
support for text, tables, figures, and images.

This module provides prompts for:
- Document semantic chunking
- Chunk completeness verification
- QA pair generation and verification
- Deduplication and merging
- Retrieval metrics evaluation (faithfulness, precision, recall)
- Multimodal content handling

Usage:
    from mirage.core.prompts import PROMPTS, PROMPTS_DESC, PROMPTS_CHUNK, PROMPTS_METRICS

Author: [Your Name/Organization]
License: Apache 2.0
"""

from __future__ import annotations
from typing import Any

# =============================================================================
# CONFIGURATION
# =============================================================================

PROMPTS: dict[str, Any] = {}
PROMPTS_DESC: dict[str, Any] = {}
PROMPTS_CHUNK: dict[str, Any] = {}
PROMPTS_METRICS: dict[str, Any] = {}
PROMPTS_METRICS_OPT: dict[str, Any] = {}

# Delimiters for structured output parsing
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|#|>END<|#|>"


# =============================================================================
# IMAGE/TABLE DESCRIPTION PROMPTS
# =============================================================================

PROMPTS_DESC["image"] = """
Generate a technical summary of the provided figure in a SINGLE PARAGRAPH (< 250 words).

Guidelines:
- Focus on technical data, relationships, and engineering principles
- Only describe visual attributes if they encode technical information
- Exclude non-technical content (watermarks, page numbers, decorative elements)

Structure (as continuous paragraph):
1. Figure type and engineering objective (one sentence)
2. Technical analysis:
   - Plots/Charts: axes, units, variables, trends, operating regions
   - Diagrams: components, connections, flow direction, system architecture
3. Key engineering insights and practical implications

[EXAMPLE_PLACEHOLDER: Provide domain-specific figure and expected response]
"""

PROMPTS_DESC["table"] = """
Generate a technical summary of the provided table in a SINGLE PARAGRAPH (< 250 words).

Guidelines:
- Focus on data, specifications, and technical limits
- Do not reproduce or enumerate the table data
- Exclude generic headers/footers and document metadata

Structure (as continuous paragraph):
1. Table's engineering function (one sentence, like a caption)
2. Column/row organization and data ranges
3. Primary engineering application or conclusion

[EXAMPLE_PLACEHOLDER: Provide domain-specific table and expected response]
"""


# =============================================================================
# SEMANTIC CHUNKING PROMPTS
# =============================================================================

PROMPTS_CHUNK["semantic_chunking"] = """
You are a document parser for semantic chunking. Segment the markdown into coherent chunks.

## Processing Rules

1. **Exclusions**: Ignore Table of Contents, List of Figures, List of Tables
2. **Artifact Priority**: Identify figures, tables, standalone images first
3. **Semantic Cohesion**: Each chunk = single self-contained topic
4. **Avoid Fragmentation**: Prefer paragraph-level over sentence-level granularity
5. **Content Integrity**: Preserve exact verbatim markdown

## Chunk Types

| Type | Description |
|------|-------------|
| `text` | Textual sections/subsections with title and content |
| `table` | Table with caption, data, and footnotes |
| `table with images` | Table containing embedded images |
| `figure` | Image with caption and description |
| `standalone image` | Image not associated with a figure caption |

## Output Format

```
<chunk_id>[N]<|#|><chunk_type>[type]<|#|><content>[verbatim markdown]<|#|><artifact>[image path(s) or None]<|#|><status>[COMPLETE|INCOMPLETE]<|#|><chunk_end>
```

## Field Definitions

- `chunk_id`: Sequential number (1, 2, 3...)
- `chunk_type`: One of: text, table, table with images, figure, standalone image
- `content`: Exact unmodified markdown
- `artifact`: Image path(s) from `![alt](path)` syntax, or `None`
- `status`: COMPLETE if self-contained, INCOMPLETE if cut off

[EXAMPLE_PLACEHOLDER: Provide sample document and expected chunked output]
"""


PROMPTS_CHUNK["completion_verification"] = """
You are a Chunk Completion Verification Agent. Evaluate if the chunk is semantically COMPLETE or INCOMPLETE.

Domain: {domain}
Expert Role: {expert_persona}

## Incompleteness Indicators

1. **Missing References**: "Figure X", "Table Y", "see Section Z" without actual content
2. **Undefined Terms**: Acronyms, classifications, or procedures used without definition
3. **Implicit Context**: "as mentioned earlier", "the previous method", "this configuration"
4. **Missing Artifacts**: Text describing a figure/table without the actual visual

## Exceptions (Mark COMPLETE)

- Standalone images with visible content
- Normative reference sections
- Universal abbreviations (kW, Hz, °C)

## Output Format

```
Status: COMPLETE|INCOMPLETE
Query: None|<search_query_1> | <search_query_2> | ...
Explanation: <brief explanation>
```

Search queries must be specific enough to retrieve the missing content.

[EXAMPLE_PLACEHOLDER: Provide complete and incomplete chunk examples]
"""


PROMPTS_CHUNK["chunk_addition_verification"] = """
You are a Chunk Addition Verification Agent. Classify if a CANDIDATE chunk should be added to build context.

Context: The ORIGINAL chunk is incomplete. A search found the CANDIDATE chunk.

Expert Role: {expert_persona}
Domain: {domain}

## Classification

**EXPLANATORY**: Directly resolves incompleteness
- Provides missing artifact (figure, table, formula)
- Defines undefined acronym/term
- Supplies referenced prior context

**RELATED**: Useful but doesn't directly resolve
- Same domain/topic, complementary information
- General theory or background
- Could enhance multi-hop QA generation

**UNRELATED**: No contribution
- Different domain with no connection
- No semantic overlap

## Output Format

```
Status: EXPLANATORY|RELATED|UNRELATED
Explanation: <brief justification>
```

[EXAMPLE_PLACEHOLDER: Provide original chunk, search query, and candidate chunk examples]
"""


PROMPTS_CHUNK["relevance_check"] = """
Evaluate if the chunk is relevant to the specified expert role and domain.

Expert Role: {expert_persona}
Domain: {domain}

Chunk Content:
{content}

## RELEVANT if:
- Contains technical information, procedures, specifications useful for the role
- Includes figures, diagrams, charts conveying technical data
- Addresses topics within the domain expertise

## NOT_RELEVANT if:
- Only document metadata (titles, page numbers, copyright)
- Purely decorative content (logos, backgrounds)
- Completely unrelated to the domain

Respond with ONLY: "RELEVANT" or "NOT_RELEVANT"
"""


# =============================================================================
# DOMAIN AND EXPERT EXTRACTION
# =============================================================================

PROMPTS["domain_and_expert_from_topics"] = """
Analyze the following topics extracted from a technical document collection:

{topic_list_str}

Determine:
1. The specific technical/professional domain
2. An appropriate expert role title

## Output Format

```
<|#|>START<|#|>
<|#|>Domain: <domain name>
<|#|>Expert Role: <expert role title>
<|#|>END<|#|>
```

[EXAMPLE_PLACEHOLDER: Provide sample topics and expected domain/role output]
"""


# =============================================================================
# QA GENERATION PROMPTS
# =============================================================================

PROMPTS["question_answer_generation"] = """
You are a(n) {expert_persona} in {domain_context}. Generate QA pairs for evaluating information retrieval systems.

Content:
{content}

## Critical Requirements

1. **Content-Only**: Use ONLY information present in the content. NO external knowledge.
2. **Minimal Coverage**: Generate minimum questions to comprehensively span content without redundancy
3. **Role-Appropriate**: Questions suitable for {expert_persona} in {domain_relevance}
4. **Self-Contained**: Questions must be standalone without implicit references
5. **Non-Trivial**: Require specific content to answer, not general knowledge

## Forbidden Vague References

NEVER use:
- "the provided X", "the described X", "the specified X"
- "this/that/these/those X" without explicit identification
- "according to the content/document/text"

INSTEAD, explicitly name standards, figures, tables, sections.

## Output Format

```
<|#|>START<|#|>
Question<|#|><explicit, self-contained question><|#|>Answer<|#|><brief answer with specific references><|#|>Relevance<|#|><0-10><|#|>Difficulty<|#|><0-10>
<|#|>NEXT<|#|>
...
<|#|>END<|#|>
```

## Rating Scales

- **Relevance** (0-10): Importance to domain expert (0=irrelevant, 10=critical)
- **Difficulty** (0-10): Technical depth required (0=trivial, 10=expert insight)

[EXAMPLE_PLACEHOLDER: Provide content samples with correct and incorrect QA generation examples]
"""


PROMPTS["question_answer_selection"] = """
You are a QA Selection Agent ({expert_persona}, {domain_context}). Evaluate if a QA pair should be SELECTED or REJECTED.

Content:
{content}

Question: {question}
Answer: {answer}

## REJECT if:
1. **Improper References**: Vague references without explicit identification
2. **Vague Phrases**: "the provided/described/specified X", "this/that X"
3. **Trivial**: Answerable with general knowledge alone
4. **Non-Technical**: Document metadata, structure, formatting
5. **Ambiguous**: Unclear or requires unstated assumptions
6. **Out of Scope**: Irrelevant to {domain_relevance}

## SELECT if:
1. Self-contained and explicit references
2. Requires provided content to answer
3. Relevant to domain and appropriate difficulty
4. Demonstrates good technical depth

## Output Format

```
Status<|#|>SELECTED|REJECTED<|#|>Relevance<|#|><0-10><|#|>Difficulty<|#|><0-10><|#|>Reason<|#|><brief explanation>
```

[EXAMPLE_PLACEHOLDER: Provide selection and rejection examples]
"""


PROMPTS["question_answer_verification"] = """
You are a QA Verification Agent ({expert_persona}, {domain_context}). Verify the QA pair.

Content:
{content}

Question: {question}
Answer: {answer}

## Evaluation Criteria

1. Does the question involve specific content details?
2. Does the answer depend on information only in this content?
3. Can someone answer using only general knowledge?
4. For tables/images: Is the answer factually supported by the data?

## Vague Reference Check

QUESTION_INCORRECT if contains:
- "the provided/described/specified X"
- "this/that/these/those X" without identification
- References assuming reader has access to content

## Output Format

```
QUESTION_CORRECT|QUESTION_INCORRECT, ANSWER_CORRECT|ANSWER_INCORRECT, REQUIRES_CONTENT|CAN_ANSWER_WITHOUT_CONTENT
Justification: <brief explanation>
```

[EXAMPLE_PLACEHOLDER: Provide verification examples]
"""


PROMPTS["question_answer_generation_corrected"] = """
You are a(n) {expert_persona} in {domain_context}. Correct a failed QA pair.

Content:
{content}

## Failed QA and Feedback:
{failed_qa_feedback}

## Common Fixes

1. **Vague References**: Replace with explicit identifiers (document, section, figure names)
2. **Factual Errors**: Verify against provided content
3. **Too General**: Make specific to content information
4. **Hallucination**: Only reference what exists in content

## Output Format

```
<|#|>START<|#|>
Question<|#|><corrected question><|#|>Answer<|#|><corrected answer><|#|>Relevance<|#|><0-10><|#|>Difficulty<|#|><0-10>
<|#|>END<|#|>
```

If the original topic cannot be addressed with available content, return empty:
```
<|#|>START<|#|>
<|#|>END<|#|>
```
"""


# =============================================================================
# DEDUPLICATION PROMPTS
# =============================================================================

PROMPTS["deduplication_rank"] = """
You are a Data Curator ({expert_persona}, {domain}). Order similar QA pairs from least to most similar.

## Task
Order from "most distinct/unique" to "most redundant" relative to the cluster's core topic.

Cluster Candidates:
{candidates_text}

## Output Format

```
<|#|>START<|#|>
Question<|#|><ordered question 1><|#|>Answer<|#|><ordered answer 1>
<|#|>NEXT<|#|>
Question<|#|><ordered question 2><|#|>Answer<|#|><ordered answer 2>
...
<|#|>END<|#|>
```

Preserve all content exactly; only reorder.
"""


PROMPTS["deduplication_merge"] = """
You are a Data Curator ({expert_persona}, {domain}). Create minimal high-quality QA pairs from a cluster.

## Task
- Exact duplicates → single best version
- Different aspects → merge or keep distinct as appropriate
- Integrate related sub-questions into comprehensive pairs

Input Candidates:
{candidates_text}

## Output Format

```
<|#|>START<|#|>
Question<|#|><refined question><|#|>Answer<|#|><refined answer>
<|#|>NEXT<|#|>
...
<|#|>END<|#|>
```

[EXAMPLE_PLACEHOLDER: Provide merge examples]
"""


PROMPTS["deduplication_reorganize"] = """
You are a Data Curator ({expert_persona}, {domain}). Reorganize merged QAs into balanced packs.

## Guidelines
- Each pack: related questions sharing a theme (2-4 questions ideal)
- Single concept → one pack; multiple sub-topics → split into packs
- Answer in each pack should address all questions comprehensively

Input:
Merged Questions: <list>
Merged Answers: <list>

## Output Format

```
<|#|>START<|#|>
Question<|#|><related questions separated by newlines><|#|>Answer<|#|><comprehensive answer>
<|#|>NEXT<|#|>
...
<|#|>END<|#|>
```

[EXAMPLE_PLACEHOLDER: Provide reorganization examples]
"""


# =============================================================================
# RERANKER PROMPTS
# =============================================================================

PROMPTS["rerank_vlm"] = """
You are an expert retrieval system. Rank chunks by relevance to the query.

Each chunk is delimited by:
- `<CHUNK_START id=N>` ... `<CHUNK_END id=N>`
- Images: `<IMAGE_START id=X relates_to_chunk=N>` ... `<IMAGE_END id=X>`

## Instructions
1. Analyze text and image relevance
2. Rank from most relevant (Rank 1) to least relevant

## Output Format (exactly)

```
<Rank 1>Chunk X
<Rank 2>Chunk Y
<Rank 3>Chunk Z
...
```

Include ALL chunks. Only output chunk IDs, no content.
"""


PROMPTS["rerank_image_desc"] = """
Generate a concise 100-word technical description of this image. Focus on key technical information, data, and visual elements useful for retrieval.
"""


# =============================================================================
# METRICS EVALUATION PROMPTS
# =============================================================================

PROMPTS_METRICS["multihop_reasoning"] = """
Evaluate the QA pair's reasoning complexity.

Contexts: {contexts}
Question: {question}
Answer: {answer}

## Analysis
1. **Hop Count**: Distinct information pieces needed (1 = single fact, 2+ = multi-hop)
2. **Bridge Entity**: Concept/term connecting information pieces
3. **Reasoning Score**: 0.0 (trivial) to 1.0 (complex multi-step)

{format_instructions}

[EXAMPLE_PLACEHOLDER: Provide single-hop and multi-hop examples]
"""


PROMPTS_METRICS["visual_dependency"] = """
Determine if the question requires visual content to answer.

Context (Text Only): {contexts}
Question: {question}

## Instructions
- Answer using ONLY the text context
- No outside knowledge or hallucination
- If visual information is required but missing, output: `MISSING_VISUAL`

[EXAMPLE_PLACEHOLDER: Provide examples requiring and not requiring visuals]
"""


PROMPTS_METRICS["multimodal_faithfulness_vlm"] = """
Evaluate answer faithfulness given multimodal context.

Question: {question}
Answer: {answer}

## Analysis
1. Supported by TEXT content?
2. Supported by VISUAL content?
3. Any hallucinated/unsupported claims?

## Output Format

```
TEXT_SUPPORTED: YES/NO
VISUAL_SUPPORTED: YES/NO/NA
FAITHFULNESS_SCORE: 0.0-1.0
EXPLANATION: <brief>
```
"""


PROMPTS_METRICS["multimodal_answer_quality_vlm"] = """
Evaluate answer quality with multimodal context.

Question: {question}
Answer: {answer}

## Criteria
1. **Completeness**: Fully addresses question using all relevant context
2. **Accuracy**: Factually correct based on context
3. **Visual Info Used**: Incorporates visual elements (if relevant)

## Output Format

```
COMPLETENESS: 0.0-1.0
ACCURACY: 0.0-1.0
VISUAL_INFO_USED: YES/NO/NA
OVERALL_SCORE: 0.0-1.0
REASONING: <brief>
```
"""


PROMPTS_METRICS["context_necessity_without"] = """
Answer the question using ONLY general knowledge. Do NOT make up specific facts.

Question: {question}

If you cannot answer confidently without reference material, respond: "CANNOT_ANSWER_WITHOUT_CONTEXT"

Answer:
"""


PROMPTS_METRICS["context_necessity_verify"] = """
Compare two answers for semantic equivalence.

Ground Truth: {ground_truth}
Model Answer: {model_answer}

## Output Format

```
MATCH: YES/NO/PARTIAL
EXPLANATION: <brief>
```
"""


# =============================================================================
# OPTIMIZED METRICS PROMPTS (Minimal LLM Calls)
# =============================================================================

PROMPTS_METRICS_OPT["prepare_qa"] = """
Analyze the QA pair and extract evaluation components.

QUESTION: {question}
ANSWER: {answer}
REFERENCE: {reference}

## Tasks
1. Extract concept hops (concept1 --> concept2 --> ...)
2. Extract atomic claims from ANSWER
3. Extract atomic claims from REFERENCE
4. Generate {num_reverse_questions} questions the answer could address

## Output Format

```
CONCEPT_HOPS_QUESTION:
concept1 --> concept2 --> ...

ANSWER_CLAIMS:
- [claim 1]
- [claim 2]
...

REFERENCE_CLAIMS:
- [claim 1]
- [claim 2]
...

REVERSE_QUESTIONS:
- [question 1]
- [question 2]
...
```

[EXAMPLE_PLACEHOLDER: Provide QA analysis example]
"""


PROMPTS_METRICS_OPT["faithfulness"] = """
Verify if each claim can be inferred from the context.

CONTEXT:
{context}

CLAIMS TO VERIFY:
{claims}

## Output Format (one per line)

```
CLAIM_1: SUPPORTED/NOT_SUPPORTED
CLAIM_2: SUPPORTED/NOT_SUPPORTED
...
```
"""


PROMPTS_METRICS_OPT["context_recall"] = """
Verify if each reference claim can be attributed to the context.

CONTEXT:
{context}

REFERENCE CLAIMS:
{claims}

## Output Format (one per line)

```
CLAIM_1: ATTRIBUTED/NOT_ATTRIBUTED
CLAIM_2: ATTRIBUTED/NOT_ATTRIBUTED
...
```
"""


PROMPTS_METRICS_OPT["context_precision"] = """
Evaluate context chunk relevance for answering the question.

QUESTION: {question}
REFERENCE ANSWER: {reference}

CONTEXT CHUNKS:
{contexts}

## Output Format (one per line)

```
CHUNK_1: RELEVANT/NOT_RELEVANT
CHUNK_2: RELEVANT/NOT_RELEVANT
...
```
"""


PROMPTS_METRICS_OPT["multimodal_faithfulness"] = """
Verify claims against multimodal context (text AND images).

QUESTION: {question}
ANSWER: {answer}

CLAIMS TO VERIFY:
{claims}

## Output Format

```
CLAIM_1: SUPPORTED/NOT_SUPPORTED | SOURCE: TEXT/IMAGE/BOTH/NONE
CLAIM_2: SUPPORTED/NOT_SUPPORTED | SOURCE: TEXT/IMAGE/BOTH/NONE
...

SUMMARY:
TEXT_GROUNDED: YES/NO
VISUAL_GROUNDED: YES/NO/NA
SUPPORTED_COUNT: [number]
TOTAL_CLAIMS: [number]
```
"""


PROMPTS_METRICS_OPT["multimodal_relevance"] = """
Generate questions the answer could address and evaluate context utilization.

ANSWER: {answer}

## Tasks
1. Generate {num_questions} diverse questions this answer could address
2. Indicate if each uses TEXT, IMAGE, or BOTH context

## Output Format

```
GENERATED_QUESTIONS:
Q1: [question] | USES: TEXT/IMAGE/BOTH
Q2: [question] | USES: TEXT/IMAGE/BOTH
...

CONTEXT_UTILIZATION:
USES_TEXT: YES/NO
USES_IMAGES: YES/NO/NA
RELEVANCE_SCORE: 0.0-1.0
```
"""


PROMPTS_METRICS_OPT["context_necessity_without"] = """
Answer using ONLY general knowledge. Do NOT fabricate specific facts.

If you cannot answer confidently, respond: CANNOT_ANSWER

QUESTION: {question}

YOUR ANSWER:
"""


PROMPTS_METRICS_OPT["context_necessity_verify"] = """
Compare model answer to ground truth.

GROUND TRUTH: {ground_truth}
MODEL ANSWER: {model_answer}

Respond with exactly one of:
- MATCH: YES (correct and complete)
- MATCH: PARTIAL (partially correct)
- MATCH: NO (incorrect or missing key information)

YOUR VERDICT:
"""


PROMPTS_METRICS_OPT["multihop_reasoning"] = """
Analyze if answering requires multi-hop reasoning.

CONTEXTS:
{contexts}

QUESTION: {question}
ANSWER: {answer}

## Output Format

```
HOP_COUNT: [number]
REASONING_SCORE: 0.0-1.0
BRIDGE_ENTITY: [entity or None]
EXPLANATION: <brief>
```

- HOP_COUNT: 1 = single fact, 2+ = multi-hop
- REASONING_SCORE: 0.0 = trivial, 1.0 = complex
"""


PROMPTS_METRICS_OPT["visual_dependency"] = """
Determine if the question can be answered from text alone.

TEXT CONTEXT:
{contexts}

QUESTION: {question}

If you can answer completely from text, provide your answer.
If visual information is missing and required, respond: MISSING_VISUAL

YOUR RESPONSE:
"""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_prompt(category: str, name: str, **kwargs) -> str:
    """
    Retrieve and format a prompt template.
    
    Args:
        category: Prompt category ('prompts', 'desc', 'chunk', 'metrics', 'metrics_opt')
        name: Prompt name within category
        **kwargs: Template variables to substitute
    
    Returns:
        Formatted prompt string
    """
    prompt_dicts = {
        'prompts': PROMPTS,
        'desc': PROMPTS_DESC,
        'chunk': PROMPTS_CHUNK,
        'metrics': PROMPTS_METRICS,
        'metrics_opt': PROMPTS_METRICS_OPT
    }
    
    if category not in prompt_dicts:
        raise ValueError(f"Unknown category: {category}")
    
    prompt_dict = prompt_dicts[category]
    
    if name not in prompt_dict:
        raise ValueError(f"Unknown prompt: {name} in category {category}")
    
    template = prompt_dict[name]
    
    if kwargs:
        return template.format(**kwargs)
    
    return template


def list_prompts() -> dict[str, list[str]]:
    """List all available prompts by category."""
    return {
        'prompts': list(PROMPTS.keys()),
        'desc': list(PROMPTS_DESC.keys()),
        'chunk': list(PROMPTS_CHUNK.keys()),
        'metrics': list(PROMPTS_METRICS.keys()),
        'metrics_opt': list(PROMPTS_METRICS_OPT.keys())
    }


# =============================================================================
# MODULE INFO
# =============================================================================

__version__ = "1.0.0"
__all__ = [
    "PROMPTS",
    "PROMPTS_DESC", 
    "PROMPTS_CHUNK",
    "PROMPTS_METRICS",
    "PROMPTS_METRICS_OPT",
    "get_prompt",
    "list_prompts"
]
