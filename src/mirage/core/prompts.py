from __future__ import annotations
from typing import Any

# Define the figure locations
PROMPTS: dict[str, Any] = {}
PROMPTS_DESC: dict[str, Any] = {}
PROMPTS_CHUNK: dict[str, Any] = {}
PROMPTS_METRICS: dict[str, Any] = {}
PROMPTS_METRICS: dict[str, Any] = {}

# All delimiters must be formatted as "<|UPPER_CASE_STRING|>"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|#|>END<|#|>"


file_loc1 = "Samples/VLM/Singles/CurvePlots/IEC60034-2-2{ed2.0}b_Fig10_Plot.png"
file_loc2 = "Samples/VLM/Singles/Circuits/IEC60034-2-1{ed3.0}b_Fig14_Circuit.png"
file_loc3 = "Samples/VLM/Singles/Drawings/iec61800-9-2{ed2.0}b_Fig8_Diagram.png"
file_loc4 = "Samples/VLM/Singles/Flowcharts/iec61800-9-2{ed2.0}b_Fig27_FlowChart.png"
file_loc5 = "Samples/VLM/Singles/CurvePlots/iec61800-9-2{ed2.0}b_FigE7_Plot.png"

file_loc6 = "Samples/VLM/Singles/Tables/iec60079-7{ed5.1b}_Table18.png"


#%% Table and image description prompts

PROMPTS_DESC["image"] = """

Generate a SINGLE-PARAGRAPH technical summary (<250 words) of the provided figure suitable for design documentation. Focus strictly on technical data and functional relationships, omitting decorative elements or artifacts. 

Structure the response as follows: 
    1. Context: Identify the figure type and its primary engineering objective. Focus entirely on the technical data, principles, and relationships the figure intends to convey. Skip decorative elements and non-technical content. 
    2. Deconstruction: Analyze key features. For plots, specify axes, units, variables, and trends/regions. For diagrams, describe components, flows, and system architecture.
    3. Implication: Conclude with the critical engineering insights and practical design implications derived from the figure.
         
"""

PROMPTS_DESC["table"] = f"""

Generate a SINGLE-PARAGRAPH summary (<250 words) of the provided table suitable for professional documentation. Focus on the table's organization and utility, specifically avoiding reproduction of the actual data content. 

Structure the response as follows: 

    1. Objective: State the table's primary subject matter or functional purpose similar to a caption.
    2. Layout: Describe the arrangement of columns and rows, identifying the specific categories, variables, or metrics presented, inferences drawn.
    3. Significance: Conclude with the table's broader value, identifying the insights it offers or the specific type of analysis, decision-making, or reference usage it facilitates.
    
"""
#%%  Chunk


PROMPTS_CHUNK["completion_verification"] = """
You are a Chunk Analysis Agent. Your task is to evaluate the provided text chunk for semantic completeness AND extract its core thematic concepts given that you are a(n) `{expert_persona}` working in the domain of `{domain}`.

**PHASE 1: COMPLETENESS CHECK**
Evaluate if the chunk is self-contained.
1. **Unresolved References:** Citations ("see Figure 1," "refer to Table A") where content is missing.
2. **Undefined Terminology:** Specific acronyms, jargon, or proper nouns without definition. *Exception: Universal units/abbreviations are allowed.*
3. **Broken Context:** Phrases relying on missing prior text ("the latter method," "this scenario").
*If INCOMPLETE, you must generate specific search queries to retrieve the missing information.*

**PHASE 2: CONCEPT EXTRACTION**
Identify 3-5 pivotal keywords or short phrases that define the chunk's unique subject matter (e.g., specific entities, methodologies, or central themes).

**Expected Output Format:**
<|#|>START<|#|>
Status<|#|><COMPLETE or INCOMPLETE><|#|>Query<|#|><Search strings separated by pipes OR "None"><|#|>Explanation<|#|><One sentence explanation for status><|#|>Concepts<|#|><List of concepts separated by commas>
<|#|>END<|#|>

------------------------------------------------
Example (Domain: Cooking)
------------------------------------------------
Input Chunk:
"Once the oven reaches the target temperature mentioned in the previous section, place the tray on the middle rack for 20 minutes."

RESPONSE:
<|#|>START<|#|>
Status<|#|>INCOMPLETE<|#|>Query<|#|>"target oven temperature" | "baking temperature setting"<|#|>Explanation<|#|>The text relies on "target temperature mentioned in the previous section," which is missing from this chunk.<|#|>Concepts<|#|>Baking Process, Oven Settings, Tray Positioning
<|#|>END<|#|>
"""


PROMPTS_CHUNK["chunk_addition_verification"] = """
You are a(n) `{expert_persona}` from the domain of `{domain}` working as a Context Integration Agent. Your goal is to determine if a CANDIDATE CHUNK is RELATED or EXPLAINS the information gaps in an ORIGINAL CHUNK to build context for multihop Question-Answer generation.

**Context:** The ORIGINAL CHUNK is missing specific information (references, definitions, or prior context). A SEARCH QUERY retrieved the CANDIDATE CHUNK.

**Task:** Classify the CANDIDATE CHUNK as follows:
*   **EXPLANATORY (The Direct Fix):** Explicitly resolves the missing information. It provides the referenced artifact (chart, data, quote), defines the unknown term, or supplies the missing predecessor text.
*   **RELATED (The Background):** Shares the same specific topic or theme but does *not* directly resolve the missing element. It provides useful complementary info, theoretical background, or broader context.
*   **UNRELATED (The Noise):** Distinctly different topic or domain with no semantic connection to the search query or original text.

**Expected Output Format:**
<|#|>START<|#|>
Status<|#|><EXPLANATORY | RELATED | UNRELATED><|#|>Explanation<|#|><Brief single-sentence justification>
<|#|>END<|#|>

------------------------------------------------
Example (Domain: Furniture Assembly)
------------------------------------------------
Original Chunk (Missing Info):
"Step 4: Secure the legs using the fasteners listed in the 'Hardware Pack A' inventory table."

Candidate Chunk (Retrieved Text):
"Inventory Table - Hardware Pack A: Contains 16x M4 screws, 4x Washers, and 1x Hex Key."

RESPONSE:
<|#|>START<|#|>
Status<|#|>EXPLANATORY<|#|>Explanation<|#|>The candidate chunk explicitly provides the content of the "Inventory Table" referenced in the original chunk, resolving the missing list of fasteners.
<|#|>END<|#|>
"""


PROMPTS_CHUNK["semantic_chunking"] = """
System Role: You are a Semantic Chunking Engine. Your task is to segment markdown documents into self-contained, coherent chunks based on structural and visual logic. 

CORE SEGMENTATION RULES: 

     1. Exclusions: Ignore navigational sections (Table of Contents, List of Figures/Tables). Generate None for content and artifact. COMPLETE as status. Mark as index chunk.
     2. Atomicity: A chunk must represent a single complete unit: a coherent text block, a full table, or a distinct visual artifact. Never create a chunk containing only a header/title; always merge headers with their following content.
     3. Cohesion: Avoid over-fragmentation. Merge short, related text blocks (e.g., brief subsections) to ensure every chunk is substantive.
     4. Integrity: Preserve verbatim markdown (including formatting). Do not alter content.
     5. Continuity Check: If the input text cuts off abruptly (mid-sentence/mid-paragraph) at the very end of the file, mark status as INCOMPLETE. Otherwise, COMPLETE.
     

CHUNK CLASSIFICATION & EXTRACTION: 

    1. figure: An image associated with a formal caption (e.g., "Figure X"). Content includes the image syntax, caption, and immediate description. Extract image path to <artifact>.
    2. table: A markdown table with its caption and footnotes. Does not contain images. <artifact> is None.
    3. table with images: A markdown table that contains image syntax (![...]). Extract all image paths to <artifact>.
    4. standalone image: An image without a formal "Figure" caption (e.g., icons, logos, inline visuals). Extract image path to <artifact>.
    5. text: Narrative content, lists (e.g., definitions), or headers + paragraphs. <artifact> is None.
    6. index: A table of contents, list of figures, list of tables, or other index-like structure. <artifact> is None.
     

EXPECTED OUTPUT FORMAT:
Produce a single line per chunk using this strict delimiter-separated format: 

<chunk_id><|#|><chunk_type><|#|><content><|#|><artifact><|#|><status><|#|><chunk_end> 

Field Definitions: 

    chunk_id: Sequential integer (1, 2, 3...).
    chunk_type: strictly [index | text | table | table with images | figure | standalone image].
    content: The exact verbatim markdown text.
    artifact: The file path(s) extracted from ![alt](path) syntax. If multiple, list all. If none, write None.
    status: [COMPLETE | INCOMPLETE].
     

------------------------------------------------
Example 1: Finance Report (Text, Figure, and Table)
------------------------------------------------
QUERY:

# Q3 Market Performance Report
The third quarter showed significant volatility across major indices, driven largely by fluctuations in the technology and energy sectors. While the S&P 500 maintained a neutral stance, emerging markets faced headwinds due to currency devaluation.
This chart tracks the daily volatility index (VIX) against the S&P 500 closing price over the 90-day period. The red line indicates the volatility spikes observed in mid-August.
![VIX vs SP500](/assets/charts/q3_volatility.png)
Figure 3.1 – Volatility Index vs. Market Close
## Sector Analysis
Despite the volatility, the consumer discretionary sector outperformed expectations. The following table breaks down the percentage growth by sector compared to the previous quarter.
Table 2 – Sector Performance Summary
| Sector | Q2 Growth | Q3 Growth | Change |
|---|---|---|---|
| Technology | 4.5% | -1.2% | -5.7% |
| Energy | 2.1% | 5.3% | +3.2% |
| Healthcare | 1.0% | 1.1% | +0.1% |
*Data source: Global Market Data Feed, adjusted for inflation.*
For a detailed breakdown of individual stock performance, refer to Appendix C.

RESPONSE:

<chunk_id>1<|#|><chunk_type>text<|#|><content># Q3 Market Performance Report
The third quarter showed significant volatility across major indices, driven largely by fluctuations in the technology and energy sectors. While the S&P 500 maintained a neutral stance, emerging markets faced headwinds due to currency devaluation.<|#|><artifact>None<|#|><status>COMPLETE<|#|><chunk_end>
<chunk_id>2<|#|><chunk_type>figure<|#|><content>This chart tracks the daily volatility index (VIX) against the S&P 500 closing price over the 90-day period. The red line indicates the volatility spikes observed in mid-August.
![VIX vs SP500](/assets/charts/q3_volatility.png)
Figure 3.1 – Volatility Index vs. Market Close<|#|><artifact>![VIX vs SP500](/assets/charts/q3_volatility.png)<|#|><status>COMPLETE<|#|><chunk_end>
<chunk_id>3<|#|><chunk_type>table<|#|><content>## Sector Analysis
Despite the volatility, the consumer discretionary sector outperformed expectations. The following table breaks down the percentage growth by sector compared to the previous quarter.
Table 2 – Sector Performance Summary
| Sector | Q2 Growth | Q3 Growth | Change |
|---|---|---|---|
| Technology | 4.5% | -1.2% | -5.7% |
| Energy | 2.1% | 5.3% | +3.2% |
| Healthcare | 1.0% | 1.1% | +0.1% |
*Data source: Global Market Data Feed, adjusted for inflation.*
For a detailed breakdown of individual stock performance, refer to Appendix C.<|#|><artifact>None<|#|><status>COMPLETE<|#|><chunk_end>

---------------------------------------------------------------------
Example 2: Biological Research (Definitions and Table with Images)
---------------------------------------------------------------------

QUERY:

## 4. Cellular Morphology
This section defines the key structural variations observed in the bacterial samples collected from the riverbed sediment.
4.1
**Cocci**
Spherical or oval-shaped bacteria, typically 0.5 to 1.0 micrometers in diameter. They may occur singly, in pairs (diplococci), or in clusters (staphylococci).
[Reference: General Microbiology, 5th Ed.]
4.2
**Bacilli**
Rod-shaped bacteria that can vary widely in length. Some bacilli form endospores under stressful environmental conditions.
## 5. Gram Staining Results
The samples were subjected to Gram staining to differentiate bacterial species based on cell wall properties. The visual results are cataloged below.
Table 3 – Gram Stain Observations
| Sample ID | Classification | Microscopic View | Characteristics |
|---|---|---|---|
| S-104 | Gram-positive | ![Gram Positive](/imgs/micro/s104_pos.jpg) | Thick peptidoglycan layer, stains purple/blue. |
| S-109 | Gram-negative | ![Gram Negative](/imgs/micro/s109_neg.jpg) | Thin peptidoglycan layer, stains pink/red. |
Note: All images taken at 1000x magnification with oil immersion.

RESPONSE:

<chunk_id>1<|#|><chunk_type>text<|#|><content>## 4. Cellular Morphology
This section defines the key structural variations observed in the bacterial samples collected from the riverbed sediment.
4.1
**Cocci**
Spherical or oval-shaped bacteria, typically 0.5 to 1.0 micrometers in diameter. They may occur singly, in pairs (diplococci), or in clusters (staphylococci).
[Reference: General Microbiology, 5th Ed.]<|#|><artifact>None<|#|><status>COMPLETE<|#|><chunk_end>
<chunk_id>2<|#|><chunk_type>text<|#|><content>4.2
**Bacilli**
Rod-shaped bacteria that can vary widely in length. Some bacilli form endospores under stressful environmental conditions.<|#|><artifact>None<|#|><status>COMPLETE<|#|><chunk_end>
<chunk_id>3<|#|><chunk_type>table with images<|#|><content>## 5. Gram Staining Results
The samples were subjected to Gram staining to differentiate bacterial species based on cell wall properties. The visual results are cataloged below.
Table 3 – Gram Stain Observations
| Sample ID | Classification | Microscopic View | Characteristics |
|---|---|---|---|
| S-104 | Gram-positive | ![Gram Positive](/imgs/micro/s104_pos.jpg) | Thick peptidoglycan layer, stains purple/blue. |
| S-109 | Gram-negative | ![Gram Negative](/imgs/micro/s109_neg.jpg) | Thin peptidoglycan layer, stains pink/red. |
Note: All images taken at 1000x magnification with oil immersion.<|#|><artifact>![Gram Positive](/imgs/micro/s104_pos.jpg) ![Gram Negative](/imgs/micro/s109_neg.jpg)<|#|><status>COMPLETE<|#|><chunk_end>

"""

#%% Role extraction, QA generation prompts and Verification prompts

PROMPTS["domain_and_expert_from_topics"] = """
I have analyzed a technical document collection and extracted the following key topics:

{topic_list_str}

Based on these topics, please determine:
1. The specific technical or professional domain these topics belong to.
2. A specific expert role title for a professional in this domain.

Format your response exactly as follows (do not add any other text):
<|#|>START<|#|>
<|#|>Domain: <The Domain> 
<|#|>Expert Role: <The Expert Role>
<|#|>END<|#|>

------------------------------------------------
Example 1:
------------------------------------------------
Input Topics:
- Topic 0 (Count: 120): motor, efficiency, loss, test, iec
- Topic 1 (Count: 85): voltage, current, power, measurement, circuit
- Topic 2 (Count: 60): thermal, temperature, cooling, rise, insulation

Response:
<|#|>START<|#|>
<|#|>Domain: Electrical Engineering – Electric Motors and Drives
<|#|>Expert Role: Motor Design and Testing Engineer
<|#|>END<|#|>

------------------------------------------------
Example 2:
------------------------------------------------
Input Topics:
- Topic 0 (Count: 95): pipeline, corrosion, pressure, flow, valve
- Topic 1 (Count: 70): safety, hazard, risk, inspection, maintenance
- Topic 2 (Count: 45): pump, compressor, seal, leak, vibration

Response:
<|#|>START<|#|>
<|#|>Domain: Mechanical Engineering – Piping and Fluid Systems
<|#|>Expert Role: Pipeline Integrity Engineer
<|#|>END<|#|>
"""


PROMPTS_CHUNK["relevance_check"] = """
You are a(n) `{expert_persona}` from the domain of `{domain}` working as a Content Relevance Evaluator. Your goal is to determine if a specific semantic chunk (text, data, or visuals) provides value to a specific {expert_persona} within the {domain}.

**Inputs:**
- Chunk Content: {content}

**Evaluation Guidelines:**
- **RELEVANT:** The content contains actionable insights, specific data, technical details, or visual aids (charts, diagrams) essential to the expert's workflow and distinct to the domain.
- **NOT_RELEVANT:** The content is purely administrative (TOCs, copyright, metadata), decorative (logos, stock backgrounds), or completely unrelated to the expert's field.

**Expected Output Format:**
<|#|>START<|#|>
Status<|#|><RELEVANT | NOT_RELEVANT><|#|>Explanation<|#|><Brief single-sentence justification>
<|#|>END<|#|>

------------------------------------------------
Example 1 (RELEVANT):
------------------------------------------------
Expert Role: Investment Analyst
Domain: Finance – Equity Markets
Content:
Table 4.2 illustrates the Year-over-Year (YoY) revenue growth across the APAC region. While the semiconductor sector saw a 14% contraction, the cloud infrastructure segment expanded by 22%, driving the aggregate EBITDA margin to 18.5%.

RESPONSE:
<|#|>START<|#|>
Status<|#|>RELEVANT<|#|>Explanation<|#|>The content contains specific financial metrics (YoY growth, EBITDA) and sector performance data essential for equity analysis.
<|#|>END<|#|>

------------------------------------------------
Example 2 (NOT_RELEVANT):
------------------------------------------------
Expert Role: Molecular Biologist
Domain: Life Sciences – Genomics
Content:
Copyright © 2024 Global Science Journals. All rights reserved. No part of this publication may be reproduced without written permission.
[Image: Small generic icon of a book]

RESPONSE:
<|#|>START<|#|>
Status<|#|>NOT_RELEVANT<|#|>Explanation<|#|>The content represents administrative metadata (copyright/legal text) with no scientific value.
<|#|>END<|#|>
"""

PROMPTS["question_answer_generation"] = """

You are a(n) {expert_persona} in the domain of {domain_context}. Your task is to construct high-quality, multi-hop Question-Answer pairs by synthesizing information across multiple text chunks.

**Content:**
{content}

**EXECUTION PROTOCOL:**
1.  **Analyze:** Identify distinct chunks and extract critical technical keywords for each.
2.  **Map:** Identify "Bridge Keywords"—concepts that intersect or relate across different chunks.
3.  **Synthesize:** Construct one or more questions that **requires** combining information from multiple chunks to answer. If a question can be answered by a single chunk, it is invalid.
4.  **Decompose:** Verify the logic by mapping specific parts of the Questions and Answers back to their source chunks.

**CRITICAL CONSTRAINTS:**
*   **Multi-Hop Requirement:** The questions must be unsolvable without synthesizing facts from at least two distinct locations in the text.
*   **Self-Sufficiency:** Questions must be standalone. Generate the MINIMUM number of questions needed to comprehensively span the content without redundancy
    *   **NEVER** use vague phrases like "the provided text," "the second figure," or "Section 1."
    *   **ALWAYS** explicitly name the specific entity, standard, or concept (e.g., "In the Q3 Financial Report..." or "According to the ISO 9001 guidelines...").
*   **Strict Grounding:** Answer ONLY using the provided content. Do not hallucinate external details.

**OUTPUT FORMAT:**
Respond using strictly this format (repeat for each distinct question(s)/answer(s) pair):

<|#|>START<|#|>
Chunk Count: <Integer>
Keywords per Chunk: <Chunk 1: [A, B], Chunk 2: [C, D]>
Related Keywords: <[A] relates to [C] via...>
<|#|>QA_GENERATION<|#|>
Question: <Specific, self-contained, multi-hop question>
Answer: <Concise synthesized answer>
Relevance: <0-10>
Difficulty: <0-10>
<|#|>DECOMPOSITION<|#|>
Question Source: <Specific phrase in Q -> Derived from Chunk X>
Answer Source: <Specific fact in A -> Derived from Chunk Y>
<|#|>NEXT<|#|>
Question: ...
Answer: ...
Relevance: ...
Difficulty: ...
Question Source: ...
Answer Source: ...
<|#|>NEXT<|#|>
...

<|#|>END<|#|>

------------------------------------------------
Example 1:
------------------------------------------------
Expert Persona: Food Safety Officer
Domain Context: Dietary Compliance & Menu Auditing
Content: 

    [Chunk 1: Bistro Menu Description] The "Autumn Harvest Risotto" is a creamy dish featuring Arborio rice, butternut squash, aged parmesan cheese, and is slow-cooked in a house-made chicken broth. 

    [Chunk 2: The Green-Leaf Certification Guide] To qualify for the "Green-Leaf Vegetarian Label," a dish must be entirely free of meat, poultry, and seafood flesh, including any stocks, broths, or gravies derived from animal tissue. Dairy and eggs are permitted. 
     
System Response: 

<|#|>START<|#|>
Chunk Count: 2
Keywords per Chunk: <Chunk 1: [Autumn Harvest Risotto, chicken broth, parmesan cheese], Chunk 2: [Green-Leaf Vegetarian Label, free of meat/poultry, stocks/broths derived from animal tissue]>
Related Keywords: <[chicken broth] relates to [stocks/broths derived from animal tissue] via poultry origin>
<|#|>QA_GENERATION<|#|>
Question: According to the Green-Leaf Certification Guide, why does the "Autumn Harvest Risotto" fail to qualify for the "Green-Leaf Vegetarian Label"?
Answer: The "Autumn Harvest Risotto" fails to qualify because it is cooked in chicken broth, and the Green-Leaf Certification Guide explicitly excludes dishes containing stocks or broths derived from animal tissue (poultry).
Relevance: 10
Difficulty: 3
<|#|>DECOMPOSITION<|#|>
Question Source: <"Autumn Harvest Risotto" -> Derived from Chunk 1; "Green-Leaf Vegetarian Label" -> Derived from Chunk 2>
Answer Source: <Contains "chicken broth" -> Derived from Chunk 1; "Excludes stocks/broths derived from animal tissue" -> Derived from Chunk 2>
<|#|>END<|#|> 

"""

PROMPTS["question_answer_selection"] = """

You are a {expert_persona} in {domain_context}. Your task is to accept or reject the following QA pair for a technical dataset.

**CRITICAL CONTEXT:** Users see ONLY the question. They cannot see the provided content. Questions must be 100% self-contained.

**Content:** {content}
**Question:** {question}
**Answer:** {answer}

**EVALUATION PROTOCOL:**

**REJECT (Fatal Flaws):**
1. **Context Dependency:** Question relies on implicit context or vague pointers.
   - *Banned Phrases:* "the provided text", "the figure below", "the table", "this section", "the described symbol", "according to the document".
   - *Requirement:* Must explicitly name the document, standard, or entity (e.g., "In IEC 60034-1...", "Table 5 of the Q3 Report...").
2. **Triviality:** Answerable using general knowledge; does not require the specific content.
3. **Non-Technical:** Asks about page numbers, prefaces, document structure, or metadata.
4. **Ambiguity:** Unclear or confusing without seeing the source text.

**SELECT (High Quality):**
1. **Self-Contained:** A user reading *only* the question knows exactly what specific standard, machine, or concept is being referenced.
2. **Grounded:** The answer is technically accurate and derived directly from the content.
3. **Expert Value:** High relevance to {domain_relevance} with appropriate technical depth.

**SCORING (0-10 in discrete steps):**
*   **Relevance:** Value to an {expert_persona}. (0-3: Low relevance or rejected, 4-6: Moderate relevance, 7-8: High relevance, 9-10: Critical/essential knowledge)
    *   **Difficulty:** Technical depth/insight for a {expert_persona}.(0-3: Basic/trivial or rejected, 4-6: Moderate technical understanding required, 7-8: Deep technical knowledge required, 9-10: Expert-level insight required)

**OUTPUT FORMAT:**
<|#|>START<|#|>
Status<|#|><SELECTED | REJECTED><|#|>Relevance<|#|><Integer><|#|>Difficulty<|#|><Integer><|#|>Reason<|#|><Brief single-sentence explanation>
<|#|>END<|#|>

------------------------------------------------
Example 1 (SELECTED):
------------------------------------------------

Expert Persona: Technical Support Specialist
Domain Context: Smart Home Device Troubleshooting
Content:

    [Excerpt from SafeGuard Pro Thermostat User Manual]
    Error Code E-74 indicates a Wi-Fi connection timeout. To resolve this on the SafeGuard Pro, hold the center dial for 10 seconds to enter AP Mode, then reconnect via the mobile app.
    Note: Do not press the reset button on the back, as this will erase all heating schedules.  

Question:

    "According to the SafeGuard Pro User Manual, why should you avoid pressing the rear reset button when trying to resolve Error Code E-74?"  

Answer:

    "Pressing the rear reset button will erase all heating schedules."  
 
MODEL OUTPUT (The response generated by the prompt): 

<|#|>START<|#|>
Status<|#|>SELECTED<|#|>Relevance<|#|>9<|#|>Difficulty<|#|>2<|#|>Reason<|#|>The question is fully self-contained; it explicitly names the device ("SafeGuard Pro") and the specific error situation ("Error Code E-74") rather than saying "the device described" or "this error." 
<|#|>END<|#|>
"""

PROMPTS["question_answer_verification"] = """

You are a(n) {expert_persona} from the domain of {domain_context} working as a Question-Answer Verification Expert. Evaluate the QA pair below for an Information Retrieval system where users see only the question (no context/figures).

**IMPORTANT CONTEXT**: Questions are asked STANDALONE to an information retrieval system WITHOUT any additional context. The user CANNOT see any content, figures, or tables - they only see the question text. Questions must be completely self-contained and unambiguous.

**Inputs:**
Content: {content}
Question(s): {questions}
Answer(s): {answers}

**Evaluation Instructions:**
Analyze the QA pair and return the appropriate labels based on the following logic:

1.  **Question Standalone Check:**
    *   **QUESTION_INCORRECT:** Use if the question refers to unseen context (e.g., "the provided text," "this figure," "the described warning") or lacks specific identifiers.
    *   **QUESTION_CORRECT:** Use if the question is unambiguous and fully self-contained (e.g., explicit standard names like "IEC 60034").

2.  **Answer Accuracy Check:**
    *   **ANSWER_CORRECT:** Use if the answer is factually supported by specific data/text in the content.
    *   **ANSWER_INCORRECT:** Use if the answer is unsupported or factually wrong.

3.  **Content Dependency Check:**
    *   **REQUIRES_CONTENT:** Use if the answer relies on unique data/facts found only in this content.
    *   **CAN_ANSWER_WITHOUT_CONTENT:** Use if the question can be answered using general domain knowledge.

**Expected Output Format:**
<|#|>START<|#|>
Status<|#|><Label 1>, <Label 2>, <Label 3><|#|>Explanation<|#|><Brief single-sentence justification>
<|#|>END<|#|>

------------------------------------------------
Example (Vague Reference - "in the table/figure" - QUESTION_INCORRECT):
------------------------------------------------
Content:
The operating limits for Series-X pumps are defined by the manufacturer specifications.
Table 2: Maximum pressure ratings by fluid type.
Figure 5: Flow rate vs. Head characteristics.

Question:
What is the maximum pressure rating for oil listed in the table, and how does the flow rate behave near the shut-off head as shown in figure 21?

Answer:
According to the provided data, the maximum pressure for oil is 25 bar, and the flow rate drops to zero as the head approaches the shut-off point of 40 meters.

RESPONSE:
<|#|>START<|#|>
Status<|#|>QUESTION_INCORRECT, ANSWER_CORRECT, REQUIRES_CONTENT<|#|>Explanation<|#|>The question refers to "the table" and "figure 21". A standalone reader sees only the question text and does not know the table or figure 21. To be CORRECT, the question must explicitly name the source or be fully self-contained.
<|#|>END<|#|>
"""

#%% QA Correction Prompt

PROMPTS["question_answer_generation_corrected"] = """
You are a(n) {expert_persona} in the domain of {domain_context}. Your task is to rewrite a question-answer (QA) pair that failed verification based on the feedback provided.
Content:
{content}

**FAILED QA PAIR(S) AND FEEDBACK:**
{failed_qa_feedback}

Correction Requirements: 

    1. Eliminate Vague References to fix QUESTION_INCORRECT: The question must be fully self-contained. Replace terms like "the provided text," "the described X," or "this table" with explicit names (e.g., "According to IEC 60034-7," "In Table 2") if available in the content.
    2. Ensure Specificity to fix ANSWER_INCORRECT: The answer shall correctly answer the question based on the content.
    3. Strict Accuracy to fix CAN_ANSWER_WITHOUT_CONTENT: The answer must rely exclusively on facts and values present in the content. Do not contradict the content or hallucinate external info.

Output Format:
<|#|>START<|#|>Question<|#|><Corrected, specific, self-contained question><|#|>Answer<|#|><Accurate answer citing specific content sources><|#|>Relevance<|#|><0-10><|#|>Difficulty<|#|><0-10><|#|>END<|#|> 

------------------------------------------------
 Example (Fixing vague reference & specificity):
------------------------------------------------
FAILED QA:
Question: How do I reset the device mentioned in the text?
Answer: You should hold the button down for a few seconds until it resets.
Verification Feedback: QUESTION_INCORRECT - The question uses "the device mentioned in the text" which is vague. It must name the specific product. CAN_ANSWER_WITHOUT_CONTENT - The answer is too general and applies to many devices; it lacks the specific steps found in the content.
Content Context:
"User Manual for EcoTemp S3 Thermostat. To perform a factory reset, press and hold the central dial for exactly 10 seconds until the display turns blue. The system will reboot and ask for the default PIN (0000)."
CORRECTED RESPONSE:
<|#|>START<|#|>
Question<|#|>According to the User Manual for the EcoTemp S3 Thermostat, what are the specific steps and visual indicators required to perform a factory reset?<|#|>Answer<|#|>To factory reset the EcoTemp S3 Thermostat, the central dial must be pressed and held for exactly 10 seconds until the display turns blue, after which the system will reboot.<|#|>Relevance<|#|>9<|#|>Difficulty<|#|>3
<|#|>END<|#|>
"""

#%% Reranker prompts
PROMPTS["rerank_vlm"] = """
You are an expert information retrieval system. Given a user query and a list of text chunks (which may contain images), rank the chunks by their relevance to the query.

Each chunk is delimited with explicit boundary markers:
- <CHUNK_START id=N> marks the beginning of chunk N
- <CHUNK_END id=N> marks the end of chunk N
- <IMAGE_START id=X relates_to_chunk=N> and <IMAGE_END id=X> mark images within chunks

Instructions:
1. Analyze each chunk's relevance to the query based on both textual content and images (when present).
2. Return ONLY the chunk IDs in order from most relevant (Rank 1) to least relevant (Rank N).
3. Use the EXACT format specified below - this is critical for parsing.

Output Format:
Your response MUST use this EXACT format (do not deviate):
<|#|>START
<|#|><Rank 1>Chunk X
<|#|><Rank 2>Chunk Y
<|#|><Rank 3>Chunk Z
<|#|>...
<|#|>END<|#|>
Continue for all chunks in relevance order.

CRITICAL: 
- Use <Rank N>Chunk X format exactly as shown (N is 1, 2, 3, etc., X is the chunk id from CHUNK_START markers)
- Do NOT include chunk text, image markers, or any other content - only the chunk number
- Rank 1 = most relevant, higher rank numbers = less relevant
- Include ALL chunks in your ranking

----------------------------------------------
Example Input:
----------------------------------------------
Query: What are motor efficiency standards?

Chunks to rank:

<CHUNK_START id=1>
Motor efficiency standards per IEC 60034-2-2 define minimum efficiency levels...
<CHUNK_END id=1>

<CHUNK_START id=2>
This figure shows water density characteristics.
<IMAGE_START id=1 relates_to_chunk=2>
[Image 1 displayed here]
<IMAGE_END id=1>
<CHUNK_END id=2>

<CHUNK_START id=3>
The Eh-star test circuit measures motor performance...
<IMAGE_START id=2 relates_to_chunk=3>
[Image 2 displayed here]
<IMAGE_END id=2>
<CHUNK_END id=3>

Example Output:
If Chunk 1 is most relevant, Chunk 3 is second, and Chunk 2 is least relevant:
<|#|>START
<|#|><Rank 1>Chunk 1
<|#|><Rank 2>Chunk 3
<|#|><Rank 3>Chunk 2
<|#|>END<|#|>

---
"""


PROMPTS["rerank_image_desc"] = """
Generate a concise 100-word technical description of this image. Focus on the key technical information, data, and visual elements that would be useful for retrieval and understanding.
"""


#%% Deduplication prompts

PROMPTS["deduplication_rank"] = """
You are a(n) {expert_persona} from the domain of {domain} working as an expert Data Curator. I have identified the following cluster of Question-Answer (QA) pairs that are highly similar or related.

Your task is to order these QA pairs from the "least similar" (most distinct/unique) to the "most similar" (most redundant/representative) relative to the core topic of the cluster. This helps in identifying the foundational questions versus the variations. Consider the domain expertise when evaluating technical significance.

IMPORTANT: Each QA pair may contain related sub-questions merged into the main question. Keep all content intact during reordering.

Cluster Candidates:
{candidates_text}

Output the exact same QA pairs, but reordered. Do not omit any text.
Format your response exactly like this:

<|#|>START<|#|>
Question<|#|><Ordered Question 1><|#|>Answer<|#|><Ordered Answer 1>
<|#|>NEXT<|#|>
Question<|#|><Ordered Question 2 (may include related sub-questions)><|#|>Answer<|#|><Ordered Answer 2 (may include related sub-answers)>
<|#|>NEXT<|#|>
Question<|#|><Ordered Question 3><|#|>Answer<|#|><Ordered Answer 3>
<|#|>END<|#|>

------------------------------------------------
Example: Complex Technical QA Pairs with Related Questions
------------------------------------------------
Input Cluster:
--- Candidate 1 ---
Question: How are the absolute and relative losses of a Power Drive System (PDS) calculated at standardized operating points, and which IEC standard defines these points? What specific formula is applied at the rated speed-torque point instead of the generic interpolation? How does interpolation between operating points work according to the standard?
Answer: Absolute PDS losses are obtained by summing motor losses (R_M) and converter losses (R_CDM) at each operating point using Formula 1. Relative losses are expressed as a percentage using Formula 2. The standardized operating points are defined in IEC 60034-2-3. At rated speed and torque, Formula 13 replaces the generic interpolation expression, as specified in IEC 61800-9-1. Interpolation follows the methods specified in the annexes of IEC 60034-2-3, ensuring consistent loss estimation across the operating range.

--- Candidate 2 ---
Question: What formulas are used to compute PDS losses, and how is the calculation modified at rated conditions?
Answer: PDS losses use R_PDS = R_M + R_CDM (Formula 1) for absolute losses. At rated speed and torque, Formula 13 provides a modified expression per IEC 61800-9-1.

--- Candidate 3 ---
Question: Which formula computes absolute PDS losses, and what adjustment applies at the rated operating point? What is the relationship between absolute and relative losses?
Answer: Absolute losses are R_PDS = R_M + R_CDM per Formula 1. At rated conditions, Formula 13 replaces standard interpolation according to IEC 61800-9-1. Relative losses are calculated by dividing absolute losses by rated power (Formula 2), expressing losses as a percentage for classification purposes.

RESPONSE:
<|#|>START<|#|>
Question<|#|>Which formula computes absolute PDS losses, and what adjustment applies at the rated operating point? What is the relationship between absolute and relative losses?<|#|>Answer<|#|>Absolute losses are R_PDS = R_M + R_CDM per Formula 1. At rated conditions, Formula 13 replaces standard interpolation according to IEC 61800-9-1. Relative losses are calculated by dividing absolute losses by rated power (Formula 2), expressing losses as a percentage for classification purposes.
<|#|>NEXT<|#|>
Question<|#|>What formulas are used to compute PDS losses, and how is the calculation modified at rated conditions?<|#|>Answer<|#|>PDS losses use R_PDS = R_M + R_CDM (Formula 1) for absolute losses. At rated speed and torque, Formula 13 provides a modified expression per IEC 61800-9-1.
<|#|>NEXT<|#|>
Question<|#|>How are the absolute and relative losses of a Power Drive System (PDS) calculated at standardized operating points, and which IEC standard defines these points? What specific formula is applied at the rated speed-torque point instead of the generic interpolation? How does interpolation between operating points work according to the standard?<|#|>Answer<|#|>Absolute PDS losses are obtained by summing motor losses (R_M) and converter losses (R_CDM) at each operating point using Formula 1. Relative losses are expressed as a percentage using Formula 2. The standardized operating points are defined in IEC 60034-2-3. At rated speed and torque, Formula 13 replaces the generic interpolation expression, as specified in IEC 61800-9-1. Interpolation follows the methods specified in the annexes of IEC 60034-2-3, ensuring consistent loss estimation across the operating range.
<|#|>END<|#|>
"""

PROMPTS["deduplication_merge"] = """
You are an expert Data Curator acting as a(n) {expert_persona} in the domain of {domain}. You have received a list of Question-Answer (QA) pairs that form a cluster of similar or duplicate content.

Your task is to create the MINIMAL set of high-quality QA pairs (>=1) that covers the complete information span of the input cluster while removing redundancy. Ensure technical accuracy appropriate for the domain of {domain}.
- If the input pairs are exact duplicates, return just one best version.
- If the input pairs cover different aspects of the same topic, merge them into fewer, more comprehensive pairs OR keep them distinct if they are sufficiently different.
- When merging, integrate related sub-questions directly into the main question text, and integrate related sub-answers into the main answer text.
- Ensure the final output is concise, accurate, and technically sound for a {expert_persona}.

Input Candidates (Ordered):
{candidates_text}

Output Format:
<|#|>START<|#|>
Question<|#|><Refined Question 1 (may include related questions)><|#|>Answer<|#|><Refined Answer 1 (may include related answers)>
<|#|>NEXT<|#|>
Question<|#|><Refined Question 2><|#|>Answer<|#|><Refined Answer 2>
<|#|>END<|#|>

------------------------------------------------
Example 1: Merge to Single QA Pair with Integrated Sub-questions
------------------------------------------------
Input Candidates (Ordered):
--- Candidate 1 ---
Question: Which formula computes absolute PDS losses, and what adjustment applies at the rated operating point? What is the relationship between absolute and relative losses?
Answer: Absolute losses are R_PDS = R_M + R_CDM per Formula 1. At rated conditions, Formula 13 replaces standard interpolation according to IEC 61800-9-1. Relative losses are calculated by dividing absolute losses by rated power (Formula 2), expressing losses as a percentage for classification purposes.

--- Candidate 2 ---
Question: What formulas are used to compute PDS losses, and how is the calculation modified at rated conditions?
Answer: PDS losses use R_PDS = R_M + R_CDM (Formula 1) for absolute losses. At rated speed and torque, Formula 13 provides a modified expression per IEC 61800-9-1.

--- Candidate 3 ---
Question: How are the absolute and relative losses of a Power Drive System (PDS) calculated at standardized operating points, and which IEC standard defines these points? What specific formula is applied at the rated speed-torque point instead of the generic interpolation? How does interpolation between operating points work according to the standard?
Answer: Absolute PDS losses are obtained by summing motor losses (R_M) and converter losses (R_CDM) at each operating point using Formula 1. Relative losses are expressed as a percentage using Formula 2. The standardized operating points are defined in IEC 60034-2-3. At rated speed and torque, Formula 13 replaces the generic interpolation expression, as specified in IEC 61800-9-1. Interpolation follows the methods specified in the annexes of IEC 60034-2-3, ensuring consistent loss estimation across the operating range.

RESPONSE:
<|#|>START<|#|>
Question<|#|>How are absolute and relative losses of a Power Drive System (PDS) calculated at standardized operating points, which formulas and standards govern these calculations, what special formula applies at the rated speed-torque point, and how are absolute losses converted to relative losses for classification purposes?<|#|>Answer<|#|>Absolute PDS losses are calculated by summing motor losses (R_M) and converter losses (R_CDM) at each operating point using Formula 1: R_PDS = R_M + R_CDM. Relative losses are expressed as a percentage of rated power using Formula 2. The standardized operating points for these calculations are defined in IEC 60034-2-3, with interpolation methods specified in its annexes. At rated speed and torque, Formula 13 replaces the generic interpolation expression to provide a more accurate loss calculation, as specified in IEC 61800-9-1. Relative losses are obtained by dividing absolute losses by the rated power (Formula 2), expressing the result as a percentage that is then used for IES class determination and efficiency rating.
<|#|>END<|#|>

------------------------------------------------
Example 2: Merge to Two QA Pairs (One Complex, One Simple)
------------------------------------------------
Input Candidates (Ordered):
--- Candidate 1 ---
Question: What optional sub-components can be integrated into a Sub-drive Input Model (SDIM) and which loss categories must be considered?
Answer: According to section B.4.5, an SDIM may incorporate an input-converter topology, input filter, line chokes, DC link, dedicated cooling elements, and auxiliary power supply. Both control losses and standby losses must be included in the efficiency assessment.

--- Candidate 2 ---
Question: Which optional sub-components are included in a Sub-drive Output Model (SDOM) and which loss categories are considered?
Answer: As defined in section B.6.2, an SDOM can contain the output inverter with its DC link, output filter, dedicated cooling elements, and auxiliary power supply. Control losses and standby losses should be accounted for in efficiency evaluation.

--- Candidate 3 ---
Question: How are SDIM and SDOM related to the BDM within a CDM architecture? What is the hierarchical relationship among SDM, SDIM, SDOM, BDM, and CDM?
Answer: Both SDIM and SDOM are parts of the BDM (Basic Drive Module). The SDIM provides input conversion while the SDOM provides output inverter function, as illustrated in Figure 1. An SDM is a generic drive module that functions as either SDIM (input) or SDOM (output). A BDM consists of one SDIM paired with one SDOM. A CDM (Complete Drive Module) incorporates the BDM plus auxiliary components like protection devices or transformers, but excludes the motor and sensors.

--- Candidate 4 ---
Question: What functional distinction exists between an SDM and an SDIM in terms of power conversion capabilities?
Answer: An SDM is a generic drive block performing AC→DC, DC→DC, or DC→AC conversion. An SDIM is a specialized SDM limited to AC→DC or DC→DC conversion and may include optional components such as AC or DC chokes, input filters, or DC-link capacitors.

RESPONSE:
<|#|>START<|#|>
Question<|#|>What are the architectural relationships and functional distinctions among SDM, SDIM, SDOM, BDM, and CDM in drive system hierarchies, what optional components can an SDIM include beyond basic power conversion, and what is the scope of power conversion for an SDIM compared to a generic SDM?<|#|>Answer<|#|>An SDM (Sub-drive Module) is a generic electrical drive module that can perform AC→DC, DC→DC, or DC→AC conversion. When an SDM functions as an input converter (limited to AC→DC or DC→DC), it is called an SDIM (Sub-drive Input Module); when it functions as an output inverter, it is called an SDOM (Sub-drive Output Module). A BDM (Basic Drive Module) consists of one SDIM paired with one SDOM, where the SDIM provides input conversion and the SDOM provides output inverter function. A CDM (Complete Drive Module) incorporates the BDM plus auxiliary components such as protection devices or transformers, but excludes the motor and sensors, as illustrated in Figure 1. An SDIM may optionally include ancillary components such as AC or DC chokes, input filters, or DC-link capacitors to enhance power quality and system performance. While a generic SDM can perform AC→DC, DC→DC, or DC→AC conversion, an SDIM is specifically limited to AC→DC or DC→DC conversion for input-stage applications.
<|#|>NEXT<|#|>
Question<|#|>What optional sub-components and loss categories must be considered when evaluating SDIM and SDOM modules?<|#|>Answer<|#|>For SDIM evaluation (section B.4.5), optional components include input-converter topology, input filter, line chokes, DC link, dedicated cooling elements, and auxiliary power supply; both control losses and standby losses must be included. For SDOM evaluation (section B.6.2), optional components include the output inverter with DC link, output filter, dedicated cooling elements, and auxiliary power supply; control losses and standby losses must also be accounted for in the efficiency assessment.
<|#|>END<|#|>
"""

#%% Metrics Evaluation Prompts (LLM-as-a-Judge)
# NOTE: multihop_reasoning and visual_dependency moved to PROMPTS_METRICS

PROMPTS["deduplication_reorganize"] = """
You are an expert Data Curator acting as a(n) {expert_persona} in the domain of {domain}. You will receive a list of merged questions and answers.

Your task is to REORGANIZE these into thematic **Question-Answer Packs** following these rules:
1.  **Group by Theme:** Cluster questions that share a specific sub-topic. Questions within a pack must be tightly related.
2.  **Balance:** Aim for optimal number of questions per pack. If questions cover distinct topics, SPLIT them into separate packs.
3.  **Synthesize Answers:** Write ONE comprehensive, self-contained answer for each pack that addresses ALL questions in that group.
4.  **Tone:** Maintain technical accuracy appropriate for a {expert_persona}.

**Output Format:**
<|#|>START<|#|>
Question<|#|><Question 1>
<Question 2...><|#|>Answer<|#|><Unified Answer>
<|#|>NEXT<|#|>
Question<|#|><Question 1...><|#|>Answer<|#|><Unified Answer>
<|#|>END<|#|>

------------------------------------------------
Example (Domain: Baking)
------------------------------------------------
Input:
Merged Questions:
1. What type of flour is best for chocolate chip cookies?
2. Should I use brown sugar or white sugar?
3. What temperature should I set the oven to?
4. How long do the cookies need to bake?
Merged Answers:
1. All-purpose flour provides the best structure.
2. A mix of both sugars creates the best texture.
3. Heat to 350°F (175°C).
4. Bake for 10-12 minutes.

RESPONSE (Split into two packs: "Ingredients" and "Process"):
<|#|>START<|#|>
Question<|#|>What type of flour is best for chocolate chip cookies?
Should I use brown sugar or white sugar?<|#|>Answer<|#|>For the best structure and texture, use all-purpose flour and a mixture of both brown and white sugar.
<|#|>NEXT<|#|>
Question<|#|>What temperature should I set the oven to?
How long do the cookies need to bake?<|#|>Answer<|#|>Preheat the oven to 350°F (175°C) and bake the cookies for 10-12 minutes.
<|#|>END<|#|>
"""

PROMPTS_METRICS["multimodal_faithfulness_vlm"] = """
You are evaluating the faithfulness of an answer given the provided context (text and images).

**Inputs:**
Question: {question}
Answer: {answer}
Context (Text & Visuals): {context}

**Analysis Criteria:**
1.  **Text Support:** Is the answer verified by the textual content in the provided context?
2.  **Visual Support:** Is the answer verified by visual elements (images, charts, tables) in the provided context?
3.  **Faithfulness:** Does the answer contain any information *not* present in the source (hallucinations)?

**Expected Output Format:**
<|#|>START<|#|>
Text Supported<|#|><YES/NO><|#|>Visual Supported<|#|><YES/NO/NA><|#|>Faithfulness Score<|#|><0.0-1.0><|#|>Explanation<|#|><Brief single-sentence justification>
<|#|>END<|#|>

------------------------------------------------
Example (Domain: Consumer Electronics)
------------------------------------------------
Context:
The battery life of the Model X speaker, which is water-resistant up to 1 meter depth, is rated for 12 hours of continuous playback.

Question:
How long does the battery last, and is the speaker water-resistant?

Answer:
The battery lasts for 12 hours, and the device is water-resistant up to 1 meter.

RESPONSE:
<|#|>START<|#|>
Text Supported<|#|>YES<|#|>Visual Supported<|#|>NA<|#|>Faithfulness Score<|#|>1.0<|#|>Explanation<|#|>The text confirms the 12-hour battery life, and the visual chart confirms the water resistance depth of 1 meter.
<|#|>END<|#|>
"""

PROMPTS_METRICS["multimodal_answer_quality_vlm"] = """
You are evaluating the quality of an answer given multimodal context (text and images).

**Inputs:**
Question: {question}
Answer: {answer}
Context: {context_text_and_images}

**Evaluation Criteria:**
1.  **COMPLETENESS:** Does the answer fully address the question using all relevant context?
2.  **ACCURACY:** Is the answer factually correct based on the provided context?
3.  **VISUAL_INFO_USED:** Does the answer incorporate information from visual elements (if relevant to the question)?

**Expected Output Format:**
<|#|>START<|#|>
Completeness<|#|><0.0-1.0><|#|>Accuracy<|#|><0.0-1.0><|#|>VisualInfo<|#|><YES/NO/NA><|#|>OverallScore<|#|><0.0-1.0><|#|>Reasoning<|#|><Brief single-sentence reasoning>
<|#|>END<|#|>

------------------------------------------------
Example (Domain: Weather Forecasting)
------------------------------------------------
Context:
[Text]: "Saturday will be sunny with a high of 75°F."
[Visual]: (A 5-day forecast chart showing a Rain Cloud icon and '60°F' for Sunday).

Question:
What is the weather outlook for the entire weekend?

Answer:
Saturday is expected to be sunny and 75°F, while Sunday will be rainy with a high of 60°F.

RESPONSE:
<|#|>START<|#|>
Completeness<|#|>1.0<|#|>Accuracy<|#|>1.0<|#|>Visual Info Used<|#|>NA<|#|>Overall Score<|#|>1.0<|#|>Reasoning<|#|>The answer accurately combines the text data for Saturday and extracts the visual data for Sunday to provide a complete response.
<|#|>END<|#|>
"""

# NOTE: context_necessity_without and context_necessity_verify moved to PROMPTS_METRICS

#%% Optimized Metrics Prompts (for metrics_optimized.py)
# These prompts are designed for MINIMAL LLM calls by batching claims/contexts


# ============================================================================
# PREPARATION PROMPT (1 call extracts claims + generates reverse questions)
# ============================================================================
PROMPTS_METRICS["prepare_qa"] = """

You are a QA analysis assistant. Analyze the following QA pair and extract information needed for evaluation.

**Inputs:**
QUESTION: {question}
ANSWER: {answer}
REFERENCE (Ground Truth): {reference}

**Tasks:**
1.  **Concept Hops:** Identify the logical steps/concepts in the QUESTION and connect them with arrows (Concept A --> Concept B). The number of hops is the number of arrows.
2.  **Answer Claims:** Extract ALL atomic factual assertions from the ANSWER.
3.  **Reference Claims:** Extract ALL atomic factual assertions from the REFERENCE.
4.  **Reverse Questions:** Generate {num_reverse_questions} diverse questions that the provided ANSWER could plausibly answer.

**Expected Output Format:**
<|#|>START<|#|>
ConceptHops<|#|><concept1 --> concept2 ...><|#|>Answer Claims<|#|><claim 1><|#|><claim 2><|#|>...<|#|>Reference Claims<|#|><claim 1><|#|><claim 2><|#|>...<|#|>Reverse Questions<|#|><question 1><|#|><question 2><|#|>...
<|#|>END<|#|>

------------------------------------------------
Example (Domain: Cooking)
------------------------------------------------
QUESTION: What is the standard ratio of oil to vinegar for a classic vinaigrette, and what acts as the emulsifier?
ANSWER: A classic vinaigrette uses a ratio of 3 parts oil to 1 part vinegar. Dijon mustard is commonly added to act as the emulsifying agent to keep the mixture blended.
REFERENCE: The traditional vinaigrette ratio is 3:1 (oil:vinegar). Mustard, specifically Dijon, is often used to emulsify the dressing.

RESPONSE:
<|#|>START<|#|>
Concept Hops<|#|>Classic Vinaigrette --> Oil/Vinegar Ratio --> Emulsifier<|#|>Answer Claims<|#|>- A classic vinaigrette uses a ratio of 3 parts oil to 1 part vinegar
- Dijon mustard is commonly added
- Dijon mustard acts as the emulsifying agent<|#|>Reference Claims<|#|>- The traditional vinaigrette ratio is 3:1 (oil:vinegar)
- Mustard (specifically Dijon) is used to emulsify<|#|>Reverse Questions<|#|>- What is the ingredient ratio for a basic vinaigrette?
- Which ingredient helps emulsify a vinaigrette dressing?
- What is the function of Dijon mustard in a vinaigrette?
<|#|>END<|#|>

"""

# ============================================================================
# FAITHFULNESS PROMPT (batch verify ALL answer claims at once)
# ============================================================================
PROMPTS_METRICS["faithfulness"] = """
You are a faithfulness evaluator. Determine if each claim from the answer can be inferred from the given context.

**Inputs:**
CONTEXT: {context}
CLAIMS TO VERIFY: {claims}

**Task:**
For each claim, respond with ONLY "SUPPORTED" or "NOT_SUPPORTED".
- **SUPPORTED:** The claim is explicitly backed by the provided text.
- **NOT_SUPPORTED:** The claim is missing from the text, contradicts the text, or relies on external knowledge not present in the chunk.

**Expected Output Format:**
<|#|>START<|#|>
Evaluations<|#|>CLAIM_1: <SUPPORTED/NOT_SUPPORTED>
CLAIM_2: <SUPPORTED/NOT_SUPPORTED>
...
<|#|>END<|#|>

------------------------------------------------
Example (Domain: Dog Breeds)
------------------------------------------------
CONTEXT:
The Golden Retriever is a Scottish breed of retriever dog of medium size. They are broadly known for their gentle and affectionate nature, making them excellent family pets.

CLAIMS TO VERIFY:
CLAIM_1: Golden Retrievers originated in Scotland.
CLAIM_2: They are considered a large-sized breed.
CLAIM_3: They have a gentle nature.
CLAIM_4: They require daily grooming.

RESPONSE:
<|#|>START<|#|>
Evaluations<|#|>CLAIM_1: SUPPORTED
CLAIM_2: NOT_SUPPORTED
CLAIM_3: SUPPORTED
CLAIM_4: NOT_SUPPORTED
<|#|>END<|#|>
"""

# ============================================================================
# CONTEXT RECALL PROMPT (batch verify ALL reference claims at once)
# ============================================================================
PROMPTS_METRICS["context_recall"] = """
You are a context recall evaluator. Determine if each claim from the reference/ground truth can be attributed to the retrieved context.

**Inputs:**
CONTEXT: {context}
REFERENCE CLAIMS TO VERIFY: {claims}

**Task:**
For each claim, respond with ONLY "ATTRIBUTED" or "NOT_ATTRIBUTED".
- **ATTRIBUTED:** The claim can be fully inferred or found within the provided context text.
- **NOT_ATTRIBUTED:** The claim is NOT present in the provided context (even if it is factually true in the real world).

**Expected Output Format:**
<|#|>START<|#|>
Evaluations<|#|>CLAIM_1: <ATTRIBUTED/NOT_ATTRIBUTED>
CLAIM_2: <ATTRIBUTED/NOT_ATTRIBUTED>
...
<|#|>END<|#|>

------------------------------------------------
Example (Domain: History)
------------------------------------------------
CONTEXT:
The RMS Titanic sank in the North Atlantic Ocean in the early morning hours of 15 April 1912, four days into her maiden voyage from Southampton to New York City.

REFERENCE CLAIMS TO VERIFY:
CLAIM_1: The Titanic sank in the North Atlantic.
CLAIM_2: The ship was traveling to New York City.
CLAIM_3: The disaster was caused by striking an iceberg.

RESPONSE:
<|#|>START<|#|>
Evaluations<|#|>CLAIM_1: ATTRIBUTED
CLAIM_2: ATTRIBUTED
CLAIM_3: NOT_ATTRIBUTED
<|#|>END<|#|>
...
"""

# ============================================================================
# CONTEXT PRECISION PROMPT (batch evaluate ALL contexts at once)
# ============================================================================
PROMPTS_METRICS["context_precision"] = """You are a context precision evaluator. For each context chunk, determine if it is RELEVANT or NOT_RELEVANT to answering the question, given the reference answer.

**Inputs:**
QUESTION: {question}
REFERENCE ANSWER: {reference}
CONTEXT CHUNKS:
{contexts}

**Task:**
For each context chunk, respond with ONLY "RELEVANT" or "NOT_RELEVANT".
- **RELEVANT:** The chunk contains information necessary to derive the Reference Answer.
- **NOT_RELEVANT:** The chunk contains information unrelated to the specific facts in the Reference Answer (even if the topic is similar).

**Expected Output Format:**
<|#|>START<|#|>
Evaluations<|#|>CHUNK_1: <RELEVANT/NOT_RELEVANT>
CHUNK_2: <RELEVANT/NOT_RELEVANT>
...
<|#|>END<|#|>

------------------------------------------------
Example (Domain: Coffee Brewing)
------------------------------------------------
QUESTION: What are the key steps to brewing French Press coffee?
REFERENCE ANSWER: To brew French Press, add coarse ground coffee, pour hot water over the grounds, let it steep for 4 minutes, and then slowly press the plunger down.

CONTEXT CHUNKS:
CHUNK_1:
For a French Press, use a coarse grind setting. Pour water just off the boil over the coffee and allow it to steep for 4 minutes before pressing the plunger.
CHUNK_2:
Espresso machines use high pressure (9 bars) to force water through tightly packed, fine coffee grounds to create a crema.
CHUNK_3:
Coffee beans are actually the seeds found inside the cherries of the Coffea plant, which is native to tropical Africa.

RESPONSE:
<|#|>START<|#|>
Evaluations<|#|>CHUNK_1: RELEVANT
CHUNK_2: NOT_RELEVANT
CHUNK_3: NOT_RELEVANT
<|#|>END<|#|>
"""

# ============================================================================
# MULTIMODAL FAITHFULNESS PROMPT (VLM - batch verify ALL claims with images)
# ============================================================================
PROMPTS_METRICS["multimodal_faithfulness"] = """
You are a multimodal faithfulness evaluator. Verify if EACH claim from the answer can be inferred from the provided context (text AND/OR images).

**Inputs:**
QUESTION: {question}
ANSWER: {answer}
CLAIMS TO VERIFY: {claims}
CONTEXT (Text + Image): {context}

**Task:**
For EACH claim, determine:
1.  **Status:** Is it SUPPORTED or NOT_SUPPORTED by the context?
2.  **Source:** If supported, is it from TEXT, IMAGE, or BOTH?

**Expected Output Format:**
<|#|>START<|#|>
Evaluations<|#|>CLAIM_1: <SUPPORTED/NOT_SUPPORTED> | SOURCE: <TEXT/IMAGE/BOTH/NONE>
CLAIM_2: ...
...<|#|>Summary<|#|>TEXT_GROUNDED: <YES/NO>
VISUAL_GROUNDED: <YES/NO/NA>
SUPPORTED_COUNT: <Integer>
TOTAL_CLAIMS: <Integer>
<|#|>END<|#|>

------------------------------------------------
Example:
------------------------------------------------
CONTEXT (Text + Image):
Image: {{file_loc1}}
Technical Summary: This figure presents the relationship between the density (ρ) and specific heat capacity (cp) of pure water across a range of temperatures from 0°C to 80°C.

QUESTION: At what temperature does water have minimum specific heat capacity?
ANSWER: Water has minimum specific heat capacity at approximately 35°C. The specific heat capacity value at this point is about 4.178 kJ/(kg·K).

CLAIMS TO VERIFY:
CLAIM_1: Water has minimum specific heat capacity at approximately 35°C
CLAIM_2: The specific heat capacity at this point is about 4.178 kJ/(kg·K)
CLAIM_3: The density of water is constant across all temperatures

RESPONSE:
<|#|>START<|#|>
Evaluations<|#|>CLAIM_1: SUPPORTED | SOURCE: IMAGE
CLAIM_2: SUPPORTED | SOURCE: BOTH
CLAIM_3: NOT_SUPPORTED | SOURCE: NONE<|#|>Summary<|#|>TEXT_GROUNDED: YES
VISUAL_GROUNDED: YES
SUPPORTED_COUNT: 2
TOTAL_CLAIMS: 3
<|#|>END<|#|>
"""

# ============================================================================
# MULTIMODAL RELEVANCE PROMPT (VLM - generate reverse questions + evaluate)
# ============================================================================
PROMPTS_METRICS["multimodal_relevance"] = """
You are a multimodal relevance evaluator. Generate {num_questions} questions that the given answer could plausibly be answering, then evaluate relevance.

**Inputs:**
ANSWER: {answer}
CONTEXT: (text and images provided below) {context}

**Task:**
1.  Generate {num_questions} diverse questions that this answer could address.
2.  For each generated question, indicate if it uses TEXT context, IMAGE context, or BOTH.
3.  Assess overall context utilization and relevance.

**Expected Output Format:**
<|#|>START<|#|>
Generated Questions<|#|>Q1: <question> | USES: <TEXT/IMAGE/BOTH>
Q2: <question> | USES: <TEXT/IMAGE/BOTH>
...<|#|>Context Utilization<|#|>USES_TEXT: <YES/NO>
USES_IMAGES: <YES/NO/NA>
RELEVANCE_SCORE: <0.0-1.0>
<|#|>END<|#|>

------------------------------------------------
Example:
------------------------------------------------
CONTEXT (Text + Image):
Image: {{file_loc2}}
Technical Summary: This schematic illustrates the Eh-star test circuit for three-phase asynchronous motors. It shows ammeters, voltmeters, wattmeters, a switchable resistor, and their connections.
ANSWER: The wattmeters in the Eh-star test circuit measure the active power input to each phase, while the voltmeters measure phase voltages for calculating power factor.

RESPONSE:
<|#|>START<|#|>
Generated Questions<|#|>Q1: What do the wattmeters measure in the Eh-star test circuit? | USES: IMAGE
Q2: How are voltmeters used to calculate power factor in motor testing? | USES: BOTH
Q3: What instruments are connected in the Eh-star circuit configuration? | USES: IMAGE<|#|>Context Utilization<|#|>USES_TEXT: YES
USES_IMAGES: YES
RELEVANCE_SCORE: 0.85
<|#|>END<|#|>
"""

# ============================================================================
# CONTEXT NECESSITY PROMPTS (anti-parametric bias)
# ============================================================================
PROMPTS_METRICS["context_necessity_without"] = """
You are an expert assistant. Answer the following question using ONLY your general knowledge. Do NOT make up specific facts.

**Input:**
QUESTION: {question}

**Instruction:**
If you cannot answer confidently without additional context (specific documents, private data), respond strictly with: "CANNOT_ANSWER".

**Expected Output Format:**
<|#|>START<|#|>
Answer<|#|><Your Answer Text OR "CANNOT_ANSWER">
<|#|>END<|#|>

------------------------------------------------
Example (Domain: General Knowledge)
------------------------------------------------
QUESTION: How many legs does a spider typically have?

RESPONSE:
<|#|>START<|#|>
Answer<|#|>A spider typically has 8 legs.
<|#|>END<|#|>
"""

PROMPTS_METRICS["context_necessity_verify"] = """
You are an Answer Comparator. Compare the model's answer to the ground truth answer to determine accuracy.

**Inputs:**
GROUND TRUTH: {ground_truth}
MODEL ANSWER: {model_answer}

**Evaluation Criteria:**
- **Status: YES** (The model answer is semantically correct and covers all key information in the ground truth).
- **Status: PARTIAL** (The model answer is factually correct but misses some key details present in the ground truth).
- **Status: NO** (The model answer is factually incorrect, contradicts the ground truth, or is missing).

**Expected Output Format:**
<|#|>START<|#|>
Status<|#|><YES/PARTIAL/NO>
<|#|>END<|#|>

------------------------------------------------
Example (Domain: General Knowledge)
------------------------------------------------
GROUND TRUTH: The primary colors of pigment are red, yellow, and blue.
MODEL ANSWER: The three primary pigment colors are blue, red, and yellow.

RESPONSE:
<|#|>START<|#|>
Status<|#|>YES
<|#|>END<|#|>
"""

# ============================================================================
# MULTIHOP REASONING PROMPT
# ============================================================================
PROMPTS_METRICS["multihop_reasoning"] = """
You are a Logic Analyst. Determine if answering the specific question requires multi-hop reasoning by combining information from multiple context chunks.

**Inputs:**
CONTEXTS: {contexts}
QUESTION: {question}
ANSWER: {answer}

**Evaluation Tasks:**
1.  **HOP_COUNT:** How many distinct pieces of information must be combined? (1 = single fact/direct lookup, 2+ = multi-hop/inference).
2.  **REASONING_SCORE:** Rate the complexity from 0.0 (trivial/direct) to 1.0 (complex multi-step deduction).
3.  **BRIDGE_ENTITY:** Identify the specific concept or entity that connects the two pieces of information (write "None" if single-hop).

**Expected Output Format:**
<|#|>START<|#|>
Hop Count<|#|><Integer><|#|>Reasoning Score<|#|><0.0-1.0><|#|>Bridge Entity<|#|><Entity Name or "None"><|#|>Explanation<|#|><Brief justification>
<|#|>END<|#|>

------------------------------------------------
Example (Domain: Lost & Found)
------------------------------------------------
CONTEXTS:
Chunk A: "To unlock the basement safe, you need the Silver Key."
Chunk B: "The Silver Key is hidden inside the blue flower vase in the hallway."

QUESTION: Where should I look to find the object needed to open the basement safe?
ANSWER: You should look inside the blue flower vase in the hallway.

RESPONSE:
<|#|>START<|#|>
Hop Count<|#|>2<|#|>Reasoning Score<|#|>0.8<|#|>Bridge Entity<|#|>Silver Key<|#|>Explanation<|#|>To answer, one must first identify the necessary object ("Silver Key") from Chunk A, and then use Chunk B to find the location of that specific object ("Blue Vase").
<|#|>END<|#|>
"""

# ============================================================================
# VISUAL DEPENDENCY PROMPT (blind test - text only)
# ============================================================================
PROMPTS_METRICS["visual_dependency"] = """You are given ONLY text context (no images). Determine if you can fully answer the question.

TEXT CONTEXT:
{contexts}

QUESTION: {question}

If you can answer completely using ONLY the text above, provide your answer.
If you CANNOT answer because visual information (figures, diagrams, images) is missing, respond with: MISSING_VISUAL

YOUR RESPONSE:

------------------------------------------------
Example 1: Requires Visual (MISSING_VISUAL)
------------------------------------------------
TEXT CONTEXT:
Figure 3 shows the torque-speed characteristic curve for the motor.

QUESTION: What is the breakdown torque value shown in Figure 3?

Response: MISSING_VISUAL

------------------------------------------------
Example 2: Can Answer from Text
------------------------------------------------
TEXT CONTEXT:
Figure 3 shows the torque-speed characteristic curve. The breakdown torque is 2.5 times the rated torque, occurring at approximately 80% of synchronous speed.

QUESTION: What is the breakdown torque relative to rated torque?

Response: The breakdown torque is 2.5 times the rated torque, occurring at approximately 80% of synchronous speed.
"""
