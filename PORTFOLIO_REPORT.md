Student engineering project output only. Not real clinical, medical, or regulatory guidance.

# mRNA-VIP | Educational ML and RAG Demonstration

## 1. Executive Summary

The **mRNA Vaccine Clinical Safety and Dosage Intelligence Platform (mRNA-VIP)** is an individual applied-AI learning project demonstrating a modular workflow for synthetic patient-record validation, reaction-risk classification, NumPy-based self-attention, retrieval-augmented generation (RAG), structured report generation, and dashboard visualization.

The project combines a synthetic patient dataset with an educational course reference. It demonstrates data preprocessing, supervised machine learning, scaled dot-product self-attention, semantic and lexical passage retrieval, Pydantic-based report validation, and visualization of saved module outputs.

The platform is organized into four main modules:

1. **Data Pipeline and Risk Classification:** Data validation, preprocessing, Logistic Regression, and Random Forest.
2. **Self-Attention Visualization:** NumPy-based Query, Key, and Value projections with scaled dot-product attention.
3. **Retrieval-Augmented Guideline Search:** Text chunking, embeddings, persistent ChromaDB indexing, and ranked retrieval.
4. **Structured Report Generation:** Pydantic validation, educational report generation, and JSON/Markdown output.

The project is intended to demonstrate AI engineering concepts and reproducible workflows. It is not clinically validated and must not be used to determine real vaccine eligibility, dosing, patient management, or regulatory decisions.

**Educational disclaimer:** The dataset, source material, model predictions, retrieved passages, attention weights, and generated reports are not validated clinical or regulatory guidance.

## 2. System Architecture

The platform consists of interconnected data-processing, model, attention, retrieval, reporting, and visualization components.

```text
Synthetic Patient Records
          |
          v
Data Validation and Cleaning
          |
          v
Cleaned Dataset
          |
          v
Risk Model Training and Evaluation
(Logistic Regression / Random Forest)
          |
          v
Saved Model and Evaluation Results
          |
          +--------------------------------+
          |                                |
          v                                v
Illustrative Nucleotide Sequence    Educational Reference Text
          |                                |
          v                                v
NumPy Self-Attention                Text Chunking and Embeddings
          |                                |
          v                                v
Attention Matrix and Heatmap        Persistent Retrieval Index
                                           |
                                           v
                                  Ranked Retrieved Passages
                                           |
                     Model Output + Retrieved Educational Passages
                                           |
                                           v
                                 Structured Report Generation
                                           |
                                           v
                                   Pydantic Schema Validation
                                           |
                                           v
                                    JSON and Markdown Reports
                                           |
                                           v
                                    Three-Panel Dashboard
```

The pipeline runner, `run_all.py`, executes the main modules in dependency order and stops if a pipeline step fails. Intermediate artifacts are saved under `data/` and `outputs/` so individual results can be inspected.

The dashboard combines saved model evaluation data, a computational self-attention demonstration, and recorded retrieval and report-generation timings.

The architecture separates data preparation, model development, attention computation, retrieval, structured output generation, and visualization to support modular execution and testing.

## 3. Module 1: Data Pipeline and Risk Classification

### 3.1 Data Preparation

The supplied dataset contains 500 records and seven required columns:

- `Patient_ID`
- `Age`
- `Biomarker_Level`
- `Pre_Existing_Conditions`
- `Vaccine_Dose_mcg`
- `Prior_Reaction_History`
- `Reaction_Score`

The data pipeline validates required columns, checks patient identifiers, verifies numeric fields, validates the binary target, and checks prior-reaction-history categories.

Missing `Biomarker_Level` values are imputed using the median. Missing `Prior_Reaction_History` values are imputed using the most frequent observed category.

The cleaned dataset is saved separately from the original raw CSV, and a validation report records missing-value counts and preprocessing details.

The target `Reaction_Score` is a binary label in the supplied synthetic dataset. It must not be interpreted as a clinically validated patient-risk label.

### 3.2 Model Training and Evaluation

Logistic Regression and Random Forest were evaluated on a stratified train-test split. The following metrics are from the saved model comparison results.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.7100 | 0.6200 |
| Precision | 0.7083 | 0.6122 |
| Recall | 0.6939 | 0.6122 |
| F1-score | 0.7010 | 0.6122 |
| ROC-AUC | 0.7911 | 0.7345 |

Logistic Regression achieved a higher F1-score and ROC-AUC on the evaluated split and was selected by the implemented model-selection procedure.

The model evaluation results, serialized model artifacts, classification reports, confusion matrices, and related figures are saved under `outputs/`.

These results describe performance on a small synthetic dataset and a particular train-test split. They do not establish clinical accuracy, external validity, or generalization to real patient populations. Independent validation on suitable external data would be required to support stronger performance claims.

## 4. Module 2: Self-Attention Visualization

The attention module demonstrates scaled dot-product self-attention using NumPy and the illustrative mRNA nucleotide sequence `ACGUACGUACGGCUA`. Each nucleotide is represented by a numerical embedding, enabling the sequence to be processed using matrix operations.

The module generates three projections: Query (Q), Key (K), and Value (V). Queries represent positions seeking information, while keys represent positions against which queries are compared. Multiplying the Query matrix by the transpose of the Key matrix produces attention scores that measure the relative compatibility between sequence positions. These scores are scaled by the square root of the key-vector dimension to control their magnitude.

A row-wise softmax function converts the scaled scores into non-negative attention weights that sum to approximately one for each row. The resulting weights are multiplied by the Value matrix to produce weighted representations containing information from different sequence positions.

The attention matrix is visualized as a heatmap saved to:

`outputs/figures/attention_heatmap.png`

Rows represent the positions attending to the sequence, while columns represent the positions receiving attention. Higher weights appear more prominent, making the distribution of attention across positions easier to inspect.

The nucleotide embeddings and projection weights are randomly initialized computational components, not parameters learned from biological training data. Therefore, this visualization demonstrates the mechanics of self-attention only. It does not establish biological significance, molecular behavior, or clinical insight.

## 5. Module 3: Retrieval-Augmented Guideline Search

### 5.1 Implementation

The retrieval module divides the educational reference into text chunks and ranks passages using semantic similarity, lexical relevance, and heuristic query-intent adjustments.

The persistent retrieval artifacts are stored under:

`outputs/chroma_store/`

The retrieval index uses the `sentence-transformers/all-MiniLM-L6-v2` embedding model and a persistent ChromaDB vector store. The index metadata records the source file, chunking configuration, embedding model, distance metric, and retrieval-ranking approach.

The retrieval evaluation is documented in:

`outputs/retrieval_tests.md`

The evaluation includes five representative questions covering:

- Refrigerated storage
- High-dose secondary review
- Longer monitoring
- Adverse-event documentation
- Temperature excursions

Each query returns three ranked passages.

### 5.2 Retrieval Evaluation Results

The following table summarizes the recorded retrieval evaluation.

| Query | Top-ranked source section | Cosine similarity | Ranking score |
|---|---|---:|---:|
| Refrigerated storage | Section 1 — Storage and Handling Requirements | 0.5558 | 0.6544 |
| High-dose secondary review | Section 2 — Dosing Guidance by Age Band | 0.5063 | 0.6159 |
| Longer monitoring | Section 3 — Contraindications and Warnings | 0.3492 | 0.4461 |
| Adverse-event documentation | Section 4 — Adverse Event Documentation | 0.7609 | 0.7842 |
| Temperature excursion | Section 1 — Storage and Handling Requirements | 0.3134 | 0.5090 |

For the high-dose query, Section 3 ranked a close second, with a ranking score of 0.6137. This demonstrates why the ranking score alone should not be used to determine whether a result is factually correct.

For the longer-monitoring query, the retrieved passage was inspected and contained the two monitoring durations stated in the educational source: at least 15 minutes after administration, or 30 minutes for individuals with a documented history of an allergic reaction.

For the temperature-excursion query, the first two retrieved passages contained two relevant actions described in the source: documenting and reviewing the excursion, and quarantining the affected batch pending quality review.

The dedicated regression test checks that both actions occur within the top two passages.

The five recorded retrieval latencies ranged from approximately 0.0236 to 0.0341 seconds during this evaluation run. These values are observations from one run and are not guaranteed performance benchmarks.

### 5.3 Chunking Configuration and Limitations

The original project target specified 300–500-token chunks with approximately 50-token overlap.

The selected embedding model has a maximum sequence length of 256 tokens. Consequently, the effective chunk size was limited to 254 tokens. The six generated chunks ranged from 45 to 252 tokens.

The configured overlap is applied when splitting long sentences but is not consistently maintained between all sentence-packed chunks.

Therefore, the implementation provides persistent retrieval and retrieval evaluation, but does not fully satisfy the original chunk-size and overlap targets.

The evaluation covers only a small set of predefined questions. Semantic similarity and lexical relevance do not guarantee that a retrieved passage fully or correctly answers a question. Intent bonuses are heuristic rules and may not generalize to unseen queries.

The retrieval regression test verifies the presence of expected text in ranked results. It does not independently verify the scientific, clinical, or regulatory accuracy of the educational source.

The reference document is a course-project artifact and must not be treated as an authoritative clinical guideline.

## 6. Module 4: Structured Report Generation

The report-generation module combines saved risk-model outputs and retrieved educational reference passages to construct a structured report. Pydantic validates the required fields and schema constraints.

The implementation supports two report-generation paths:

**1. Offline rule-based generation**

Generates a structured educational report using locally available model outputs and retrieved reference passages, without requiring a live LLM API call.

**2. LLM-based generation**

Implements an OpenAI-compatible LangChain integration with structured-output validation and a single retry after a validation failure. The live API integration is implemented but has not been tested with an actual API call.

The latest verified sample report was generated using the offline rule-based path.

### 6.1 Generated Report Artifacts

The validated sample report is saved in two formats:

- `outputs/reports/sample_report.json`
- `outputs/reports/sample_report.md`

Pipeline timing information is saved to:

`outputs/report_generation_latency.json`

The offline timing record distinguishes retrieval-stage time, model-scoring time, report-generation time, and total pipeline time. These measurements describe recorded local execution and are not live LLM API performance benchmarks.

### 6.2 Structured Output Schema

The report uses the `ClinicalSafetyReport` Pydantic model. It contains five output fields:

| Field | Data type | Purpose |
|---|---|---|
| `Eligible_For_Vaccine` | Boolean | Demonstration eligibility output |
| `Recommended_Dose_mg` | Float | Candidate dose value represented in milligrams |
| `Reasoning_Steps` | List of strings | Explanation of the generated assessment |
| `FDA_Safety_Warnings_Cited` | List of strings | Warning statements extracted from retrieved educational passages |
| `Disclaimer` | String | Educational-use disclaimer |

The saved sample report was checked against the expected field names and validated successfully using the Pydantic schema.

### 6.3 Complete Generated JSON Report

The following is the actual structured report generated by the offline report-generation pipeline and saved to `outputs/reports/sample_report.json`. It demonstrates the Pydantic-compatible output schema, including eligibility status, candidate dose conversion, reasoning steps, safety-warning references, and the mandatory educational disclaimer.

```json
{
  "Eligible_For_Vaccine": false,
  "Recommended_Dose_mg": 0.0835,
  "Reasoning_Steps": [
    "The experimental classifier returned a 'Low' label with a model score of 0.252. This score is not a clinically validated estimate of individual risk.",
    "The source record lists a candidate dose of 83.5 micrograms (0.0835 mg). This is a unit conversion only, not a dose recommendation.",
    "The recorded biomarker level is 4.530, but the supplied reference does not define a numerical threshold for an elevated biomarker. Its status cannot be classified reliably by this prototype.",
    "The supplied course-project reference is not a verified clinical protocol. A qualified clinician must use current, product-specific official guidance for real medical decisions."
  ],
  "FDA_Safety_Warnings_Cited": [
    "section 3 contraindications and warnings Absolute contraindications include a known severe allergic reaction to a prior dose of the same product or to any component of the formulation, and an active unresolved severe adverse reaction from a previous dose that has not been cleared by an allergist or treating physician.",
    "Secondary review is required for three or more relevant pre-existing conditions, a prior moderate reaction to an mRNA-based product, a candidate dose exceeding 100 micrograms combined with an elevated biomarker level, or age under 5 years combined with any pre-existing condition.",
    "Patients should be monitored for at least 15 minutes after administration, or 30 minutes for patients with a documented history of any allergic reaction.",
    "Localized injection-site reactions do not by themselves require secondary review.",
    "Systemic reactions lasting more than 48 hours should be documented and reported.",
    "section 2 dosing guidance by age band Dose selection for mRNA therapeutic products is generally determined by age, body weight where applicable, and the specific clinical indication."
  ],
  "Disclaimer": "Student engineering project output only. Not real clinical, medical, or regulatory guidance."
}
```

**Validation:** All five required fields are present, with four reasoning entries, six warning references, and the exact project disclaimer. The saved JSON is the source artifact for this example.

**Important limitation:** This is an educational software demonstration using supplied project data and references. It is not a validated clinical report or a real vaccine eligibility or dosing recommendation.

### 6.4 Data Grounding and Field Interpretation

The report-generation workflow uses two principal inputs:

1. Module 1: saved risk-model output and patient-record information.
2. Module 3: retrieved passages from the educational reference document.

The report fields have the following intended roles:

- `Eligible_For_Vaccine`: a demonstration output produced by the report-generation logic. It is not a real-world eligibility determination.
- `Recommended_Dose_mg`: a candidate numerical value converted into milligrams. It must not be interpreted as a dosing recommendation.
- `Reasoning_Steps`: records the reasoning used by the educational report-generation workflow.
- `FDA_Safety_Warnings_Cited`: contains warning statements selected from retrieved educational passages.
- `Disclaimer`: communicates the non-clinical and educational purpose of the project.

The offline implementation applies its own explicit-dose checks and secondary-review logic. These are software demonstration rules, not validated medical thresholds or dosing policies.

### 6.5 Validation and Error Handling

The Pydantic schema validates the expected field names and types, including a non-negative numerical dose value and non-empty reasoning and warning lists.

The report-generation implementation includes validation retry behavior. The LLM-based path attempts structured generation again following a Pydantic validation failure.

The saved sample report was verified with the following results:

| Validation check | Result |
|---|---|
| Exact expected field names | Passed |
| Pydantic schema validation | Passed |
| Eligibility field | `false` |
| Candidate dose value | `0.0835 mg` |
| Reasoning entries | 4 |
| Warning entries | 6 |
| Exact educational disclaimer | Passed |
| Matching Markdown report | Passed |

These results verify the saved offline report artifact. They do not demonstrate successful execution of the live LLM integration.

### 6.6 Limitations

The offline report generator uses deterministic software logic and retrieved text. The LLM integration is implemented but remains unverified through an actual API call.

Schema validation confirms structural correctness, not factual correctness, complete source grounding, or clinical appropriateness. The project also does not establish that retrieved educational warnings are authoritative or suitable for real-world decisions.

Any eligibility, dose-conversion, or risk-related fields in the generated report are demonstration outputs. They must not be interpreted as real-world medical recommendations or dosing policies.

## 7. Dashboard and Visual Outputs

The dashboard is generated by `src/dashboard.py` and saved to:

`outputs/figures/dashboard.png`

The latest dashboard generation completed successfully. The image is configured for 200 DPI, exceeding the project's 150 DPI output requirement.

The dashboard contains three panels:

1. **Risk classifier ROC curve:** Displays the evaluated model's ROC curve and AUC.
2. **Self-attention heatmap:** Visualizes attention weights across positions in an illustrative nucleotide sequence.
3. **Pipeline stage latency:** Compares recorded RAG retrieval time with offline rule-based report-generation time.

The latency comparison represents measurements from the offline execution workflow. It does not measure live LLM API response time or establish production performance.

The dashboard includes an educational-use disclaimer and is a static PNG visualization, not an interactive application.

Additional saved figures include:

- Baseline ROC curves
- Logistic Regression confusion matrix
- Random Forest confusion matrix
- Feature-correlation heatmap
- Numeric feature distributions
- Reaction-score distribution
- Reaction-rate summaries
- Self-attention heatmap

These figures support inspection of the dataset and software evaluation outputs.

The attention heatmap represents a computational demonstration and does not establish biological significance or clinical insight.

## 8. Testing and Quality Assurance

The latest verified automated test run, performed on **4 November 2026**, completed successfully with **30 tests passed and zero failures**.

The test suite covers:

- Data cleaning and validation
- Self-attention calculations and output validation
- Pydantic schema validation and disclaimer enforcement
- Text chunking and retrieval input validation
- Persistent RAG index loading and retrieval behavior
- Structured report generation and validation retries
- Risk-model preprocessing, training, evaluation, and serialization

### 8.1 Test Results

| Test module | Tests passed |
|---|---:|
| `test_project.py` | 15 |
| `test_rag_retrieval.py` | 2 |
| `test_report_generator.py` | 6 |
| `test_risk_model.py` | 7 |
| **Total** | **30** |

The complete test suite was executed using:

```powershell
python -m pytest -v
```

**Result:** 30 passed in 219.73 seconds (3 minutes 39 seconds).

### 8.2 Dependency Verification

Dependency consistency was also verified using:

```powershell
python -m pip check
```

Result:

`No broken requirements found.`

### 8.3 Quality Assurance and Test Limitations

The automated tests verify selected software behaviors, including:

- Report-validation retries
- Retrieval regression cases
- Model serialization and prediction consistency
- Attention output shapes and numerical properties
- Data-cleaning and schema-validation behavior

The test suite does not cover every execution path in the application. Statement coverage should be reported separately using a fresh coverage run against the current codebase and test suite.

Test results establish only the software behaviors exercised by the test suite. They do not establish clinical validity, scientific accuracy, or suitability for real-world medical decisions.

## 9. Limitations and Future Work

The project has several important limitations:

1. **Synthetic data:** The dataset is small and synthetic and cannot establish real-world clinical performance.
2. **Limited model evaluation:** Metrics are based on a particular train-test split and do not demonstrate external validity.
3. **Illustrative attention:** The attention components are not biologically trained and do not establish biological relationships.
4. **Retrieval limitations:** Some queries may return passages that are related to a topic but do not fully answer the question.
5. **Chunking limitations:** The selected embedding model restricts the effective chunk size below the original 300–500-token target, and overlap is not consistently maintained.
6. **Report-generation limitations:** Schema validation does not establish factual correctness or clinical safety.
7. **Incomplete test coverage:** Automated tests cover selected software behaviors, not every module or execution path.
8. **No clinical validation:** The project has not been validated for patient care, vaccine eligibility, dosing, or regulatory use.
9. **Educational source limitations:** The source reference is not an authoritative clinical or regulatory document.
10. **Live LLM testing limitation:** The LLM integration is implemented but has not been verified through an actual API call.
11. **Dashboard latency limitation:** The dashboard compares retrieval with offline report-generation latency rather than live LLM response time.

Future work could include more comprehensive automated testing, broader retrieval evaluation, improved source traceability, improved chunking, independent model evaluation, and clearer error handling.

Any real-world medical application would require authoritative sources, qualified clinical oversight, privacy and security controls, and rigorous validation appropriate to its intended use.

## 10. Reproduction Instructions

Run the commands from the project root in Windows PowerShell.

### 10.1 Create and Activate the Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If the virtual environment already exists, activate it rather than creating another one.

### 10.2 Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

Some dependencies, including the sentence-transformer embedding model when not cached locally, may require internet access for initial setup.

### 10.3 Run the Complete Pipeline

```powershell
python run_all.py
```

The runner executes the pipeline in dependency order:

1. Data validation and cleaning
2. Risk model training and evaluation
3. Self-attention analysis
4. Retrieval evaluation
5. Structured report generation
6. Dashboard generation

The pipeline stops if a step raises an exception. Some generated reports and indexes may be refreshed when their modules run.

### 10.4 Run Individual Modules

```powershell
python -m src.data_pipeline
python -m src.risk_model
python -m src.attention
python -m src.rag_index
python -m src.report_generator
python -m src.dashboard
```

Run the modules in dependency order when the required intermediate artifacts do not already exist.

### 10.5 Run Automated Tests

```powershell
python -m pytest -q
```

To measure statement coverage:

```powershell
python -m pytest --cov=src --cov-report=term-missing
```

### 10.6 Project Outputs

The project writes generated cleaned data, models, evaluation reports, figures, retrieval artifacts, structured reports, and the dashboard under `data/` and `outputs/`.

Important output locations include:

| Artifact | Location |
|---|---|
| Cleaned dataset | `data/patient_records_clean.csv` |
| Risk model | `outputs/risk_model.joblib` |
| Model evaluation | `outputs/risk_evaluation.joblib` |
| Attention heatmap | `outputs/figures/attention_heatmap.png` |
| Persistent RAG index | `outputs/chroma_store/` |
| Retrieval evaluation | `outputs/retrieval_tests.md` |
| JSON report | `outputs/reports/sample_report.json` |
| Markdown report | `outputs/reports/sample_report.md` |
| Pipeline latency | `outputs/report_generation_latency.json` |
| Dashboard | `outputs/figures/dashboard.png` |

## Conclusion

The mRNA-VIP project demonstrates how data preprocessing, supervised classification, NumPy-based self-attention, retrieval-augmented search, structured report validation, and dashboard visualization can be combined into a modular AI engineering workflow.

The latest verified results include:

- **30 passing automated tests** with zero failures.
- Successful dependency verification with no reported broken requirements.
- Logistic Regression and Random Forest model comparison.
- Five representative RAG retrieval evaluations.
- Persistent ChromaDB indexing.
- Successful offline structured report generation in JSON and Markdown.
- Successful generation of the consolidated dashboard.

The project demonstrates implementation progress and software integration rather than clinical validity. Its synthetic dataset, limited evaluation scope, illustrative attention weights, educational reference material, chunking constraints, and untested live LLM integration define important boundaries on the conclusions that can be drawn.

**mRNA-VIP is an educational AI engineering project only. It is not a clinically validated safety, eligibility, or dosing system and must not be used for medical or regulatory decisions.**