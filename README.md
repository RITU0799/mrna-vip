Student engineering project output only. Not real clinical, medical, or regulatory guidance.

# mRNA-VIP | Educational ML and RAG Demonstration

mRNA-VIP is an individual applied-AI learning project that demonstrates a modular workflow combining synthetic patient-record preprocessing, reaction-risk classification, NumPy-based self-attention, retrieval-augmented generation (RAG), structured report validation, and dashboard visualization.

The project uses a synthetic dataset and an educational course reference. **It is not a clinically validated system and must not be used for medical, vaccine eligibility, dosing, or regulatory decisions.**

## Project Structure

- `data/`: Input dataset, educational reference text, and cleaned dataset
- `notebooks/`: Exploratory data analysis notebook and visualizations
- `src/`: Data processing, model training, attention, retrieval, report generation, and dashboard modules
- `outputs/`: Saved models, evaluation results, retrieval index, reports, and figures
- `tests/`: Automated tests for selected project behaviors
- `run_all.py`: Pipeline runner that executes project modules in dependency order
- `requirements.txt`: Python dependencies
- `PORTFOLIO_REPORT.md`: Detailed project implementation and evaluation report

## Setup

Run the following commands in Windows PowerShell from the project root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If the virtual environment already exists, activate it rather than creating another one.

## Run the Project

Run the complete pipeline:

```powershell
python run_all.py
```

Run the automated tests:

```powershell
python -m pytest -q
```

Run individual modules when their required input artifacts are available:

```powershell
python -m src.data_pipeline
python -m src.risk_model
python -m src.attention
python -m src.rag_index
python -m src.report_generator
python -m src.dashboard
```

The full pipeline executes preprocessing and model evaluation before report and dashboard generation. The retrieval module may require an initial download of its embedding model, depending on the local cache and installed dependencies.

## Implementation

### Data Validation and Preprocessing

The data pipeline validates seven required columns, checks patient identifiers and target values, validates numeric fields and reaction-history categories, and handles missing values. Missing `Biomarker_Level` values are imputed using the median; missing `Prior_Reaction_History` values are imputed using the most frequent observed category.

The pipeline saves a cleaned CSV and a data-validation report. Imputation can introduce bias and must be considered when interpreting model results.

### Risk Classification

The model module evaluates Logistic Regression and Random Forest using a stratified train-test split. The selected model and preprocessing pipeline are serialized for subsequent inference and report generation.

Evaluation metrics describe performance on the project's synthetic dataset and test split. They do not establish clinical accuracy or performance on real patient populations.

### Self-Attention Demonstration

The attention module implements scaled dot-product self-attention using NumPy, deterministic nucleotide embeddings, and Query, Key, and Value projections. It saves an attention heatmap for an illustrative nucleotide sequence.

The demonstration is computational only. It does not establish biological significance or clinical insight.

### Retrieval-Augmented Guideline Search

The retrieval module indexes an educational reference as overlapping text chunks and uses semantic similarity, lexical matching, and query-intent adjustments to rank relevant passages. The project persists its retrieval artifacts under `outputs/chroma_store/`.

Retrieval evaluation records representative questions, ranked passages, relevance observations, scores, and timing information. A high ranking or similarity score does not guarantee that a passage completely or correctly answers a question.

### Structured Report Generation (Week 4)

The report generator supports two explicit modes:

- **Offline mode (default):** deterministic rule-based educational summary; it requires no API key.
- **LLM mode:** LangChain `ChatOpenAI` with an OpenAI-compatible API and Pydantic structured output. It uses only the patient record, experimental classifier output, and retrieved course-reference chunks supplied in the prompt. The LLM integration is implemented but live API execution has not been verified; offline report generation and validation have been tested.

To enable LLM mode:

1. Install the updated dependencies with `python -m pip install -r requirements.txt`.
2. Copy `.env.example` to `.env` and set your own `OPENAI_API_KEY`.
3. Run `python -m src.report_generator --llm`.

The API key must remain private and `.env` must never be committed. The default model is `gpt-4o-mini`; override it with `MRNA_VIP_LLM_MODEL` if needed. `MRNA_VIP_USE_LLM=true` enables LLM mode for the pipeline runner; otherwise the pipeline remains offline by default.

The LLM path retries once only when Pydantic schema validation fails. Deterministic post-generation guardrails force `Eligible_For_Vaccine` to `false`, replace the dose field with the source-record dose converted from micrograms to milligrams (not a recommendation), enforce the fixed educational disclaimer, and discard warning text not found in retrieved source chunks. The legacy field name `Retrieved_Safety_Reference_Excerpts` is retained for PRD schema compatibility; the values are excerpts from the supplied educational course reference, **not verified FDA citations**.

The report and its decision-related fields are illustrative outputs. Schema validation checks structure; it does not establish factual or clinical correctness. The 100-microgram secondary-review threshold in the supplied course reference is not interpreted as a maximum safe dose.

### Dashboard

The dashboard module generates a static three-panel figure containing:

1. Risk-classifier ROC curve and AUC
2. Self-attention heatmap
3. Recorded retrieval and report-generation timings

The figure is saved to `outputs/figures/dashboard.png`.

## Outputs

Important generated artifacts include:

- `outputs/risk_model.joblib` â€” serialized risk-model artifact
- `outputs/risk_evaluation.joblib` â€” saved model evaluation data
- `outputs/figures/attention_heatmap.png` â€” attention visualization
- `outputs/figures/dashboard.png` â€” consolidated project dashboard
- `outputs/chroma_store/` â€” persistent retrieval artifacts
- `outputs/reports/data_validation.txt` â€” data validation report
- `outputs/reports/sample_report.json` â€” structured report
- `outputs/reports/sample_report.md` â€” readable report
- `outputs/retrieval_tests.md` â€” retrieval evaluation results
- `outputs/report_generation_latency.json` â€” recorded pipeline timing data and report-generation mode

Some outputs are created or refreshed when their corresponding modules run.

## Testing and Quality Assurance

Automated tests cover selected data-validation, model, attention, retrieval, and report-generation behaviors. Run `python -m pytest -q` to verify the current test suite.

Record the test count and coverage from the latest actual test run rather than relying on older report values. Passing tests verify only the behaviors they exercise and do not establish clinical validity.

## Limitations and Future Work

- The dataset is small and synthetic.
- Model metrics are based on a limited evaluation split.
- The attention demonstration uses illustrative components rather than biologically trained parameters.
- Retrieval may return related passages that do not fully answer a question.
- Report schema validation does not guarantee factual or clinical correctness.
- The dashboard is a static visualization, not an interactive clinical application.
- The project has not been validated for patient care or regulatory use.

Future work may include stronger automated testing, retrieval-quality evaluation, better source traceability, independent model evaluation, and review by qualified domain experts.

**Educational disclaimer:** This repository demonstrates AI engineering concepts only. Its data, models, retrieved text, visualizations, and reports are not medical advice or validated clinical or regulatory guidance.
