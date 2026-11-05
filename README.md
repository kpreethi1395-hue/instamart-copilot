# Customer Intelligence & Support Copilot — Teaching Kit

> Educational implementation of the supplied fictional Instamart-style capstone. All generated records are synthetic; no real customer data, hosted LLM, API key, or network call is needed.

## Learning outcomes
Students learn to (1) fetch/read CSV exports and audit data, (2) clean and persist trusted tables in SQLite, (3) write SQL and pandas analytics, (4) segment customers and train a leakage-aware lapse model, (5) redact PII and triage text with retries/output validation, and (6) retrieve policy passages and execute bounded database-backed tools.

## Setup and run
Python 3.11+ recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_data.py
python run_project.py
```

Open `notebooks/Customer_Intelligence_Support_Copilot.ipynb` and run cells from top to bottom. The notebook calls the same modules used by the scripts.

## Repository map
- `generate_data.py`: deterministic fictional CSV and policy generation.
- `mock_llm.py`: offline stand-in for classification, policy answers and agent tool calls.
- `src/data_pipeline.py`: audit, clean, SQLite load, validation.
- `src/customer_models.py`: RFM, K-means, PCA, lapse model with train/test separation.
- `src/triage.py`: token guardrails, PII redaction, retries, response validation, routing.
- `src/policy_rag.py`: chunking, TF-IDF retrieval and grounded policy evaluation.
- `src/refund_agent.py`: order lookup, refund calculation, ledger and bounded tool loop.
- `sql/queries.sql`: seven SQL learning patterns.
- `outputs/`: generated artifacts; `state/`: agent checkpoint examples.

## How students should learn the data-fetching flow
1. `pd.read_csv("data/raw/orders.csv", keep_default_na=False)` loads local export rows. In a production project, a fetch function might download from an approved endpoint or query a warehouse, but this exercise intentionally uses local synthetic data and has no credentials.
2. Inspect shape, data types, null/blank counts, duplicate rows, keys and impossible values before cleaning.
3. Apply one documented cleaning rule at a time and compare row/value counts before and after.
4. Write clean CSVs for easy inspection and SQLite tables for repeatable SQL queries.
5. Run assertions before downstream modelling. A failed assertion should stop the pipeline rather than silently propagate bad data.

## Part A — Data foundation
The pipeline standardizes city/gender, treats invalid ages as missing, resolves duplicate orders by keeping the latest timestamp, imputes missing amounts from catalogue price, sets negative delivery times to missing and imputes only completed-order delivery durations by product category. Review every rule before teaching; the capstone handout's numeric checkpoints should be used as acceptance tests.

## Part B — Segments and lapse
### Lapse problem
- Cutoff 2025-09-01. The cohort is every customer with at least one completed order before the cutoff: 1,156 customers, of whom 443 lapsed (38.32%). Train 924, test 232.
- Baseline (always predict "not lapsed"): accuracy 61.6% and lapsed-class F1 0.
- Features: recency, frequency, monetary, average discount, cancellation rate, average delivery time, favourite category, city, age, gender and tenure, all computed from orders dated before the cutoff.

**Leakage statement.** No encoder, scaler or resampler saw test rows. No data from 2025-09-01 onwards entered the features. The segment labels from the K-means step were not used as features.

The scaler's learned mean frequency (8.1147) differs from the full-cohort mean (8.0285) because the scaler was fitted on the 924 training rows only, while the cohort mean also includes the 232 test rows.

### Model results
| Model | Best parameters | CV F1 | Test precision | Test recall | Test F1 |
|---|---|---|---|---|---|
| Logistic regression | C = «paste» | «paste» | 0.644 | 0.652 | 0.648 |
| Random forest | max_depth 8, min_samples_leaf 10 | 0.681 | 0.659 | 0.652 | 0.655 |

Confusion matrices ([[TN, FP], [FN, TP]]) on the 232 test customers:
- Logistic regression: [[111, 32], [31, 58]]
- Random forest: [[113, 30], [31, 58]]

**Model choice.** I selected the random forest because it had the higher cross-validated F1 (0.681) and a test F1 of 0.655, well above the 0.55 requirement and the baseline's 0. Missing a lapsing customer is the costlier error, because that customer leaves and the revenue is lost, while wrongly targeting a loyal customer only costs a small discount or message. Recall on the lapsed class therefore matters, and F1 is a sensible single number because it also penalises wasting offers on customers who would have stayed. The model was chosen on cross-validation, and the test set was used only for the final evaluation after tuning was finished.

### Class-balance experiment
The training set was over-sampled by adding random duplicates of lapsed training rows (with replacement, random_state=42) until both classes had 570 rows. The best logistic regression was refitted on this data and evaluated on the original, unchanged test set.

| Metric | Original LR | Balanced LR | Change |
|---|---|---|---|
| Precision | 0.644 | 0.589 | −0.055 |
| Recall | 0.652 | 0.854 | +0.202 |
| F1 | 0.648 | 0.697 | +0.049 |

Recall moved the most, rising by 0.202, while precision did not rise and fell slightly. For the retention team this means that balancing catches many more of the customers who will lapse (85% instead of 65%), but a larger share of the people flagged would have stayed anyway. It is worth it only when the retention offer is cheap relative to the value of a saved customer.

### Interpretation, persistence and retention list
**Top 5 random-forest features:**
1. `frequency` (0.280): how often a customer orders is the strongest signal; customers with a regular habit rarely leave.
2. `monetary` (0.271): total spend reflects how embedded the customer is, so low spenders are easier to lose.
3. `recency` (0.115): the longer since the last order, the likelier the customer has drifted away.
4. `avg_delivery` (0.061): slower average deliveries point to a service-quality problem that can push customers away.
5. `cancel_rate` (0.052): customers who cancel often are showing friction with the service.

**Logistic-regression coefficient signs** (on standardised features):
- `frequency` is negative (−1.698): more orders lower the lapse probability. This is by far the largest effect.
- `recency` is positive (+0.040): a longer gap since the last order raises the lapse probability, although the effect is small once frequency is in the model.
- `avg_delivery` is positive (+0.079): slower deliveries raise the lapse probability slightly.

Frequency and monetary are strongly related, so the coefficient of `monetary` (+0.285) should not be read on its own.

**Persistence.** The selected model was saved to `outputs/lapse_model.joblib`, reloaded with `joblib.load`, and reproduces the original test predictions exactly (True).

**Retention list.** `outputs/retention_list.csv` holds the 10 Premium customers in the test set with the highest predicted lapse probability; 1 of them truly lapsed. The Premium flag comes from whole-window RFM that includes the September window that defines the lapse label, so this list is illustrative and not a leakage-free evaluation.

## Part C — Ticket triage and privacy
## Part C: Support Ticket Triage

- **Input guardrails:** 4 empty tickets (`skipped_empty`) and 4 long tickets (`flagged_long`, 734 estimated tokens each, above the limit of 400). None was sent to the model.
- **Redaction:** emails 55, phone numbers 91, order ids 132, 278 substitutions in total. Re-applying the three patterns to the redacted texts found 0. An assertion confirms that no prompt contains an email, phone number or order id, and the user message is exactly the redacted ticket text.
- **Prompts:** a zero-shot and a few-shot system prompt. Both contain CLASSIFY, the four allowed categories and a JSON-only reply format. The few-shot prompt adds an `EXAMPLES:` line with four labelled examples, including a fraud example.
- **Retries:** 19 TransientLLMErrors in a full run, all recovered by one retry (maximum 3 retries, exponential backoff with jitter, base delay 0.05 s). Every wait was between 0.0451 and 0.0574 s, and no ticket ended with a permanent error.
- **Output validation:** valid JSON with an allowed category is accepted (`classified_json`). Otherwise the lowercase words are extracted with `re.findall(r"[a-z]+", ...)` and the reply is accepted only if exactly one allowed category word is present (`classified_keyword`). Everything else becomes `flagged_invalid_output` and goes to human review.
- **Status counts (full run of 156 eligible tickets):** classified_json 135, classified_keyword 12, flagged_invalid_output 9.

| Run | Classified | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|
| Zero-shot | 147 | 85.7% (126/147) | 9,556 | 65.0 |
| Few-shot | 147 | 100% (147/147) | 25,312 | 172.2 |

Zero-shot confusion table (rows true, columns predicted):

| true \ predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

All 21 zero-shot errors are fraud tickets predicted as billing (11) or technical tickets predicted as billing (10).

**When few-shot is worth the tokens.** The few-shot prompt costs 172.2 tokens per classified ticket against 65.0 for zero-shot, about 107 more per ticket and 2.6 times the total (25,312 against 9,556). In return it removes all 21 zero-shot errors, raising accuracy from 85.7% to 100%. The errors matter because a fraud ticket sent to the billing queue misses the 2-hour deadline for reaching the security team, so the extra tokens are well spent whenever a wrong route costs more than a few more tokens. When volume is very large, the budget is tight, or a misrouted ticket is cheap to correct, the shorter zero-shot prompt may be enough.

- **Budget demo:** zero-shot run with TOKEN_BUDGET = 6000. Once the running total reached the budget, no more model calls were made and **58 tickets** ended `budget_exhausted` (`outputs/triage_budget_demo.csv`). Only 86 JSON and 7 keyword classifications were completed before the budget ran out.
- **Final output:** `outputs/triage_results.csv` (164 rows) from the few-shot run with TOKEN_BUDGET = 100000. Fraud goes to `security_team` (HIGH), premium customers to `premium_desk` (HIGH), everything else to `support_queue` (NORMAL), and unresolved tickets to `human_review` (NORMAL).

| Route | Tickets |
|---|---|
| support_queue | 102 |
| security_team | 34 |
| human_review | 17 |
| premium_desk | 11 |

Priority: 45 HIGH and 119 NORMAL.

- **Rule-validated:** the 147 tickets with status `classified_json` (135) or `classified_keyword` (12), which were routed by the rules above.
- **Sent to human review:** the 17 tickets with status `skipped_empty` (4), `flagged_long` (4) or `flagged_invalid_output` (9). In the budget demo, `budget_exhausted` tickets also go to human review.

## Part D — Policy retrieval and refund tools
### Retrieval
The five policy files are split into 40-word chunks with a 10-word overlap, giving 15 chunks: refund 4, delivery 3, escalation 3, payment 3, privacy 2. TF-IDF is fitted on the chunks, and chunks are ranked by cosine similarity. For the query "refund policy late delivery" all three top chunks come from `refund_policy.txt`.

TF (term frequency) measures how often a word appears in a chunk, so it rewards words the chunk talks about. IDF (inverse document frequency) measures how rare a word is across all chunks, so it gives more weight to distinctive words than to common ones. A limitation is that TF-IDF matches words rather than meaning: a question such as "when do I get my money back?" shares no words with "refunds are credited", so the right chunk could be missed.

| Setting | Chunks | Expected doc in top 3 (of 12) | Fact in answer (of 12) | Fallback "I could not find" (of 2) |
|---|---|---|---|---|
| Overlap 10 | 15 | 12 | 12 | 2 |
| Overlap 0 | 13 | 12 | 12 | 10 |

With overlap 0 this question breaks: «paste the question from the FAILS line». The sentence holding the answer is cut across a chunk boundary, so the retrieved chunk starts partway through it (for the refund-approval sentence, the chunk begins at "1000 must be approved..." and loses "A refund above INR"). A 10-word overlap repeats the boundary words in both neighbouring chunks, which keeps short facts intact.
FAILS: 'What delivery time makes a completed order eligible for a full refund?' | expected_doc in top3: True | fact in answer: False
   answer: receives a full refund of the order amount.
FAILS: 'How soon is money returned when a payment fails but is deducted?' | expected_doc in top3: True | fact in answer: False
   answer: If a payment fails but money is deducted, the amount is returned automatically.

   ### Agent results
| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1087 |
| S2 | REFUND | 135.00 | 5 | 1087 |
| S3 | ESCALATE | 1,999.00 | 5 | 1074 |
| S4 | DENIED | 0 | 4 | 757 |
| S5 | DENIED | 0 | 3 | 451 |
| S6 | NOT_FOUND | 0 | 2 | 148 |
| S7 | FALLBACK_MAX_ITERATIONS | 0 | 6 | 3228 |

«check: confirm the model-call column against your printed lines.» Each result is appended to `outputs/agent_runs.jsonl`, and the issued and escalated actions are in `outputs/refund_ledger.jsonl`.

### Stop-safely demonstrations
- S1 with `token_limit=200`: **FALLBACK_TOKEN_LIMIT**. The run stops right after the model call that pushes the total over the limit, before the requested tool is executed.
- S1 with `time_limit_seconds=0`: **FALLBACK_TIMEOUT**. The check runs at the top of each iteration, so the run stops before the first model call.
- S7 with the default limits: **FALLBACK_MAX_ITERATIONS** after all 6 model calls, because the `[LOOP]` message keeps asking for `search_policy`.

Every fallback returns the steps completed so far and a short human-handoff summary in `final_answer`.

### Crash and resume (S1)
A checkpoint is written to `state/` after every tool observation. A simulated crash was raised right after the second tool execution, and `resume_agent` then continued from the checkpoint. Tools executed across the crash and the resume, in order: `get_order`, `search_policy`, `calculate_refund`, `issue_refund` (four executions, none repeated). Final status REFUND with INR 128.25.

### ReAct versus plan-and-execute
My ReAct agent makes one model call per step: 5 calls for an issued or escalated refund (S1 to S3), 4 for an on-time delivery (S4), 3 for a cancelled order (S5) and 2 for an unknown order (S6). Each call re-sends the whole message history, so tokens grow with every step; S1 used «S1 tokens» in total. A plan-and-execute design asks the model once for the full step list and then runs the tools without further model calls, so a normal refund would take about 2 calls (the plan and the final answer) and noticeably fewer tokens.

The difference shows up on unexpected observations. In ReAct the model sees `order not found` or `status: cancelled` right away and stops after 2 or 3 calls, so no refund tool is ever requested. Plan-and-execute has already committed to a plan containing `calculate_refund` and `issue_refund`, so it needs a check after every step or an extra replanning call to avoid acting on a bad order.

For this refund workflow I would choose ReAct. The path is short, so the extra tokens are small, and money is involved, so checking each observation before the next action is worth more than saving calls. Per-step checkpoints also give a safe point to resume from. Plan-and-execute would suit longer, predictable workflows where model calls are expensive.

## GitHub suggestions
These public repositories/documentation projects are useful for learning concepts; do not copy a student's capstone solution. Adapt the patterns and cite the source in your notes:
- pandas documentation: https://github.com/pandas-dev/pandas — CSV loading, cleaning, groupby, pivot tables.
- scikit-learn: https://github.com/scikit-learn/scikit-learn — pipelines, preprocessing, clustering, model selection, metrics and TF-IDF.
- Python tutorial: https://github.com/python/cpython/tree/main/Doc/tutorial — Python fundamentals.
- Jupyter Notebook: https://github.com/jupyter/notebook — notebook workflow.
- SQLite: https://github.com/sqlite/sqlite — SQL database engine (official mirror).
- scikit-learn text feature extraction guide: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction.

## Limitations and next steps
- This is a teaching implementation and may need adjustment to match every numeric checkpoint in a strict grading rubric; compare actual outputs with the supplied brief and investigate differences rather than hardcoding expected answers.
- The offline mock model is deterministic rule-based simulation, not a real language model.
- The refund ledger is a local JSONL audit trail, not a real transaction system.
- Production work would add access controls, secure secret management, monitoring, schema contracts, data lineage, model drift checks and human approval workflows.
