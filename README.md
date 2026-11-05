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

The raw exports are deliberately dirty. This part audits them, cleans them with explicit rules, stores the clean tables in SQLite (`data/instamart.db`) and answers business questions in SQL and pandas. Scripts: `src/a1_audit.py`, `src/a2_clean.py`, `src/a3_validate.py`, `src/a4_sql.py`, `src/a5_eda.py`; queries in `sql/queries.sql`.

### Audit of the raw data
Each raw table was loaded with pandas (`tickets.csv` with `keep_default_na=False`). The first three audit figures are the table sizes: customers has 1,200 rows, products 40, orders 13,084 and tickets 164. Data types and missing values per column are printed by `a1_audit.py`.

| Figure | Value |
|---|---|
| Raw order rows | 13,084 |
| Exact duplicate rows | 119 |
| Rows with status pending | 656 |
| Distinct order_id | 12,309 |
| Non-numeric order_amount entries | 120 |
| Negative delivery_minutes | 106 |
| Blank delivery_minutes | 2,541 |
| Customers: blank age | 61 |
| Customers: age outside 10–100 | 6 |
| Customers: distinct raw city spellings | 24 |
| Customers: distinct non-blank raw gender spellings | 10 |
| Customers: blank gender | 30 |

### Cleaning rules and values changed

**Customers**

| # | Rule | Values changed |
|---|---|---|
| C1 | City: strip leading and trailing spaces, then title case | 24 raw spellings reduced to exactly 6 cities |
| C2 | Gender: map every spelling to male, female or unknown (blank becomes unknown) | 10 raw spellings reduced to 3 values; 30 blank became unknown |
| C3 | Age: values outside 10–100 set to missing, then all missing ages filled with the median age of the customer's city (`groupby` + `transform`) | 6 set to missing; 67 filled (61 blank + 6) |
| C4 | `signup_date` converted to a datetime | all 1,200 rows |

**Orders** (applied in this order)

| # | Rule | Values changed |
|---|---|---|
| O1 | `order_ts` converted to a datetime | all rows |
| O2 | Exact duplicate rows removed | 119 rows removed |
| O3 | For each `order_id`, keep only the record with the latest `order_ts` (drops the earlier pending record) | 656 rows removed |
| O4 | `order_amount` converted to numeric (unparseable becomes missing), then missing amounts filled with `round(price_inr × quantity × (1 − discount_pct / 100), 2)` | 112 amounts filled |
| O5 | Negative `delivery_minutes` set to missing; for completed orders every missing value is filled with the median delivery time of the order's product category (`groupby` + `transform`); cancelled orders are left missing | 105 set to missing; 774 completed orders filled; 1,199 cancelled orders left missing |

Cancelled orders keep a missing delivery time on purpose: they were never delivered, so imputing a time would invent data.

**Result after cleaning**
- Cities: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184.
- Genders: female 610, male 560, unknown 30. No missing ages, and every age is between 18 and 60.
- Orders: 12,309 rows, one per `order_id` (11,110 completed and 1,199 cancelled), no pending rows, no missing `order_amount`.
- No missing `delivery_minutes` among completed orders; 1,199 missing among cancelled orders.
- Sum of `order_amount` over completed orders: INR 2,998,668.80.

### Storage and validation
The clean tables are written to `data/clean/customers.csv`, `products.csv` and `orders.csv` and to the SQLite database `data/instamart.db` as tables `customers`, `products` and `orders` (`order_ts` stored as text in the form YYYY-MM-DD HH:MM:SS). `src/a3_validate.py` asserts 14 checkpoints (table names, row counts, status counts, no pending rows, no missing amounts, delivery-time rules, completed revenue, city counts, gender counts and the age range) and prints exactly:

```
ALL CHECKPOINTS PASSED
```

### SQL analysis
Each query was saved in `sql/queries.sql` and run with `pd.read_sql_query` against `data/instamart.db`. Only completed orders count as revenue or spend.

| Query | Technique | Result |
|---|---|---|
| Q1 | WHERE then HAVING | 30 customers with a total above INR 8,000 over completed orders above INR 500 |
| Q2 | Three-table join | 30 rows (city × category); the revenues add up to INR 2,998,668.80; the largest cell is Bengaluru / Electronics at INR 233,909.45 |
| Q3 | Scalar subquery | 11 products priced above the average catalogue price |
| Q4 | LEFT JOIN | 25 customers with no completed order |
| Q5 | Common table expressions | 550 customers above the average total spend of INR 2,552.06 |
| Q6 | RANK, DENSE_RANK, ROW_NUMBER over city | see the table below; 8 rows have RANK different from ROW_NUMBER |
| Q7 | Running total | 183 daily rows; the last cumulative value is INR 2,998,668.80 |

Top customer per city (Q6):

| City | Customer | Total spend (INR) |
|---|---|---|
| Bengaluru | C0082 | 12,856.20 |
| Chennai | C0930 | 10,151.40 |
| Delhi | C0291 | 13,742.60 |
| Hyderabad | C0743 | 12,730.05 |
| Mumbai | C0693 | 14,766.55 |
| Pune | C0426 | 10,377.10 |

RANK() and ROW_NUMBER() differ in 8 rows because customers with exactly equal spend share the same RANK, while ROW_NUMBER() always gives every row a unique number, so after a tie the two sequences drift apart.

### pandas cross-check
Q2 was rebuilt with `pd.pivot_table(..., margins=True)`. Its grand total is INR 2,998,668.80, equal to the SQL total.

Per-city sum, mean and count of completed `order_amount` (`groupby` with named aggregation):

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Chennai |  463629.15 | 272.72 | 1700 |
| Delhi | 502829.80 | 243.62 | 2064 |
| Hyderabad | 528589.90 | 284.95 | 1855 |
| Pune | 500918.05 | 252.48 | 1984 |

### Exploratory analysis of completed order_amount
| Statistic | Value |
|---|---|
| Mean | 269.91 |
| Median | 135.00 |
| Skewness | about 4.61 |
| Q1 / Q3 | 64.80 / 297.00 |
| 1.5 × IQR fences | lower below zero (no outliers), upper 645.30 |
| Outliers beyond the fences | 897 above the upper fence, none below the lower fence |

Correlation matrix of `order_amount`, `discount_pct`, `quantity` and `delivery_minutes` over completed orders: the pair with the largest absolute correlation is `order_amount` with `discount_pct` at about −0.095; `order_amount` with `quantity` is +0.087.

**What is this data about?** It describes completed quick-commerce orders from six Indian cities across five product categories, with an amount, discount, quantity and delivery time for each order. **What do the distributions and statistics look like?** Order values are strongly right-skewed: the mean (269.91) is almost twice the median (135.00) and the skewness is about 4.61, so most orders are small and a few are large. **Are there outliers or anomalies?** 897 orders lie above the upper fence of 645.30 and none below. They are mostly expensive Electronics orders, so they are genuine large orders and not data errors. **What relationships exist between variables?** The largest absolute correlation is order_amount with discount_pct at about −0.095, just ahead of order_amount with quantity at +0.087. Both are very weak, and a correlation of this size does not establish causation.

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
