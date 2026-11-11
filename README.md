# Salon Experiment & Causal Analysis

**YuvaIntern – Data Science with Python Analyst Internship**

A beginner-friendly, end-to-end Python project demonstrating experiment design, A/B testing,
causal inference, segmentation, and CUPED for a fictional 22-outlet salon chain in Coimbatore.

> **Data transparency:** The internship page did not provide a dataset when this project was
> prepared. Therefore, this repository generates reproducible **synthetic data**. It is not
> real customer, salon, or YuvaIntern data. Replace the synthetic data with authorised
> internship data if it becomes available, and rerun the pipeline.

## Business scenario

The fictional salon chain tests an automated appointment reminder initiative. The randomized
experiment compares customers receiving automated reminders (treatment) with customers
continuing the existing process (control). The primary outcome is appointment no-show rate.
Retention and revenue per visit are secondary outcomes; cancellation rate and complaints are
guardrails.

A separate historical rollout is analysed with Difference-in-Differences (DiD): 11 outlets
receive a historical initiative and 11 comparable outlets form the control region. This is a
synthetic observational example and does not establish real-world impact.

## Requirements covered

- Business problem, hypothesis, primary/secondary/guardrail metrics
- A/B test design, two-proportion power/sample-size calculation, duration estimate
- Randomization balance and Sample Ratio Mismatch (SRM) checks
- Effect estimates, 95% confidence intervals, Welch t-tests and bootstrap intervals
- Difference-in-Differences and a pre-treatment parallel-trends diagnostic
- Segment-level effect estimates
- CUPED/regression-style variance reduction and estimated sample-size/duration savings
- Go / no-go / extend decision memo and limitations

## Quick start (Windows / VS Code)

1. Extract the ZIP.
2. Open the extracted `Salon-Experiment-Causal-Analysis` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, do not change execution policy; use `.venv\Scripts\python.exe`
   in place of `python` in the commands below.

5. Install libraries:

   ```powershell
   python -m pip install -r requirements.txt
   ```

6. Run the complete pipeline:

   ```powershell
   python run_project.py
   ```

   Or, if activation did not work:

   ```powershell
   .\.venv\Scripts\python.exe run_project.py
   ```

The first run generates about 300,000 appointment records and supporting tables. It then
cleans the data, performs analysis, creates charts, and writes Word documents.

## Project structure

```text
Salon-Experiment-Causal-Analysis/
├── data/
│   ├── raw/                 # generated synthetic CSV files
│   └── processed/           # cleaned/analysis-ready data
├── notebooks/               # optional notebook walkthrough
├── outputs/                 # result tables and JSON summary
├── reports/                 # generated Word reports + report guide
├── src/
│   ├── generate_data.py
│   ├── analysis.py
│   └── make_reports.py
├── visualizations/           # generated PNG charts
├── run_project.py
├── requirements.txt
└── README.md
```

## Dataset generated

- `appointments.csv`: 300,000 appointment records
- `customers.csv`: customer profile and pre-experiment history
- `outlets.csv`: 22 fictional Coimbatore outlets and region assignments
- `stylists.csv`: 150 stylists
- `services.csv`: service catalogue and canonical service names
- `product_sales.csv`: synthetic product transactions
- `experiment_assignment.csv`: customer-level randomized assignment

The appointment data intentionally includes missing customer IDs for walk-ins, duplicate-like
service labels across outlets, cancelled appointments with a no-show reason, and seasonal
patterns. The generator uses a fixed random seed so results are reproducible.

## Important methodological notes

1. The experiment is randomized at the **customer level** to avoid assigning one customer to
   both groups. Appointment outcomes are aggregated to customer level before inference.
2. No-show is represented in the raw appointment table as `status="cancelled"` and
   `cancellation_reason="no_show"` to reflect the task description. The analysis derives a
   separate `no_show` indicator from that reason.
3. Anonymous walk-ins have no customer ID and are excluded from customer-level retention and
   randomized customer experiment analyses; they remain in outlet-level operational summaries.
4. The sample-size calculation uses a two-sided two-proportion normal approximation. The
   achieved power and actual required sample are not guaranteed by a synthetic example.
5. The DiD estimate relies on parallel trends, no differential concurrent shocks, stable
   composition, and no spillovers. A pre-trend test is a diagnostic, not proof of the assumption.
6. Segment analyses are exploratory and can be underpowered. Interpret confidence intervals,
   not only p-values.
7. CUPED is calculated on customer-level outcomes using pre-period revenue as a covariate.
   Variance reduction is estimated from this dataset; it is not a guarantee of the same savings
   in a future experiment.

## Outputs

After running the pipeline, inspect:

- `data/processed/appointments_clean.csv`
- `outputs/ab_test_results.csv`
- `outputs/segment_results.csv`
- `outputs/did_results.json`
- `outputs/cuped_results.json`
- `outputs/analysis_summary.json`
- `visualizations/`
- `reports/Experiment_Design.docx`
- `reports/Decision_Memo.docx`
- `reports/Project_Report.docx`

## How to explain this project in an interview

"I designed a customer-level randomized experiment for automated salon appointment reminders.
I defined no-show rate as the primary metric, selected retention and revenue per visit as
secondary metrics, and used cancellation and complaint rates as guardrails. I checked
randomization balance and sample ratio mismatch, estimated effects with confidence intervals,
and used a historical Difference-in-Differences analysis with a pre-trend diagnostic. I also
applied CUPED using pre-experiment revenue to estimate variance reduction. The data in this
portfolio version is synthetic and clearly labelled."

## Ethical and privacy note

Do not publish real customer names, phone numbers, appointment details, or confidential company
data. Use only data you are authorised to process. This project generates fictional records.
