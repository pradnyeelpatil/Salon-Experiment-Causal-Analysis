from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt
import json

def _base_doc(title):
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(.65); sec.bottom_margin = Inches(.65)
    sec.left_margin = Inches(.75); sec.right_margin = Inches(.75)
    doc.styles["Normal"].font.name = "Arial"
    doc.styles["Normal"].font.size = Pt(10)
    doc.add_heading(title, 0)
    return doc

def _add_table(doc, rows):
    table = doc.add_table(rows=1, cols=len(rows[0]))
    table.style = "Light Shading Accent 1"
    for i, h in enumerate(rows[0]): table.rows[0].cells[i].text = str(h)
    for row in rows[1:]:
        cells = table.add_row().cells
        for i, val in enumerate(row): cells[i].text = str(val)
    return table

def create_reports(root, results):
    root=Path(root); report_dir=root/"reports"; report_dir.mkdir(exist_ok=True)
    ab=results["ab_test"]; did=results["did"]; cup=results["cuped"]

    # Experiment design document.
    d=_base_doc("Experiment Design Document")
    d.add_paragraph("Project: Salon Experiment & Causal Analysis | YuvaIntern Data Science with Python")
    d.add_heading("1. Business problem",1)
    d.add_paragraph("A fictional 22-outlet salon chain wants to determine whether automated appointment reminders reduce no-shows without harming customer experience or revenue.")
    d.add_heading("2. Hypotheses",1)
    d.add_paragraph("H0: Automated reminders do not change the appointment no-show rate. H1: Automated reminders reduce the no-show rate. The primary analysis is two-sided at alpha=0.05.")
    d.add_heading("3. Experiment design",1)
    d.add_paragraph("Randomize eligible identified customers 1:1 to treatment (automated reminder) or control (existing process). Keep assignment stable at customer level. Exclude anonymous walk-ins from customer-level retention and randomized analyses; report their operational outcomes separately.")
    d.add_heading("4. Metrics",1)
    _add_table(d,[["Type","Metric","Definition"],["Primary","No-show rate","No-show appointments / eligible appointments"],["Secondary","60-day retention","Customers with at least two completed visits in the observation window / eligible customers"],["Secondary","Revenue per visit","Appointment revenue / eligible appointments"],["Guardrail","Cancellation rate","Cancelled appointments / eligible appointments"],["Guardrail","Complaints","Complaints per 1,000 eligible appointments (requires complaint data)"]])
    d.add_heading("5. Sample size and duration",1)
    ss=ab["sample_size_assumptions"]
    d.add_paragraph(f"Illustrative planning assumptions: control no-show rate={ss['baseline_control_rate']:.0%}, treatment target={ss['target_treatment_rate']:.0%}, alpha={ss['alpha']:.2f}, power={ss['power']:.0%}. Approximate required sample: {ss['required_customers_per_group']:,} customers per arm using a two-proportion normal approximation. Estimated duration from the generated data's observed eligible daily customer volume: {ss['estimated_days_for_required_customers']} days. Recalculate with a real baseline, unique eligible customers, clustering/repeat visits, and operational constraints before launch.")
    d.add_heading("6. Randomization and monitoring",1)
    d.add_paragraph("Check baseline covariates and outlet mix by group. Check sample ratio mismatch (SRM) against the planned 50:50 split. Investigate assignment bugs, exclusions, and missing records before interpreting outcomes.")
    d.add_heading("7. Analysis plan",1)
    d.add_paragraph("Estimate treatment-minus-control effects with 95% bootstrap confidence intervals. Use a two-proportion test for appointment-level no-show proportions and Welch tests for continuous/customer-level outcomes. Report absolute percentage-point effects and relative changes separately.")
    d.add_heading("8. Guardrails and stopping",1)
    d.add_paragraph("Do not stop early just because a p-value becomes significant. Run for the planned duration and sample, monitor safety/complaints, and extend if the sample is inadequate or intervals remain too wide. Define operational harm thresholds with the business before launch.")
    d.add_heading("9. Limitations",1)
    d.add_paragraph("Synthetic data are illustrative. Customer-level assignment may still have interference across household members or shared phone numbers. Missing customer IDs limit retention analysis. Multiple metrics and segments increase false-positive risk. Ensure message delivery logs and consent are available in a real experiment.")
    d.save(report_dir/"Experiment_Design.docx")

    # Decision memo, deliberately concise.
    d=_base_doc("One-Page Decision Memo")
    d.add_paragraph("To: Salon Chain Management | From: Data Science Analyst | Subject: Automated appointment reminder experiment")
    d.add_heading("Executive summary",1)
    d.add_paragraph("This memo reports a reproducible synthetic-data demonstration. It is not evidence about a real salon chain.")
    d.add_heading("Primary outcome",1)
    d.add_paragraph(f"Treatment-minus-control customer-level no-show-rate difference: {ab['primary_difference_treatment_minus_control']:.4f}; bootstrap 95% CI {ab['primary_bootstrap_95_ci'][0]:.4f} to {ab['primary_bootstrap_95_ci'][1]:.4f}. Customer-level Welch-test p-value: {ab['customer_level_welch_p_value']:.4g}.")
    d.add_paragraph(f"SRM chi-square p-value: {ab['srm_chi_square_p_value']:.4g}. Customer counts: treatment {ab['treatment_customers']:,}; control {ab['control_customers']:,}.")
    d.add_heading("Historical observational estimate",1)
    d.add_paragraph(f"Outlet-month DiD estimate for revenue per appointment: {did['did_estimate']:.2f}; cluster-robust 95% CI {did['cluster_robust_95_ci'][0]:.2f} to {did['cluster_robust_95_ci'][1]:.2f}; p={did['did_p_value']:.4g}. Pre-trend diagnostic p={did['pretrend_p_value']:.4g}. {did['parallel_trends_diagnostic']}")
    d.add_heading("Variance reduction",1)
    d.add_paragraph(f"CUPED estimated variance reduction: {cup['estimated_variance_reduction_percent']:.1f}%; approximate sample-size/duration reduction: {cup['approx_duration_reduction_percent']:.1f}%.")
    d.add_heading("Recommendation",1)
    d.add_paragraph("EXTEND / VALIDATE before a real rollout. This repository uses synthetic data, so no real-world go/no-go decision is justified. For a real deployment, verify source data, consent and message delivery, prespecify business harm thresholds, run the planned randomized test, and base the decision on confidence intervals and guardrails.")
    d.add_heading("What could make the estimate wrong?",1)
    d.add_paragraph("Incorrect customer identity, non-delivered reminders, contamination between groups, seasonal shocks, outlet imbalance, missing walk-in IDs, changing service mix, concurrent discounts, interference, and a violation of parallel trends can bias the estimates.")
    d.save(report_dir/"Decision_Memo.docx")

    # Main report.
    d=_base_doc("Salon Experiment & Causal Analysis – Project Report")
    d.add_paragraph("YuvaIntern | Data Science with Python Analyst Internship")
    d.add_paragraph("Dataset disclosure: All records in this portfolio project are synthetic and generated for demonstration because no official dataset was available at project setup.")
    sections=[
        ("1. Introduction","This project applies data science and causal inference to a fictional salon chain with 22 Coimbatore outlets."),
        ("2. Problem statement","Management wants to know whether an automated appointment reminder initiative reduces no-shows and improves business outcomes, beyond seasonal changes."),
        ("3. Objectives","Design a randomized experiment; clean and explore salon data; estimate effects with confidence intervals; evaluate a historical rollout with DiD; segment results; and assess CUPED variance reduction."),
        ("4. Dataset","The generator creates 300,000 appointments, 45,000 customer profiles, 22 outlets, 150 stylists, service catalogue records and 65,000 product sales. The data intentionally contains anonymous walk-ins, inconsistent service labels, cancellations marked with a no-show reason and seasonal patterns."),
        ("5. Data preprocessing","Appointment IDs are deduplicated; service labels are standardised; no-show is derived from status=cancelled and cancellation_reason=no_show; missing customer IDs are retained for operational analysis but excluded from customer-level retention analysis."),
        ("6. Exploratory analysis","Monthly revenue, outlet revenue per appointment, appointment outcomes and product sales summaries are generated in outputs/ and visualizations/."),
        ("7. A/B test design","Eligible identified customers are randomized 1:1 at customer level to automated reminders or business-as-usual. No-show rate is primary; retention and revenue per visit are secondary; cancellation and complaints are guardrails."),
        ("8. Statistical analysis","The pipeline checks sample ratio mismatch, summarizes group balance, estimates customer-level differences with bootstrap 95% confidence intervals, and uses a two-proportion test for appointment no-show rates and Welch tests on customer-level outcomes (the randomization unit)."),
        ("9. Difference-in-Differences","The historical rollout compares 11 treatment outlets with 11 control outlets before and after 1 July 2026. The model includes outlet and month fixed effects and outlet-clustered standard errors. A pre-period treatment-by-time interaction is used as a parallel-trends diagnostic."),
        ("10. Segmentation","No-show effects are estimated separately for New, Returning and Loyal customer segments. These are exploratory comparisons and should be interpreted cautiously."),
        ("11. CUPED","Customer baseline spend is used as a pre-experiment covariate to adjust revenue-per-visit outcomes. The report records estimated variance reduction and approximate sample-size savings."),
        ("12. Results","See the generated JSON/CSV result files and charts. Exact figures are generated by running run_project.py and should be read from outputs/ rather than copied from example text."),
        ("13. Limitations","The dataset is synthetic; complaint data and actual message delivery logs are not present; walk-in IDs are missing; pre-trend tests cannot prove parallel trends; and segment-level tests may have limited power."),
        ("14. Conclusion","The project demonstrates an end-to-end analytical workflow. No real business decision should be made from synthetic estimates. Replace the data with authorised salon records and validate assumptions before operational use.")
    ]
    for h,p in sections: d.add_heading(h,1); d.add_paragraph(p)
    d.add_heading("Selected generated results",1)
    d.add_paragraph(f"A/B primary effect (treatment minus control): {ab['primary_difference_treatment_minus_control']:.4f}; bootstrap 95% CI {ab['primary_bootstrap_95_ci']}.")
    d.add_paragraph(f"DiD estimate: {did['did_estimate']:.2f}; 95% CI {did['cluster_robust_95_ci']}.")
    d.add_paragraph(f"CUPED variance reduction: {cup['estimated_variance_reduction_percent']:.1f}%.")
    d.save(report_dir/"Project_Report.docx")
    print("Created Experiment_Design.docx, Decision_Memo.docx and Project_Report.docx.")
