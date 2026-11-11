from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
import statsmodels.formula.api as smf

def _bootstrap_diff(a, b, reps=1500, seed=42):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diffs = np.empty(reps)
    for i in range(reps):
        diffs[i] = rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
    return float(np.mean(a) - np.mean(b)), [float(x) for x in np.quantile(diffs, [.025, .975])]

def _two_prop_sample_size(p_control=.20, p_treatment=.15, alpha=.05, power=.80):
    effect = abs(proportion_effectsize(p_control, p_treatment))
    n = NormalIndPower().solve_power(effect_size=effect, alpha=alpha, power=power,
                                     ratio=1.0, alternative="two-sided")
    return int(np.ceil(n))

def run_analysis(root):
    root = Path(root)
    raw, processed, out, viz = root/"data"/"raw", root/"data"/"processed", root/"outputs", root/"visualizations"
    processed.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True); viz.mkdir(parents=True, exist_ok=True)
    ap = pd.read_csv(raw/"appointments.csv", parse_dates=["appointment_date"])
    customers = pd.read_csv(raw/"customers.csv")
    assignment = pd.read_csv(raw/"experiment_assignment.csv")
    outlets = pd.read_csv(raw/"outlets.csv")
    sales = pd.read_csv(raw/"product_sales.csv", parse_dates=["sale_date"])

    # Basic data quality checks.
    quality = {
        "rows_before": int(len(ap)),
        "duplicate_appointment_ids": int(ap["appointment_id"].duplicated().sum()),
        "missing_customer_ids": int(ap["customer_id"].isna().sum()),
        "missing_stylist_ids": int(ap["stylist_id"].isna().sum()),
        "unknown_status_values": int((~ap["status"].isin(["completed", "cancelled"])).sum())
    }
    ap = ap.drop_duplicates("appointment_id").copy()
    ap["service_name"] = (ap["service_name_raw"].astype(str).str.strip().str.lower()
                          .replace({"hair cut":"haircut", "hair cutting":"haircut",
                                    "men hair cut":"mens haircut", "hair cut men":"mens haircut",
                                    "face cleanup":"facial", "facial service":"facial"}))
    ap["no_show"] = ((ap["status"] == "cancelled") & (ap["cancellation_reason"] == "no_show")).astype(int)
    ap["completed"] = (ap["status"] == "completed").astype(int)
    ap["month"] = ap["appointment_date"].dt.to_period("M").astype(str)
    ap.to_csv(processed/"appointments_clean.csv", index=False)

    # EDA summary and plots.
    monthly = ap.groupby("month").agg(
        appointments=("appointment_id","count"), revenue=("revenue","sum"),
        no_show_rate=("no_show","mean"), completed=("completed","sum")
    ).reset_index()
    monthly.to_csv(out/"monthly_kpis.csv", index=False)
    outlet_kpis = ap.groupby("outlet_id").agg(
        appointments=("appointment_id","count"), revenue=("revenue","sum"),
        revenue_per_appointment=("revenue","mean"), no_show_rate=("no_show","mean")
    ).reset_index()
    outlet_kpis.to_csv(out/"outlet_kpis.csv", index=False)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12,5)); sns.lineplot(data=monthly, x="month", y="revenue", marker="o")
    plt.xticks(rotation=60); plt.title("Monthly salon revenue (synthetic data)"); plt.tight_layout()
    plt.savefig(viz/"monthly_revenue.png", dpi=150); plt.close()

    plt.figure(figsize=(10,5)); sns.barplot(data=outlet_kpis, x="outlet_id", y="revenue_per_appointment")
    plt.xticks(rotation=60); plt.title("Revenue per appointment by outlet"); plt.tight_layout()
    plt.savefig(viz/"outlet_performance.png", dpi=150); plt.close()

    plt.figure(figsize=(7,5)); sns.barplot(x=["No-show","Completed"], y=[ap.no_show.mean(), ap.completed.mean()])
    plt.ylabel("Share of all appointments"); plt.title("Appointment outcomes"); plt.tight_layout()
    plt.savefig(viz/"appointment_outcomes.png", dpi=150); plt.close()

    # Customer-level experiment dataset; anonymous walk-ins are excluded from retention and randomization.
    exp = ap[(ap["experiment_period"] == 1) & ap["customer_id"].notna()].copy()
    exp = exp.merge(assignment, on="customer_id", how="inner", suffixes=("", "_assigned"))
    # Customer level is the analysis unit because randomization is by customer.
    exp["treatment"] = (exp["treatment_group"] == "Treatment").astype(int)
    customer_exp = exp.groupby(["customer_id","treatment_group"], as_index=False).agg(
        appointments=("appointment_id","count"), no_shows=("no_show","sum"),
        revenue=("revenue","sum"), completed_visits=("completed","sum"),
        cancellation_count=("status", lambda x: int((x=="cancelled").sum()))
    )
    customer_exp["no_show_rate"] = customer_exp["no_shows"] / customer_exp["appointments"]
    customer_exp["revenue_per_visit"] = customer_exp["revenue"] / customer_exp["appointments"]
    # Approximate 60-day repeat/retention: >=2 completed visits in the experiment observation window.
    customer_exp["retained_60d"] = (customer_exp["completed_visits"] >= 2).astype(int)
    customer_exp.to_csv(processed/"experiment_customer_level.csv", index=False)

    t = customer_exp[customer_exp.treatment_group=="Treatment"]
    c = customer_exp[customer_exp.treatment_group=="Control"]
    n_t, n_c = len(t), len(c)
    observed_ratio = n_t / max(n_c,1)
    srm_p = float(stats.chisquare([n_t,n_c], f_exp=[(n_t+n_c)/2,(n_t+n_c)/2]).pvalue)
    balance_rows = []
    for col in ["appointments","no_show_rate","revenue_per_visit","retained_60d"]:
        balance_rows.append({"variable":col,"treatment_mean":float(t[col].mean()),"control_mean":float(c[col].mean()),
                             "difference":float(t[col].mean()-c[col].mean())})
    pd.DataFrame(balance_rows).to_csv(out/"experiment_group_summary.csv", index=False)

    # Primary metric: no-show rate, customer-level mean. Difference is Treatment - Control.
    primary_diff, primary_ci = _bootstrap_diff(t.no_show_rate, c.no_show_rate, seed=11)
    # Welch test on customer-level no-show rates respects customer-level randomization.
    no_show_test = stats.ttest_ind(t.no_show_rate, c.no_show_rate, equal_var=False, nan_policy="omit")
    p_no_show = float(no_show_test.pvalue)
    # Secondary outcomes: Welch t-test and bootstrap CI.
    secondary = {}
    for metric in ["revenue_per_visit","retained_60d"]:
        diff, ci = _bootstrap_diff(t[metric], c[metric], seed=19 if metric=="revenue_per_visit" else 23)
        test = stats.ttest_ind(t[metric], c[metric], equal_var=False, nan_policy="omit")
        secondary[metric] = {"treatment_mean":float(t[metric].mean()),"control_mean":float(c[metric].mean()),
                             "difference_treatment_minus_control":diff,"bootstrap_95_ci":ci,"welch_p_value":float(test.pvalue)}

    # Sample size and duration: illustrative baseline and MDE, explicitly documented.
    required_n = _two_prop_sample_size(.20, .15)
    eligible_customers_daily = max(1, int(customer_exp.customer_id.nunique() / 61))
    duration_days = int(np.ceil((2*required_n) / eligible_customers_daily))

    ab_results = {
        "analysis_unit":"customer (randomized at customer level)",
        "primary_metric":"no-show rate",
        "treatment_customers":int(n_t),"control_customers":int(n_c),
        "treatment_no_show_rate":float(t.no_shows.sum()/max(t.appointments.sum(),1)),
        "control_no_show_rate":float(c.no_shows.sum()/max(c.appointments.sum(),1)),
        "primary_difference_treatment_minus_control":primary_diff,
        "primary_bootstrap_95_ci":primary_ci,
        "customer_level_welch_p_value":float(p_no_show),
        "srm_chi_square_p_value":srm_p,
        "observed_treatment_to_control_customer_ratio":float(observed_ratio),
        "secondary_metrics":secondary,
        "sample_size_assumptions":{"baseline_control_rate":.20,"target_treatment_rate":.15,"alpha":.05,"power":.80,
                                   "required_customers_per_group":required_n,"observed_eligible_customers_per_day":eligible_customers_daily,
                                   "estimated_days_for_required_customers":duration_days}
    }
    (out/"ab_test_results.json").write_text(json.dumps(ab_results, indent=2))

    # Segment effects: customer segment joined from customer master.
    seg = customer_exp.merge(customers[["customer_id","customer_segment"]], on="customer_id", how="left")
    segment_rows = []
    for name, grp in seg.groupby("customer_segment"):
        a, b = grp[grp.treatment_group=="Treatment"], grp[grp.treatment_group=="Control"]
        if len(a) > 1 and len(b) > 1:
            d, ci = _bootstrap_diff(a.no_show_rate, b.no_show_rate, reps=700, seed=31)
            segment_rows.append({"segment":name,"treatment_n":len(a),"control_n":len(b),
                                 "treatment_no_show_rate":a.no_show_rate.mean(),
                                 "control_no_show_rate":b.no_show_rate.mean(),
                                 "difference_treatment_minus_control":d,"bootstrap_95_ci":ci})
    pd.DataFrame(segment_rows).to_csv(out/"segment_results.csv", index=False)

    # Historical DiD: outlet-month panel; intervention at 2026-07-01.
    panel = ap.copy()
    panel["post"] = (panel["appointment_date"] >= pd.Timestamp("2026-07-01")).astype(int)
    panel["treat"] = (panel["region"] == "Treatment").astype(int)
    panel["month_period"] = panel["appointment_date"].dt.to_period("M").astype(str)
    did = panel.groupby(["outlet_id","region","treat","post","month_period"], as_index=False).agg(
        revenue_per_visit=("revenue","mean"), no_show_rate=("no_show","mean"), appointments=("appointment_id","count")
    )
    did["month_date"] = pd.to_datetime(did["month_period"] + "-01")
    did["month_index"] = (did["month_date"].dt.year-2025)*12 + did["month_date"].dt.month
    model = smf.ols("revenue_per_visit ~ treat * post + C(outlet_id) + C(month_period)", data=did).fit(
        cov_type="cluster", cov_kwds={"groups":did["outlet_id"]}
    )
    did_effect = float(model.params.get("treat:post", np.nan))
    did_ci = [float(model.conf_int().loc["treat:post",0]), float(model.conf_int().loc["treat:post",1])]
    # Pre-treatment parallel-trends diagnostic: treatment-specific linear slope before July 2026.
    pre = did[did.post==0].copy()
    pre_model = smf.ols("revenue_per_visit ~ treat * month_index + C(outlet_id) + C(month_period)", data=pre).fit(
        cov_type="cluster", cov_kwds={"groups":pre["outlet_id"]}
    )
    trend_term = "treat:month_index"
    trend_p = float(pre_model.pvalues.get(trend_term, np.nan))
    trend_ci = [float(pre_model.conf_int().loc[trend_term,0]), float(pre_model.conf_int().loc[trend_term,1])]
    did_summary = {
        "outcome":"outlet-month revenue per appointment",
        "treatment_outlets":11,"control_outlets":11,"intervention_date":"2026-07-01",
        "did_estimate":did_effect,"cluster_robust_95_ci":did_ci,"did_p_value":float(model.pvalues.get("treat:post",np.nan)),
        "pretrend_treat_slope_difference":float(pre_model.params.get(trend_term,np.nan)),
        "pretrend_95_ci":trend_ci,"pretrend_p_value":trend_p,
        "parallel_trends_diagnostic":"No statistically significant differential linear pre-trend at 5%." if trend_p >= .05 else
                                     "Evidence of differential linear pre-trends; DiD causal interpretation is weakened.",
        "caveat":"A non-significant pre-trend test does not prove parallel trends."
    }
    (out/"did_results.json").write_text(json.dumps(did_summary, indent=2))

    did_plot = did.groupby(["month_period","region"],as_index=False).revenue_per_visit.mean()
    plt.figure(figsize=(11,5)); sns.lineplot(data=did_plot,x="month_period",y="revenue_per_visit",hue="region",marker="o")
    plt.axvline(x=18.5, color="black", linestyle="--", label="Initiative start")
    plt.xticks(rotation=60); plt.title("DiD: treatment and control outlet trends"); plt.tight_layout()
    plt.savefig(viz/"did_parallel_trends.png",dpi=150); plt.close()

    # CUPED on customer-level revenue per visit using pre-experiment customer baseline spend.
    cup = customer_exp.merge(customers[["customer_id","baseline_spend"]], on="customer_id", how="left").dropna()
    x, y = cup["baseline_spend"].to_numpy(), cup["revenue_per_visit"].to_numpy()
    theta = float(np.cov(y,x,ddof=1)[0,1] / max(np.var(x,ddof=1),1e-9))
    cup["cuped_outcome"] = y - theta*(x-x.mean())
    var_before = float(np.var(y,ddof=1)); var_after = float(np.var(cup.cuped_outcome,ddof=1))
    variance_reduction = max(0.0, 1-var_after/max(var_before,1e-12))
    cuped_t = cup[cup.treatment_group=="Treatment"].cuped_outcome
    cuped_c = cup[cup.treatment_group=="Control"].cuped_outcome
    cuped_test = stats.ttest_ind(cuped_t,cuped_c,equal_var=False)
    cuped_summary = {
        "covariate":"customer baseline_spend",
        "theta":theta,"variance_before":var_before,"variance_after":var_after,
        "estimated_variance_reduction_percent":100*variance_reduction,
        "approx_sample_size_reduction_percent":100*variance_reduction,
        "approx_duration_reduction_percent":100*variance_reduction,
        "cuped_treatment_minus_control":float(cuped_t.mean()-cuped_c.mean()),
        "cuped_welch_p_value":float(cuped_test.pvalue),
        "note":"Variance reduction and duration savings are estimates from synthetic data; validate covariate timing and predictive relationship before real deployment."
    }
    (out/"cuped_results.json").write_text(json.dumps(cuped_summary,indent=2))

    # Product sales descriptive summary.
    product_summary = sales.groupby("product_name",as_index=False).agg(
        units=("quantity","sum"), sales=("sales_amount","sum")
    ).sort_values("sales",ascending=False)
    product_summary.to_csv(out/"product_sales_summary.csv",index=False)

    quality.update({"rows_after_deduplication":int(len(ap)),
                    "walk_in_or_missing_customer_rows":int(ap.customer_id.isna().sum()),
                    "no_show_rows_derived_from_cancellation_reason":int(ap.no_show.sum()),
                    "clean_service_categories":int(ap.service_name.nunique())})
    (out/"data_quality_summary.json").write_text(json.dumps(quality,indent=2))
    summary = {"data_quality":quality,"ab_test":ab_results,"did":did_summary,"cuped":cuped_summary,
               "interpretation_warning":"Synthetic demonstration only. Do not report these estimates as real salon findings."}
    (out/"analysis_summary.json").write_text(json.dumps(summary,indent=2))
    print("Analysis complete. Key files written to outputs/ and visualizations/.")
    return summary
