from pathlib import Path
import numpy as np
import pandas as pd

SEED = 20261111
N_APPOINTMENTS = 300_000
N_CUSTOMERS = 45_000
N_OUTLETS = 22

def generate_all(root):
    rng = np.random.default_rng(SEED)
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    outlet_ids = [f"O{i:02d}" for i in range(1, N_OUTLETS + 1)]
    outlets = pd.DataFrame({
        "outlet_id": outlet_ids,
        "outlet_name": [f"Coimbatore Salon {i:02d}" for i in range(1, N_OUTLETS + 1)],
        "region": ["Treatment" if i <= 11 else "Control" for i in range(1, N_OUTLETS + 1)],
        "city": "Coimbatore",
        "capacity_band": rng.choice(["Small", "Medium", "Large"], N_OUTLETS, p=[.25, .5, .25])
    })
    outlets.to_csv(raw / "outlets.csv", index=False)

    customer_ids = [f"C{i:06d}" for i in range(1, N_CUSTOMERS + 1)]
    customers = pd.DataFrame({
        "customer_id": customer_ids,
        "age": rng.integers(18, 66, N_CUSTOMERS),
        "customer_segment": rng.choice(["New", "Returning", "Loyal"], N_CUSTOMERS, p=[.30, .45, .25]),
        "signup_channel": rng.choice(["Walk-in", "Website", "WhatsApp", "Referral"], N_CUSTOMERS),
        "baseline_spend": np.round(rng.gamma(4.0, 180.0, N_CUSTOMERS), 2)
    })
    customers.to_csv(raw / "customers.csv", index=False)

    stylist_ids = [f"S{i:03d}" for i in range(1, 151)]
    stylists = pd.DataFrame({
        "stylist_id": stylist_ids,
        "outlet_id": rng.choice(outlet_ids, 150),
        "experience_years": rng.integers(1, 16, 150),
        "speciality": rng.choice(["Hair", "Skin", "Spa", "Nails"], 150)
    })
    stylists.to_csv(raw / "stylists.csv", index=False)

    services = pd.DataFrame({
        "service_id": [f"SV{i:02d}" for i in range(1, 13)],
        "canonical_service": ["Haircut", "Mens Haircut", "Hair Colour", "Facial", "Cleanup",
                              "Manicure", "Pedicure", "Head Massage", "Hair Spa", "Waxing",
                              "Bridal Package", "Body Spa"],
        "base_price": [350, 250, 1800, 1200, 650, 500, 600, 450, 1500, 700, 6500, 2500]
    })
    services.to_csv(raw / "services.csv", index=False)

    # Customer-level randomized assignment for eligible identified customers.
    assignment = pd.DataFrame({
        "customer_id": customer_ids,
        "treatment_group": rng.choice(["Treatment", "Control"], N_CUSTOMERS)
    })
    assignment.to_csv(raw / "experiment_assignment.csv", index=False)

    # Appointments span 2025-01 to 2026-09 to support pre/post historical analysis.
    dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 638, N_APPOINTMENTS), unit="D"
    )
    cust_idx = rng.integers(0, N_CUSTOMERS, N_APPOINTMENTS)
    cust = np.array(customer_ids, dtype=object)[cust_idx]
    walkin = rng.random(N_APPOINTMENTS) < 0.08
    cust[walkin] = None

    outlet_idx = rng.integers(0, N_OUTLETS, N_APPOINTMENTS)
    outlet = np.array(outlet_ids)[outlet_idx]
    region = np.where(outlet_idx < 11, "Treatment", "Control")
    service_idx = rng.integers(0, len(services), N_APPOINTMENTS)
    base_price = services["base_price"].to_numpy()[service_idx]
    stylist = rng.choice(stylist_ids, N_APPOINTMENTS)

    month = pd.DatetimeIndex(dates).month.to_numpy()
    seasonal = np.where(np.isin(month, [10, 11, 12]), 0.035, 0.0)
    # Historical initiative after 2026-07-01: treatment outlets gain revenue and attendance.
    historical_post = (dates >= np.datetime64("2026-07-01"))
    hist_effect = (region == "Treatment") & historical_post

    # Assign experiment group consistently by customer.
    group_map = dict(zip(assignment.customer_id, assignment.treatment_group))
    exp_group = np.array([group_map.get(x, "Walk-in") if x is not None else "Walk-in" for x in cust])
    experiment_period = (dates >= np.datetime64("2026-08-01")) & (dates <= np.datetime64("2026-09-30"))
    in_exp = experiment_period & (cust != None)
    treatment_reminder = (exp_group == "Treatment") & in_exp

    base_no_show = 0.19 + seasonal - 0.035 * hist_effect - 0.055 * treatment_reminder
    no_show = rng.random(N_APPOINTMENTS) < np.clip(base_no_show, .04, .35)
    cancellation = (~no_show) & (rng.random(N_APPOINTMENTS) < .075)
    status = np.where(no_show | cancellation, "cancelled", "completed")
    reason = np.where(no_show, "no_show", np.where(cancellation, "customer_cancelled", ""))
    # Small portion of service labels intentionally inconsistent across outlets.
    canonical = services["canonical_service"].to_numpy()[service_idx]
    aliases = {"Haircut": ["Hair Cut", "Haircut", "Hair Cutting"],
               "Mens Haircut": ["Men Hair Cut", "Mens Haircut", "Hair cut men"],
               "Facial": ["Facial", "Face cleanup", "Facial Service"]}
    service_name = canonical.copy().astype(object)
    for key, choices in aliases.items():
        mask = canonical == key
        service_name[mask] = rng.choice(choices, mask.sum())
    price = base_price * rng.normal(1.0, .08, N_APPOINTMENTS)
    price *= np.where(hist_effect, 1.025, 1.0)
    price = np.where(status == "completed", price, 0)
    price = np.round(np.maximum(price, 0), 2)

    appointment_df = pd.DataFrame({
        "appointment_id": [f"A{i:07d}" for i in range(1, N_APPOINTMENTS + 1)],
        "appointment_date": dates,
        "customer_id": cust,
        "outlet_id": outlet,
        "stylist_id": stylist,
        "service_name_raw": service_name,
        "status": status,
        "cancellation_reason": reason,
        "revenue": price,
        "region": region,
        "historical_initiative": hist_effect.astype(int),
        "experiment_group": exp_group,
        "experiment_period": in_exp.astype(int)
    })
    # Include a few missing values in optional fields, but do not corrupt key IDs/dates.
    miss = rng.random(N_APPOINTMENTS) < .004
    appointment_df.loc[miss, "stylist_id"] = None
    appointment_df.to_csv(raw / "appointments.csv", index=False)

    # Product sales: separate transaction table.
    n_sales = 65_000
    product_names = np.array(["Shampoo", "Conditioner", "Hair Serum", "Face Wash",
                              "Sunscreen", "Hair Oil", "Face Cream", "Body Lotion"])
    product_sales = pd.DataFrame({
        "sale_id": [f"P{i:07d}" for i in range(1, n_sales + 1)],
        "sale_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 638, n_sales), unit="D"),
        "outlet_id": rng.choice(outlet_ids, n_sales),
        "customer_id": rng.choice(customer_ids, n_sales),
        "product_name": rng.choice(product_names, n_sales),
        "quantity": rng.integers(1, 4, n_sales),
        "sales_amount": np.round(rng.uniform(120, 1600, n_sales), 2)
    })
    product_sales.to_csv(raw / "product_sales.csv", index=False)

    print(f"Created {N_APPOINTMENTS:,} synthetic appointments, {N_CUSTOMERS:,} customers, "
          f"{N_OUTLETS} outlets, 150 stylists and {n_sales:,} product sales.")
