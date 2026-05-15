# 1. Project overview and requirements
"""
IBI Group Project: ward occupancy, infection wave, vaccination effect,
and an additional infection-growth assessment.

Run this file directly:
    python IBI-Group_Project.py

Requirements:
    numpy
    scipy
    matplotlib
"""

from __future__ import annotations

import os
from html import escape
from pathlib import Path


# 2. Configure output folders and runtime cache
OUTPUT_DIR = Path(__file__).with_name("ibi_outputs")
CACHE_DIR = OUTPUT_DIR / "runtime_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# 3. Define shared day labels
DAYS = np.arange(1, 8)
DAY_LABELS = [f"Day {day}" for day in DAYS]


# 4. Validate weekly input data
def validate_week_data(values, name):
    """Check that one input list contains seven non-negative daily counts."""
    values = np.asarray(values, dtype=float)

    if values.size != 7:
        raise ValueError(f"{name} must contain exactly seven values.")
    if np.any(values < 0):
        raise ValueError(f"{name} cannot contain negative values.")
    if not np.all(values == values.astype(int)):
        raise ValueError(f"{name} must contain whole-number patient counts.")

    return values.astype(int)


# 5. Task 1: calculate ward occupancy
def calculate_ward_occupancy(admissions, discharges):
    """
    Task 1: calculate the number of patients on the ward for each day.

    The ward is assumed to start with no patients, as stated in the assignment.
    """
    admissions = validate_week_data(admissions, "Admissions")
    discharges = validate_week_data(discharges, "Discharges")
    occupancy = np.cumsum(admissions - discharges)

    if np.any(occupancy < 0):
        raise ValueError("Discharges cannot exceed the number of patients on the ward.")

    return occupancy


# 6. Task 1: plot ward occupancy
def plot_ward_occupancy(admissions, discharges, output_path):
    """Create a clearly labelled graph for Task 1."""
    admissions = validate_week_data(admissions, "Admissions")
    discharges = validate_week_data(discharges, "Discharges")
    occupancy = calculate_ward_occupancy(admissions, discharges)

    plt.figure(figsize=(8.5, 5))
    plt.bar(DAYS - 0.18, admissions, width=0.36, label="Admissions", alpha=0.55)
    plt.bar(DAYS + 0.18, discharges, width=0.36, label="Discharges", alpha=0.55)
    plt.plot(DAYS, occupancy, marker="o", linewidth=2.5, label="Ward occupancy")

    plt.title("Ward Occupancy Across Seven Days", weight="bold")
    plt.xlabel("Day")
    plt.ylabel("Number of patients")
    plt.xticks(DAYS, DAY_LABELS)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

    return occupancy


# 7. Task 2: analyse infection wave
def analyse_infection_wave(admissions, discharges):
    """
    Task 2: find the greatest daily increase and decide whether the peak
    increase appears to have passed.
    """
    admissions = validate_week_data(admissions, "Admissions")
    discharges = validate_week_data(discharges, "Discharges")
    calculate_ward_occupancy(admissions, discharges)

    net_changes = admissions - discharges
    max_index = int(np.argmax(net_changes))
    largest_net_change = int(net_changes[max_index])
    greatest_positive_increase = max(largest_net_change, 0)
    peak_passed = max_index < 6

    if largest_net_change <= 0:
        message = "No positive daily increase was observed."
        peak_passed = True
    elif peak_passed:
        message = "The greatest increase occurred before Day 7, so the peak increase appears to have passed."
    else:
        message = "The greatest increase occurred on Day 7, so the peak has not clearly passed."

    return {
        "daily_net_changes": net_changes.tolist(),
        "largest_net_change": largest_net_change,
        "greatest_positive_increase": greatest_positive_increase,
        "day_of_largest_net_change": f"Day {max_index + 1}",
        "peak_passed": peak_passed,
        "interpretation": message,
    }


# 8. Task 3: assess vaccination effectiveness
def assess_vaccination_effectiveness(
    baseline_admissions,
    baseline_discharges,
    vaccine_admissions,
    vaccine_discharges,
):
    """
    Task 3: decide whether vaccination reduced overall ward occupancy.

    Patient-days are used as the main comparison because they summarize the
    total occupancy burden across the whole week.
    """
    baseline_occupancy = calculate_ward_occupancy(
        baseline_admissions, baseline_discharges
    )
    vaccine_occupancy = calculate_ward_occupancy(vaccine_admissions, vaccine_discharges)

    baseline_patient_days = int(np.sum(baseline_occupancy))
    vaccine_patient_days = int(np.sum(vaccine_occupancy))
    reduction_percent = (
        (baseline_patient_days - vaccine_patient_days) / baseline_patient_days * 100
        if baseline_patient_days > 0
        else 0
    )
    vaccine_effective = vaccine_patient_days < baseline_patient_days

    if vaccine_effective:
        message = "Vaccination reduced total patient-days, suggesting lower overall ward occupancy."
    elif vaccine_patient_days == baseline_patient_days:
        message = "Total patient-days were unchanged, so no clear reduction was observed."
    else:
        message = "Total patient-days increased, so these data do not support reduced occupancy."

    return {
        "baseline_occupancy": baseline_occupancy.tolist(),
        "vaccine_occupancy": vaccine_occupancy.tolist(),
        "baseline_patient_days": baseline_patient_days,
        "vaccine_patient_days": vaccine_patient_days,
        "baseline_mean_occupancy": round(float(np.mean(baseline_occupancy)), 2),
        "vaccine_mean_occupancy": round(float(np.mean(vaccine_occupancy)), 2),
        "baseline_peak_occupancy": int(np.max(baseline_occupancy)),
        "vaccine_peak_occupancy": int(np.max(vaccine_occupancy)),
        "occupancy_reduction_percent": round(float(reduction_percent), 1),
        "vaccine_effective": vaccine_effective,
        "interpretation": message,
    }


# 9. Task 5 level 1: calculate admission growth ratios
def calculate_admission_growth_ratios(admissions):
    """
    Task 5, level 1: calculate an admission-based growth proxy.

    This is not a true epidemiological R number. It is a simple indicator of
    whether hospital admissions are increasing or decreasing from day to day,
    and is used before choosing whether an exponential model is appropriate.
    """
    admissions = validate_week_data(admissions, "Admissions")
    ratios = []

    for previous, current in zip(admissions[:-1], admissions[1:]):
        ratios.append(np.nan if previous == 0 else current / previous)

    return np.asarray(ratios, dtype=float)


# 10. Task 5 level 2: fit exponential growth model
def fit_exponential_growth(admissions):
    """
    Task 5, level 2: fit early sustained growth to y = A * exp(r * day).

    Taking logs gives log(y) = log(A) + r * day, so scipy's linear regression
    estimates the exponential growth rate r. This is more precise than the
    day-to-day proxy, but only suitable when admissions are positive and show
    a sustained growth pattern. The model assumes errors are roughly even on
    the log scale, so it is best treated as a compact trend estimate rather
    than proof of the exact biological growth process.
    """
    admissions = validate_week_data(admissions, "Admissions")
    if np.any(admissions <= 0):
        raise ValueError("Exponential modelling requires positive admissions.")

    time = np.arange(len(admissions))
    log_admissions = np.log(admissions)
    regression = stats.linregress(time, log_admissions)

    growth_rate = regression.slope
    fitted_initial = np.exp(regression.intercept)
    fitted_admissions = fitted_initial * np.exp(growth_rate * time)
    confidence_interval = stats.t.interval(
        confidence=0.95,
        df=len(admissions) - 2,
        loc=growth_rate,
        scale=regression.stderr,
    )
    doubling_time = np.log(2) / growth_rate if growth_rate > 0 else None

    if growth_rate > 0.08:
        message = "The model suggests clear exponential growth in admissions."
    elif growth_rate > 0:
        message = "The model suggests slow growth in admissions."
    else:
        message = "The model suggests stable or declining admissions."

    return {
        "growth_rate_per_day": round(float(growth_rate), 4),
        "approximate_95_ci": tuple(round(float(value), 4) for value in confidence_interval),
        "fitted_initial_admissions": round(float(fitted_initial), 2),
        "doubling_time_days": None if doubling_time is None else round(float(doubling_time), 2),
        "r_squared_log_scale": round(float(regression.rvalue**2), 3),
        "fitted_admissions": fitted_admissions,
        "interpretation": message,
    }


# 11. Task 5: combine proxy and model
def assess_infection_growth(admissions, min_mean_ratio=1.05, min_increasing_days=4):
    """
    Task 5: combine a simple growth proxy with a conditional exponential model.

    This keeps the additional function biologically meaningful while still
    showing a more advanced statistical method when the data justify it. The
    model thresholds are explicit so they can be explained during the demo.
    Exponential growth is treated as an early-outbreak follow-up model, not as
    the only possible epidemic shape.
    """
    admissions = validate_week_data(admissions, "Admissions")
    ratios = calculate_admission_growth_ratios(admissions)
    valid_ratios = ratios[~np.isnan(ratios)]
    mean_ratio = float(np.mean(valid_ratios)) if valid_ratios.size else np.nan
    increasing_days = int(np.sum(valid_ratios > 1))

    if np.isnan(mean_ratio):
        trend = "Insufficient data"
        preliminary_conclusion = "A preliminary trend cannot be made from these ratios."
    elif mean_ratio > 1.05:
        trend = "Admissions are generally increasing"
        preliminary_conclusion = "The admission-growth proxy suggests rising admission pressure."
    elif mean_ratio < 0.95:
        trend = "Admissions are generally decreasing"
        preliminary_conclusion = "The admission-growth proxy suggests falling admission pressure."
    else:
        trend = "Admissions are broadly stable"
        preliminary_conclusion = "The admission-growth proxy suggests broadly stable admission pressure."

    should_model = (
        np.all(admissions > 0)
        and valid_ratios.size >= 5
        and mean_ratio > min_mean_ratio
        and increasing_days >= min_increasing_days
    )

    if should_model:
        model = fit_exponential_growth(admissions)
        model_reason = (
            "Sustained early growth was detected, so the exponential model was "
            "used as a follow-up estimate of growth rate and doubling time."
        )
        limitation_note = (
            "This does not claim that all outbreaks are exponential. It is an "
            "early-growth approximation for admissions that pass the screening step."
        )
    else:
        model = None
        model_reason = (
            "The exponential model was not fitted because the data did not meet "
            f"the preset criteria: mean growth ratio > {min_mean_ratio} and at "
            f"least {min_increasing_days} increasing days."
        )
        limitation_note = (
            "This is reported as a preliminary trend only. More detailed model "
            "analysis would require more observations, especially if the curve "
            "shows plateauing, turning points, or multiple growth phases."
        )

    return {
        "analysis_scope": "Two-stage admission-based early outbreak assessment",
        "admission_growth_ratios": np.round(ratios, 3).tolist(),
        "mean_admission_growth_ratio": None if np.isnan(mean_ratio) else round(mean_ratio, 3),
        "growth_trend": trend,
        "preliminary_conclusion": preliminary_conclusion,
        "increasing_days": increasing_days,
        "model_criteria": {
            "min_mean_growth_ratio": min_mean_ratio,
            "min_increasing_days": min_increasing_days,
        },
        "model_used": should_model,
        "model_reason": model_reason,
        "model_result": model,
        "limitation_note": limitation_note,
    }


# 12. Task 5: plot growth assessment
def plot_infection_growth(admissions, output_path):
    """Plot the Task 5 admission data, growth ratios, and fitted model if used."""
    admissions = validate_week_data(admissions, "Admissions")
    assessment = assess_infection_growth(admissions)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7), gridspec_kw={"height_ratios": [1.1, 1]})

    ax1.plot(DAYS, admissions, marker="o", linewidth=2.5, label="Observed admissions")
    if assessment["model_used"]:
        ax1.plot(
            DAYS,
            assessment["model_result"]["fitted_admissions"],
            linestyle="--",
            linewidth=2,
            label="Exponential model fit",
        )

    ratios = np.asarray(assessment["admission_growth_ratios"], dtype=float)
    ratio_days = DAYS[1:]
    colors = np.where(ratios > 1, "#D55E00", "#009E73")
    ax2.bar(ratio_days, ratios, color=colors, alpha=0.75)
    ax2.axhline(1, color="black", linestyle=":", linewidth=1)

    ax1.set_title("Task 5: Infection Growth Assessment", weight="bold")
    ax1.set_ylabel("Daily admissions")
    ax1.set_xticks(DAYS, DAY_LABELS)
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend()

    ax2.set_title("Admission growth ratio compared with previous day")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Growth ratio")
    ax2.set_xticks(ratio_days, [f"Day {day}" for day in ratio_days])
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

    clean_assessment = assessment.copy()
    if clean_assessment["model_result"] is not None:
        clean_assessment["model_result"] = clean_assessment["model_result"].copy()
        clean_assessment["model_result"].pop("fitted_admissions")
    return clean_assessment


# 13. Generate an HTML report for presentation
def format_html_value(value):
    """Convert nested result values into compact HTML."""
    if isinstance(value, dict):
        items = "\n".join(
            f"<li><strong>{escape(str(key))}:</strong> {format_html_value(item)}</li>"
            for key, item in value.items()
        )
        return f"<ul>{items}</ul>"
    if isinstance(value, list) or isinstance(value, tuple):
        return escape(", ".join(str(item) for item in value))
    return escape(str(value))


def html_result_block(title, result):
    """Create one result card for the HTML report."""
    if isinstance(result, dict):
        body = "\n".join(
            f"<li><strong>{escape(str(key))}:</strong> {format_html_value(value)}</li>"
            for key, value in result.items()
        )
        body = f"<ul>{body}</ul>"
    else:
        body = f"<p>{format_html_value(result)}</p>"

    return f"""
        <section class="card">
            <h2>{escape(title)}</h2>
            {body}
        </section>
    """


def generate_html_report(
    output_path,
    task1_result,
    task2_result,
    task3_result,
    task5_result,
):
    """Save a polished, offline HTML report for the demo."""
    html = f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>IBI Group Project Report</title>
    <style>
        body {{
            margin: 0;
            font-family: Arial, Helvetica, sans-serif;
            color: #1f2933;
            background: #f5f7fa;
            line-height: 1.5;
        }}
        header {{
            background: #16324f;
            color: white;
            padding: 28px 40px;
        }}
        header h1 {{
            margin: 0 0 8px 0;
            font-size: 30px;
        }}
        header p {{
            margin: 0;
            max-width: 920px;
            color: #d8e6f3;
        }}
        main {{
            max-width: 1120px;
            margin: 0 auto;
            padding: 28px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(310px, 1fr));
            gap: 18px;
        }}
        .card {{
            background: white;
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            padding: 18px 20px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
        }}
        h2 {{
            margin: 0 0 12px 0;
            font-size: 19px;
            color: #16324f;
        }}
        ul {{
            margin: 0;
            padding-left: 20px;
        }}
        li {{
            margin: 7px 0;
        }}
        .figure {{
            margin: 24px 0;
        }}
        .figure img {{
            width: 100%;
            max-width: 980px;
            display: block;
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            background: white;
        }}
        .note {{
            border-left: 5px solid #2f80ed;
            background: #edf5ff;
        }}
    </style>
</head>
<body>
    <header>
        <h1>IBI Group Project Demo Report</h1>
        <p>
            Ward occupancy, infection wave, vaccination effect, and a two-stage
            admission-based early outbreak growth assessment.
        </p>
    </header>
    <main>
        <section class="card note">
            <h2>Task 5 Narrative</h2>
            <p>
                The additional function does not assume that every outbreak is
                exponential. It first checks admission growth ratios as a simple
                screening step. The exponential model is only fitted when the
                data show sustained early growth.
            </p>
        </section>

        <div class="grid">
            {html_result_block("Task 1: Ward Occupancy", task1_result)}
            {html_result_block("Task 2: Infection Wave", task2_result)}
            {html_result_block("Task 3: Vaccination Effectiveness", task3_result)}
            {html_result_block("Task 5: Growth Assessment", task5_result)}
        </div>

        <section class="figure">
            <h2>Task 1 Figure</h2>
            <img src="task1_ward_occupancy.png" alt="Ward occupancy graph">
        </section>

        <section class="figure">
            <h2>Task 5 Figure</h2>
            <img src="task5_infection_growth.png" alt="Infection growth assessment graph">
        </section>
    </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


# 14. Print demo results clearly
def print_result(title, result):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    if isinstance(result, dict):
        for key, value in result.items():
            print(f"{key}: {value}")
    else:
        print(result)


# 15. Run all tasks with example data
def run_demo():
    """Run a complete example that a marker can execute without editing code."""
    output_dir = OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    baseline_admissions = [5, 8, 12, 10, 7, 6, 4]
    baseline_discharges = [2, 3, 4, 6, 8, 5, 3]
    vaccine_admissions = [3, 5, 6, 5, 4, 3, 2]
    vaccine_discharges = [2, 3, 4, 5, 5, 4, 3]
    task5_admissions = [4, 5, 7, 9, 12, 15, 20]

    task1_output = output_dir / "task1_ward_occupancy.png"
    task5_output = output_dir / "task5_infection_growth.png"
    report_output = output_dir / "report.html"

    occupancy = plot_ward_occupancy(baseline_admissions, baseline_discharges, task1_output)
    task1_result = {"daily_occupancy": occupancy.tolist()}
    print_result("Task 1: Ward occupancy", task1_result)
    print(f"Task 1 graph saved to: {task1_output}")

    wave_result = analyse_infection_wave(baseline_admissions, baseline_discharges)
    print_result("Task 2: Infection wave", wave_result)

    vaccine_result = assess_vaccination_effectiveness(
        baseline_admissions,
        baseline_discharges,
        vaccine_admissions,
        vaccine_discharges,
    )
    print_result("Task 3: Vaccination effectiveness", vaccine_result)

    growth_result = plot_infection_growth(task5_admissions, task5_output)
    print_result("Task 5: Additional infection-growth assessment", growth_result)
    print(f"Task 5 graph saved to: {task5_output}")

    report_path = generate_html_report(
        report_output,
        task1_result,
        wave_result,
        vaccine_result,
        growth_result,
    )
    print(f"HTML report saved to: {report_path}")


# Start the demo when this file is run directly
if __name__ == "__main__":
    run_demo()
