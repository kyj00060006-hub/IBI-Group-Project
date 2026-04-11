"""
Hospital Ward Analysis
"""

import numpy as np
import matplotlib.pyplot as plt


# Task 1: Ward Occupancy

def compute_occupancy(admissions, discharges):
    current = 0
    daily_count = []
    for i in range(7):
        current += admissions[i] - discharges[i]
        daily_count.append(current)
    return daily_count


def plot_occupancy(admissions, discharges, daily_count):
    # Labels
    days = [f"Day {i+1}" for i in range(7)]
    x = np.arange(7) #0-6

    fig, ax1 = plt.subplots(figsize=(10, 5)) # ax1 in fig
    # Bar Chart
    ax1.bar(x - 0.15, admissions, width=0.3, label="Admissions", color="#4E9AF1", alpha=0.8)
    ax1.bar(x + 0.15, discharges, width=0.3, label="Discharges", color="#F17F4E", alpha=0.8)
    ax1.set_ylabel("Admissions / Discharges")
    ax1.set_xticks(x)
    ax1.set_xticklabels(days)

    ax2 = ax1.twinx()
    ax2.plot(x, daily_count, color="#2ECC71", marker="o", linewidth=2.5, label="Occupancy")
    ax2.set_ylabel("In-ward Patients")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    plt.title("Ward Occupancy - 7-Day Overview")
    plt.tight_layout()
    plt.savefig("ward_occupancy.png", dpi=150)
    plt.show()
    print("Chart saved → ward_occupancy.png")


# Task 2: Infection Wave

def analyse_infection_wave(daily_count):
    daily_changes = [daily_count[i] - daily_count[i-1] for i in range(1, 7)]
    max_increase = max(daily_changes)
    peak_day = daily_changes.index(max_increase) + 2  # +2 because changes start from day 2

    if daily_changes[-1] < daily_changes[-2]:
        status = "Peak has PASSED – growth rate is declining."
    else:
        status = "Still GROWING – peak has not yet been reached."

    return {
        "peak_day": peak_day,
        "max_increase": max_increase,
        "status": status,
        "daily_changes": daily_changes,
    }


# Task 3: Vaccination Effectiveness

def assess_vaccination(pre_count, post_count, alpha=0.05):
    from math import lgamma, exp, log

    pre  = np.array(pre_count,  dtype=float)
    post = np.array(post_count, dtype=float)
    n = len(pre)

    pre_mean, post_mean = pre.mean(), post.mean()
    reduction = pre_mean - post_mean

    # Welch's t-statistic and degrees of freedom
    se = np.sqrt(pre.var(ddof=1) / n + post.var(ddof=1) / n)
    t_stat = (pre_mean - post_mean) / se

    df = (pre.var(ddof=1)/n + post.var(ddof=1)/n)**2 / (
         (pre.var(ddof=1)/n)**2/(n-1) + (post.var(ddof=1)/n)**2/(n-1))

    # one-tailed p-value via regularised incomplete beta (no scipy needed)
    def reg_incomplete_beta(x_val, a_val, b_val, terms=200):
        if x_val <= 0: return 0.0
        if x_val >= 1: return 1.0
        lbeta_ab = lgamma(a_val) + lgamma(b_val) - lgamma(a_val + b_val)
        front = exp(a_val * log(x_val) + b_val * log(1 - x_val) - lbeta_ab)
        total, term = 1.0, 1.0
        for m in range(1, terms):
            term *= x_val * (a_val + b_val + m - 1) / (a_val + m)
            total += term
            if abs(term) < 1e-10:
                break
        return front * total / a_val

    x_beta = df / (df + t_stat**2)
    p_value = 0.5 * reg_incomplete_beta(x_beta, df / 2, 0.5)
    if t_stat < 0:
        p_value = 1 - p_value

    if p_value < alpha:
        verdict = (f"Vaccine IS effective: occupancy dropped by {reduction:.1f} "
                   f"patients/day on average (p={p_value:.4f} < alpha={alpha}).")
    else:
        verdict = (f"Vaccine effectiveness NOT confirmed at alpha={alpha} "
                   f"(p={p_value:.4f}). Reduction of {reduction:.1f} may be by chance.")

    return {
        "pre_mean":  round(pre_mean, 2),
        "post_mean": round(post_mean, 2),
        "reduction": round(reduction, 2),
        "t_stat":    round(t_stat, 4),
        "p_value":   round(p_value, 4),
        "verdict":   verdict,
    }


# Demo

if __name__ == "__main__":
    admissions = [5, 8, 12, 10, 7, 6, 4]
    discharges  = [2, 3,  4,  6, 8, 5, 3]

    post_admissions = [3, 5, 6, 4, 3, 2, 2]
    post_discharges = [2, 4, 5, 4, 3, 2, 2]

    # Task 1
    print("=" * 50)
    print("TASK 1 · Ward Occupancy")
    print("=" * 50)
    daily_count = compute_occupancy(admissions, discharges)
    print(f"Daily occupancy: {daily_count}")
    plot_occupancy(admissions, discharges, daily_count)

    # Task 2
    print("\n" + "=" * 50)
    print("TASK 2 · Infection Wave")
    print("=" * 50)
    wave = analyse_infection_wave(daily_count)
    print(f"Day-over-day changes : {wave['daily_changes']}")
    print(f"Largest increase     : +{wave['max_increase']} on Day {wave['peak_day']}")
    print(f"Status               : {wave['status']}")

    # Task 3
    print("\n" + "=" * 50)
    print("TASK 3 · Vaccination Effectiveness")
    print("=" * 50)
    post_count = compute_occupancy(post_admissions, post_discharges)
    vax = assess_vaccination(daily_count, post_count)
    print(f"Pre-vaccine mean  : {vax['pre_mean']}")
    print(f"Post-vaccine mean : {vax['post_mean']}")
    print(f"t-statistic       : {vax['t_stat']}")
    print(f"p-value (1-tail)  : {vax['p_value']}")
    print(f"Verdict → {vax['verdict']}")
