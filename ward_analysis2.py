"""
Hospital Ward Analysis

Designed in leading of GREATEST HAO YUN QI
"""

# input 'bash setup.sh '
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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
    pre_mean  = np.mean(pre_count)
    post_mean = np.mean(post_count)
    reduction = pre_mean - post_mean

    res = stats.ttest_ind(pre_count, post_count, equal_var=False, alternative='greater')
    t_stat  = res.statistic
    p_value = res.pvalue

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
    
# Task 5: Transmission Rate Approximation (R-value)

def estimate_r_value(admissions):
    """
    Estimates a daily approximate R-value (transmission rate) based on the ratio of new admissions on consecutive days.

    R_approx(day i) = admissions[i] / admissions[i-1]

    Interpretation:
      R>1.0 → infection spreading
      R=1.0 → stable
      R<1.0 → infection declining

    Parameters:
        admissions (list): daily admission counts (length 7)

    Returns:
        dict with r_values, mean_r, trend description, and per-day labels
    """
    r_values=[]
    labels=[]

    for i in range(1,len(admissions)):
        if admissions[i-1]==0:
            r=None  # avoid division by zero
        else:
            r=round(admissions[i]/admissions[i-1],3)
        r_values.append(r)
        labels.append(f"Day {i+1}")

    valid_r=[r for r in r_values if r is not None]
    mean_r=round(np.mean(valid_r),3) if valid_r else None

    # Trend: compare second half vs first half of R series
    mid=len(valid_r)//2
    first_half_mean=np.mean(valid_r[:mid]) if valid_r[:mid] else None
    second_half_mean=np.mean(valid_r[mid:]) if valid_r[mid:] else None

    if first_half_mean is not None and second_half_mean is not None:
        if second_half_mean<first_half_mean:
            trend = "R is DECLINING-transmission slowing down. ✓"
        elif second_half_mean > first_half_mean:
            trend = "R is INCREASING-transmission accelerating. ✗"
        else:
            trend = "R is STABLE-transmission rate unchanged."
    else:
        trend = "Insufficient data to determine trend."

    return {
        "labels":    labels,
        "r_values":  r_values,
        "mean_r":    mean_r,
        "trend":     trend,
    }


def plot_r_value(r_result):
    labels   =r_result["labels"]
    r_values =r_result["r_values"]
    mean_r   =r_result["mean_r"]

    x=np.arange(len(labels))
    colors=[]
    for r in r_values:
        if r is None:
            colors.append("#AAAAAA")
        elif r>1.0:
            colors.append("#E74C3C")#red: spreading
        elif r<1.0:
            colors.append("#2ECC71")#green: declining
        else:
            colors.append("#F1C40F")#yellow: stable

    fig, ax=plt.subplots(figsize=(9, 5))
    bars=ax.bar(x,[r if r is not None else 0 for r in r_values],
                  color=colors,alpha=0.85,width=0.5,zorder=2)

    ax.axhline(y=1.0,color="black",linestyle="--",linewidth=1.5,
               label="R =1 (threshold)",zorder=3)
    if mean_r is not None:
        ax.axhline(y=mean_r,color="#8E44AD",linestyle=":",linewidth=1.5,
                   label=f"Mean R = {mean_r}",zorder=3)

    # value labels on bars
    for bar,r in zip(bars,r_values):
        if r is not None:
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.03,
                    str(r),ha="center",va="bottom",fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Approximate R-value")
    ax.set_title("Daily Transmission Rate Approximation (R-value)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("r_value.png",dpi=150)
    plt.show()
    print("Chart saved → r_value.png")


# Demo

print('---郝蕴奇屁股大---')

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

    # Task 5
    print("\n" + "="*50)
    print("TASK 5: Transmission Rate (R-value Approximation)")
    print("="*50)
    print('郝蕴奇'*100)
    r_result=estimate_r_value(admissions)
    for label, r in zip(r_result["labels"],r_result["r_values"]):
        indicator=("↑" if r and r>1 else "↓" if r and r<1 else "→") if r else "N/A"
        print(f"{label}: R ≈ {r if r is not None else 'N/A'}  {indicator}")
    print(f"Mean R (7-day)  :{r_result['mean_r']}")
    print(f"Trend           :{r_result['trend']}")
    plot_r_value(r_result)
