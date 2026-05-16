import matplotlib.pyplot as plt

def recovery_ratio_trend_analysis(admissions, discharges):
    # Initialize lists to store daily occupancy and recovery ratios
    occupancy = [0] * 7
    recovery_ratios = [0] * 7
    
    # Variables to detect consecutive low recovery periods
    low_period = 0
    prolonged_stress = False

    # Calculate daily ward occupancy (current patients in ward)
    occupancy[0] = admissions[0] - discharges[0]

    # Loop to compute occupancy for days 1 to 6
    for i in range(1, 7):
        occupancy[i] = occupancy[i-1] + admissions[i] - discharges[i]

    # Calculate daily recovery ratio (day 1 to 6 only)
    # Formula: today's discharges / previous day's occupancy
    for i in range(1, 7):
        if occupancy[i-1] <= 0:
            recovery_ratios[i] = 0
        else:
            recovery_ratios[i] = discharges[i] / occupancy[i-1]

    # Check for 3+ consecutive days with recovery ratio < 0.2
    for i in range(1, 7):
        if recovery_ratios[i] < 0.2:
            low_period += 1
            if low_period >= 3:
                prolonged_stress = True
        else:
            low_period = 0

    # Determine overall recovery trend
    valid_ratios = recovery_ratios[1:7]
    trend = "Stable"

    # Check if trend is Improving
    if all(valid_ratios[i] <= valid_ratios[i+1] for i in range(len(valid_ratios)-1)):
        trend = "Improving"
    # Check if trend is Declining
    elif all(valid_ratios[i] >= valid_ratios[i+1] for i in range(len(valid_ratios)-1)):
        trend = "Declining"
    # Check if trend is Fluctuating
    elif max(valid_ratios) - min(valid_ratios) >= 0.15:
        trend = "Fluctuating"
    # Otherwise, Stable
    else:
        trend = "Stable"

    # Display analysis results
    print("===== Task 6: Recovery Ratio Trend Analysis =====")
    print("Daily recovery ratios (Day1-Day6):", [round(ratio, 2) for ratio in valid_ratios])
    print("Recovery trend:", trend)
    print("Prolonged stress (3+ days low recovery):", prolonged_stress)

    if prolonged_stress:
        print("Warning: Slow recovery - ward under prolonged stress!")
    else:
        print("Normal: Recovery performance is within acceptable range.")

    # Visualize recovery ratio trend
    days = [1, 2, 3, 4, 5, 6]
    plt.figure(figsize=(8, 4))
    plt.plot(days, valid_ratios, marker='o', color='blue', linewidth=2, label='Recovery Ratio')
    plt.axhline(y=0.2, color='red', linestyle='--', label='Low threshold (0.2)')
    plt.title('7-Day Recovery Ratio Trend')
    plt.xlabel('Day')
    plt.ylabel('Recovery Ratio')
    plt.xticks(days)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# Test data for 7 days
admissions = [2, 5, 6, 4, 3, 2, 1]
discharges = [0, 1, 2, 3, 2, 1, 1]

# Run the function
recovery_ratio_trend_analysis(admissions, discharges)
