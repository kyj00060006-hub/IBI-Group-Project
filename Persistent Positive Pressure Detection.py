#Persistent Positive Pressure Detection
import matplotlib.pyplot as plt

def persistent_positive_pressure(admissions, discharges):
    # Initialize variables to track periods and totals
    current_period = 0
    longest_period = 0
    total_positive = 0
    is_increasing = []

    # Loop through 7 days of data
    for i in range(7):
        a = admissions[i]
        d = discharges[i]

        # Check if the number of patients is increasing
        if a > d:
            current_period = current_period + 1
            total_positive = total_positive + 1
            is_increasing.append(True)

            # Update the longest increasing period
            if current_period > longest_period:
                longest_period = current_period
        else:
            # Reset the current period when increase stops
            current_period = 0
            is_increasing.append(False)

    # Check if the outbreak is still increasing on the last day
    last_increasing = admissions[6] > discharges[6]

    # Classify outbreak severity
    if longest_period >= 5:
        severity = "Severe outbreak"
    elif longest_period >= 3:
        severity = "Moderate outbreak"
    else:
        severity = "Mild outbreak"

    # Display the results
    print("Longest consecutive increasing period:", longest_period, "days")
    print("Total increasing days:", total_positive, "days")
    print("Outbreak severity:", severity)
    print("Still increasing on last day:", last_increasing)

    # Prepare day labels for the plot
    days = [1, 2, 3, 4, 5, 6, 7]

    # Create a plot of admissions and discharges
    plt.figure(figsize=(8, 4))
    plt.plot(days, admissions, marker='o', label='Admissions')
    plt.plot(days, discharges, marker='s', label='Discharges')

    # Shade the days where patients are increasing
    for i in range(7):
        if is_increasing[i]:
            plt.axvspan(i+0.5, i+1.5, color='red', alpha=0.15)

    # Add plot labels and show the chart
    plt.title('Daily Admissions & Discharges')
    plt.xlabel('Day')
    plt.ylabel('Number of patients')
    plt.xticks(days)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# Test data
admissions = [3, 5, 6, 4, 2, 1, 0]
discharges = [1, 2, 2, 3, 1, 2, 1]
persistent_positive_pressure(admissions, discharges)
