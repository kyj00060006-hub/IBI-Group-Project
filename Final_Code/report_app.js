function formatValue(value) {
    if (value === null || value === undefined) {
        return "None";
    }
    if (Array.isArray(value)) {
        return value.join(", ");
    }
    if (typeof value === "object") {
        return renderList(value);
    }
    return String(value);
}

function renderList(data) {
    const items = Object.entries(data).map(([key, value]) => (
        `<li><strong>${escapeHtml(key)}:</strong> ${formatValue(value)}</li>`
    ));
    return `<ul>${items.join("")}</ul>`;
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
}

function renderCard(title, data) {
    return `
        <section class="card">
            <h2>${escapeHtml(title)}</h2>
            ${renderList(data)}
        </section>
    `;
}

function renderReport() {
    const data = window.REPORT_DATA;
    if (!data) {
        document.getElementById("result-grid").innerHTML = renderCard(
            "Report Data Missing",
            { message: "Run main.py first to generate ibi_outputs/report_data.js." }
        );
        return;
    }

    document.getElementById("result-grid").innerHTML = [
        renderCard("Task 1: Ward Occupancy", data.task1),
        renderCard("Task 2: Infection Wave", data.task2),
        renderCard("Task 3: Vaccination Effectiveness", data.task3),
        renderCard("Task 5: Growth Assessment", data.task5),
    ].join("");

    if (data.figures) {
        document.getElementById("task1-figure").src = data.figures.task1;
        document.getElementById("task2-figure").src = data.figures.task2;
        document.getElementById("task3-figure").src = data.figures.task3;
        document.getElementById("task5-figure").src = data.figures.task5;
    }
}

function parseWeekData(id, label) {
    const values = document.getElementById(id).value
        .split(",")
        .map(item => Number(item.trim()));
    if (values.length !== 7 || values.some(value => !Number.isInteger(value) || value < 0)) {
        throw new Error(label + " must contain exactly seven non-negative integers.");
    }
    return values;
}

function mean(values) {
    return values.reduce((total, value) => total + value, 0) / values.length;
}

function linearRegression(xValues, yValues) {
    const xMean = mean(xValues);
    const yMean = mean(yValues);
    let sumXX = 0;
    let sumXY = 0;
    let sumYY = 0;
    for (let i = 0; i < xValues.length; i++) {
        const dx = xValues[i] - xMean;
        const dy = yValues[i] - yMean;
        sumXX += dx * dx;
        sumXY += dx * dy;
        sumYY += dy * dy;
    }
    const slope = sumXY / sumXX;
    const intercept = yMean - slope * xMean;
    const fitted = xValues.map(x => intercept + slope * x);
    const residuals = yValues.map((y, i) => y - fitted[i]);
    const sse = residuals.reduce((total, value) => total + value * value, 0);
    const mse = sse / (xValues.length - 2);
    const stderr = Math.sqrt(mse / sumXX);
    const rSquared = sumYY === 0 ? 1 : 1 - sse / sumYY;
    return { slope, intercept, stderr, rSquared };
}

function randomInteger(minimum, maximum) {
    return Math.floor(Math.random() * (maximum - minimum + 1)) + minimum;
}

function fillRandomDemoData() {
    const admissions = [];
    const discharges = [];
    let currentOccupancy = 0;

    for (let day = 0; day < 7; day++) {
        const admission = randomInteger(2 + day, 12 + day * 2);
        currentOccupancy += admission;
        const dischargeLimit = Math.min(currentOccupancy, Math.max(1, Math.floor(admission * 0.75)));
        const discharge = randomInteger(0, dischargeLimit);
        currentOccupancy -= discharge;
        admissions.push(admission);
        discharges.push(discharge);
    }

    document.getElementById("admissions-input").value = admissions.join(",");
    document.getElementById("discharges-input").value = discharges.join(",");
    runBrowserDemo();
}

function runBrowserDemo() {
    const output = document.getElementById("tester-output");
    try {
        const admissions = parseWeekData("admissions-input", "Admissions");
        const discharges = parseWeekData("discharges-input", "Discharges");

        let current = 0;
        const occupancy = admissions.map((admission, i) => {
            current += admission - discharges[i];
            if (current < 0) {
                throw new Error("Discharges cannot exceed the number of patients on the ward.");
            }
            return current;
        });

        const netChanges = admissions.map((admission, i) => admission - discharges[i]);
        const largestNetChange = Math.max(...netChanges);
        const largestDays = netChanges
            .map((value, i) => value === largestNetChange ? i + 1 : null)
            .filter(day => day !== null);
        const latestLargestDay = largestDays[largestDays.length - 1];
        const peakPassed = largestNetChange <= 0 || latestLargestDay < 7;

        const ratios = admissions.slice(1).map((value, i) => (
            admissions[i] === 0 ? NaN : value / admissions[i]
        ));
        const validRatios = ratios.filter(value => !Number.isNaN(value));
        const meanRatio = validRatios.length ? mean(validRatios) : NaN;
        const increasingDays = validRatios.filter(value => value > 1).length;
        const shouldModel = admissions.every(value => value > 0)
            && validRatios.length >= 5
            && meanRatio > 1.05
            && increasingDays >= 4;

        let modelText = "Exponential model: not fitted because sustained early growth criteria were not met.";
        if (shouldModel) {
            const xValues = admissions.map((_, i) => i);
            const logAdmissions = admissions.map(value => Math.log(value));
            const model = linearRegression(xValues, logAdmissions);
            const growthRate = model.slope;
            const doublingTime = growthRate > 0 ? Math.log(2) / growthRate : null;
            const tCritical = 2.571;
            const ciLow = growthRate - tCritical * model.stderr;
            const ciHigh = growthRate + tCritical * model.stderr;
            modelText = [
                "Exponential model: fitted as early-growth follow-up",
                "growth_rate_per_day = " + growthRate.toFixed(4),
                "approximate_95_ci = (" + ciLow.toFixed(4) + ", " + ciHigh.toFixed(4) + ")",
                "doubling_time_days = " + (doublingTime === null ? "None" : doublingTime.toFixed(2)),
                "r_squared_log_scale = " + model.rSquared.toFixed(3),
            ].join("\n");
        }

        output.textContent = [
            "Daily occupancy: " + occupancy.join(", "),
            "Daily net changes: " + netChanges.join(", "),
            "Largest net change: " + largestNetChange + " on Day(s) " + largestDays.join(", "),
            "Latest largest net change day: Day " + latestLargestDay,
            "Peak passed: " + (peakPassed ? "Yes" : "No"),
            "Admission growth ratios: " + ratios.map(value => Number.isNaN(value) ? "NaN" : value.toFixed(3)).join(", "),
            "Mean admission growth ratio: " + (Number.isNaN(meanRatio) ? "None" : meanRatio.toFixed(3)),
            "Increasing days: " + increasingDays,
            modelText,
        ].join("\n");
    } catch (error) {
        output.textContent = "Input error: " + error.message;
    }
}

renderReport();
