/* =========================================
TERRAPLUS — DASHBOARD INTERACTIONS
========================================= */

"use strict";

document.addEventListener("DOMContentLoaded", () => {
initializeRiskChart();
initializeFormValidation();
initializeSmoothNavigation();
});

/* -----------------------------------------

1. INTERACTIVE RISK CHART
   ----------------------------------------- */

function initializeRiskChart() {
const canvas = document.getElementById("riskChart");

if (!canvas || typeof Chart === "undefined") {
    return;
}

let values;

try {
    values = JSON.parse(canvas.dataset.values);
} catch (error) {
    console.error("Unable to read environmental risk data:", error);
    return;
}

if (
    !Array.isArray(values) ||
    values.length !== 5 ||
    !values.every(
        value =>
            typeof value === "number" &&
            Number.isFinite(value) &&
            value >= 0 &&
            value <= 100
    )
) {
    console.error("The environmental chart data is invalid.");
    return;
}

const existingChart = Chart.getChart(canvas);

if (existingChart) {
    existingChart.destroy();
}

const chartColors = [
    "#D5FF54",
    "#5CF4ED",
    "#FF796C",
    "#FFD85E",
    "#B7A0FF"
];

const chart = new Chart(canvas, {
    type: "bar",

    data: {
        labels: [
            "Air Quality",
            "Flood",
            "Forest Fire",
            "Heatwave",
            "Drought"
        ],

        datasets: [{
            label: "Illustrative risk score",
            data: values,
            backgroundColor: chartColors,
            hoverBackgroundColor: [
                "#E8FF9C",
                "#9CFFF8",
                "#FFAB9F",
                "#FFEB9C",
                "#D6C8FF"
            ],
            borderColor: chartColors,
            borderWidth: 1,
            borderRadius: 9,
            borderSkipped: false,
            maxBarThickness: 48
        }]
    },

    options: {
        responsive: true,
        maintainAspectRatio: false,

        animation: {
            duration: 850,
            easing: "easeOutQuart"
        },

        interaction: {
            intersect: false,
            mode: "index"
        },

        plugins: {
            legend: {
                display: false
            },

            tooltip: {
                backgroundColor: "#17152F",
                titleColor: "#D5FF54",
                bodyColor: "#FFFDF7",
                borderColor: "rgba(255,255,255,0.15)",
                borderWidth: 1,
                padding: 13,
                cornerRadius: 10,

                callbacks: {
                    label(context) {
                        return `Illustrative score: ${context.parsed.y}/100`;
                    }
                }
            }
        },

        scales: {
            x: {
                grid: {
                    display: false
                },

                border: {
                    display: false
                },

                ticks: {
                    color: "#C5C0E0",
                    font: {
                        family: "DM Sans",
                        size: 11
                    },
                    maxRotation: 35,
                    minRotation: 0
                }
            },

            y: {
                beginAtZero: true,
                min: 0,
                max: 100,

                ticks: {
                    stepSize: 20,
                    color: "#A8A2C9",
                    padding: 10
                },

                grid: {
                    color: "rgba(255,255,255,0.08)",
                    drawTicks: false
                },

                border: {
                    display: false
                }
            }
        }
    }
});

canvas.dataset.chartInitialized = "true";

return chart;

}

/* -----------------------------------------
2. ENVIRONMENTAL INPUT VALIDATION
AND PREDICTION FORM SUBMISSION
----------------------------------------- */

function initializeFormValidation() {
const form = document.querySelector(".prediction-form");

if (!form) {
    console.error("Prediction form .prediction-form was not found.");
    return;
}

const defaultValues = {
    temperature: "29",
    humidity: "65",
    rainfall: "20",
    wind_speed: "15",
    pressure: "1012",
    pm25: "40",
    pm10: "65",
    water_level: "3",
    soil_moisture: "45"
};

const fieldLabels = {
    temperature: "Temperature",
    humidity: "Humidity",
    rainfall: "Rainfall",
    wind_speed: "Wind speed",
    pressure: "Atmospheric pressure",
    pm25: "PM2.5",
    pm10: "PM10",
    water_level: "Water level",
    soil_moisture: "Soil moisture"
};

const getInput = name =>
    form.querySelector(`[name="${name}"]`) ||
    form.querySelector(`#${name}`);

// Add demonstration defaults to fields that already exist.
Object.entries(defaultValues).forEach(([name, value]) => {
    const input = getInput(name);

    if (input && input.value.trim() === "") {
        input.value = value;
    }
});

form.querySelectorAll("input").forEach(input => {
    input.addEventListener("input", () => {
        input.setAttribute("aria-invalid", "false");
        input.setCustomValidity("");
    });
});

form.addEventListener("submit", async event => {
    event.preventDefault();

    let firstInvalidInput = null;

    // Validate all existing required numeric inputs.
    for (const input of form.querySelectorAll("input[required]")) {
        const value = input.value.trim();
        const number = Number(value);

        let valid =
            value !== "" &&
            Number.isFinite(number);

        if (input.type === "number" && value !== "") {
            if (input.min !== "" && number < Number(input.min)) {
                valid = false;
            }

            if (input.max !== "" && number > Number(input.max)) {
                valid = false;
            }
        }

        input.setAttribute("aria-invalid", String(!valid));

        if (!valid && !firstInvalidInput) {
            firstInvalidInput = input;
        }
    }

    if (firstInvalidInput) {
        firstInvalidInput.focus();
        firstInvalidInput.reportValidity();
        return;
    }

    // Collect the nine inputs expected by the trained risk model.
    const body = new FormData();

    for (const [name, defaultValue] of Object.entries(defaultValues)) {
        const input = getInput(name);

        if (input) {
            const value = input.value.trim();

            if (value === "" || !Number.isFinite(Number(value))) {
                alert(`Please enter a valid value for ${fieldLabels[name]}.`);
                input?.focus();
                return;
            }

            body.set(name, value);
        } else {
            // app.py provides matching defaults for these missing fields.
            body.set(name, defaultValue);
        }
    }

    const button = form.querySelector(
        'button[type="submit"], input[type="submit"]'
    );

    const originalText = button
        ? (button.tagName === "INPUT" ? button.value : button.textContent)
        : "";

    if (button) {
        button.disabled = true;

        if (button.tagName === "INPUT") {
            button.value = "Analyzing...";
        } else {
            button.textContent = "Analyzing...";
        }
    }

    try {
        // Flask defines @app.route("/predict", methods=["POST"]).
        const response = await fetch("/predict", {
            method: "POST",
            body: body
        });

        const contentType =
            response.headers.get("content-type") || "";

        if (!response.ok) {
            const errorDetails = await response.text();

            console.error(
                "Prediction request failed:",
                response.status,
                errorDetails
            );

            alert(
                `Prediction failed (HTTP ${response.status}). ` +
                "Please check the VS Code terminal."
            );

            return;
        }

        if (contentType.includes("application/json")) {
            const result = await response.json();
            displayPredictionResults(result);
            return;
        }

        // app.py returns render_template("index.html", ...).
        // Display that returned page, including prediction results.
        const html = await response.text();

        document.open();
        document.write(html);
        document.close();

    } catch (error) {
        console.error("Prediction request error:", error);

        alert(
            "Could not connect to Flask. Make sure the website server is running."
        );

    } finally {
        if (button && document.contains(button)) {
            button.disabled = false;

            if (button.tagName === "INPUT") {
                button.value = originalText || "Start Prediction";
            } else {
                button.textContent = originalText || "Start Prediction";
            }
        }
    }
});

}

/* -----------------------------------------
3. OPTIONAL JSON RESULT DISPLAY
----------------------------------------- */

function displayPredictionResults(data) {
let results =
document.getElementById("prediction-results") ||
document.getElementById("results");

if (!results) {
    results = document.createElement("section");
    results.id = "prediction-results";
    results.style.cssText =
        "margin:24px 0;padding:20px;border-radius:14px;" +
        "background:#17152F;color:#FFFDF7;";

    document.body.appendChild(results);
}

results.replaceChildren();

const heading = document.createElement("h2");
heading.textContent = "Environmental Risk Results";
results.appendChild(heading);

const labels = {
    air_quality: "Air quality",
    flood_risk: "Flood risk",
    fire_risk: "Forest fire risk",
    heatwave_risk: "Heatwave risk",
    drought_risk: "Drought risk",
    overall_risk: "Overall risk",
    overall_score: "Overall score",
    predicted_temperature: "Predicted temperature",
    predicted_rainfall: "Predicted rainfall"
};

Object.entries(labels).forEach(([key, label]) => {
    if (data[key] === undefined) {
        return;
    }

    const paragraph = document.createElement("p");
    const strong = document.createElement("strong");

    strong.textContent = `${label}: `;
    paragraph.appendChild(strong);
    paragraph.appendChild(
        document.createTextNode(String(data[key]))
    );

    results.appendChild(paragraph);
});

if (Array.isArray(data.recommendations)) {
    const subheading = document.createElement("h3");
    subheading.textContent = "Recommendations";
    results.appendChild(subheading);

    const list = document.createElement("ul");

    data.recommendations.forEach(recommendation => {
        const item = document.createElement("li");
        item.textContent = String(recommendation);
        list.appendChild(item);
    });

    results.appendChild(list);
}

results.scrollIntoView({
    behavior: "smooth",
    block: "start"
});

}

/* -----------------------------------------
4. SMOOTH NAVIGATION
----------------------------------------- */

function initializeSmoothNavigation() {
const links = document.querySelectorAll('a[href^="#"]');

links.forEach(link => {
    link.addEventListener("click", event => {
        const targetId = link.getAttribute("href");

        if (!targetId || targetId === "#") {
            return;
        }

        let target;

        try {
            target = document.querySelector(targetId);
        } catch (error) {
            return;
        }

        if (!target) {
            return;
        }

        event.preventDefault();

        const reduceMotion = window.matchMedia(
            "(prefers-reduced-motion: reduce)"
        ).matches;

        target.scrollIntoView({
            behavior: reduceMotion ? "auto" : "smooth",
            block: "start"
        });

        history.replaceState(null, "", targetId);
    });
});

}

// =====================================
// LIVE WEATHER API INTEGRATION
// =====================================

const weatherButton = document.getElementById("fetch-weather");

if (weatherButton) {
    weatherButton.addEventListener("click", async () => {
        const cityInput = document.getElementById("weather-city");
        const message = document.getElementById("weather-message");
        const status = document.getElementById("weather-status");

        const city = cityInput.value.trim();

        if (!city) {
            message.textContent = "Please enter a city name.";
            return;
        }

        weatherButton.disabled = true;
        message.textContent = "Fetching live weather...";
        
        if (status) {
            status.textContent = "Loading...";
        }

        try {
            const response = await fetch(
                `/api/weather?city=${encodeURIComponent(city)}`
            );

            const data = await response.json();

            if (!response.ok) {
                throw new Error(
                    data.error || "Unable to fetch weather."
                );
            }

            document.getElementById("weather-temperature").textContent =
                `${data.temperature ?? "--"} °C`;

            document.getElementById("weather-humidity").textContent =
                `${data.humidity ?? "--"} %`;

            document.getElementById("weather-rainfall").textContent =
                `${data.rainfall ?? "--"} mm`;

            document.getElementById("weather-wind").textContent =
                `${data.wind_speed ?? "--"} km/h`;

            document.getElementById("weather-pressure").textContent =
                `${data.pressure ?? "--"} hPa`;

            document.getElementById("weather-pm25").textContent =
                data.pm25 == null
                    ? "Unavailable"
                    : `${data.pm25} µg/m³`;

            document.getElementById("weather-condition").textContent =
                data.condition ?? "Unavailable";

            message.textContent =
                `${data.city}, ${data.country} | Updated ${data.updated_at}`;

            if (status) {
                status.textContent = "Weather loaded successfully";
            }

        } catch (error) {
            message.textContent = error.message;

            if (status) {
                status.textContent = "Weather unavailable";
            }

        } finally {
            weatherButton.disabled = false;
        }
    });
}