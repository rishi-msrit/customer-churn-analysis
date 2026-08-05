// In-browser logistic regression predictor.
// Loads the exported model coefficients and runs inference client-side.
// The model was trained with StandardScaler, so we re-apply the same
// (mean, std) values that were exported from the Python script.

let model = null;

async function initPredictor() {
  const resp = await fetch("/data/model_coefficients.json");
  model = await resp.json();
  updatePrediction();
}

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

// Build the feature value map from the form state
function readFormValues() {
  const contract = document.getElementById("contract").value;
  const internet = document.getElementById("internet").value;
  const tenure   = parseFloat(document.getElementById("tenureSlider").value) || 0;
  const charges  = parseFloat(document.getElementById("chargesSlider").value) || 0;

  return {
    tenure:                        tenure,
    MonthlyCharges:                charges,
    TotalCharges:                  tenure * charges,  // derived, same as training
    SeniorCitizen:                 document.getElementById("senior").checked ? 1 : 0,
    "Contract_One year":           contract === "one_year"  ? 1 : 0,
    "Contract_Two year":           contract === "two_year"  ? 1 : 0,
    "InternetService_Fiber optic": internet === "fiber"     ? 1 : 0,
    "InternetService_No":          internet === "none"      ? 1 : 0,
    TechSupport_Yes:               document.getElementById("techSupport").checked  ? 1 : 0,
    OnlineSecurity_Yes:            document.getElementById("security").checked      ? 1 : 0,
    PaperlessBilling_Yes:          document.getElementById("paperless").checked     ? 1 : 0,
    MultipleLines_Yes:             document.getElementById("multiLine").checked     ? 1 : 0,
    StreamingTV_Yes:               document.getElementById("streamTV").checked      ? 1 : 0,
    StreamingMovies_Yes:           document.getElementById("streamMovie").checked   ? 1 : 0,
  };
}

const FEAT_LABELS = {
  tenure:                        "Tenure",
  MonthlyCharges:                "Monthly Charges",
  TotalCharges:                  "Total Charges",
  SeniorCitizen:                 "Senior Citizen",
  "Contract_One year":           "1-Year Contract",
  "Contract_Two year":           "2-Year Contract",
  "InternetService_Fiber optic": "Fiber Internet",
  "InternetService_No":          "No Internet",
  TechSupport_Yes:               "Tech Support",
  OnlineSecurity_Yes:            "Online Security",
  PaperlessBilling_Yes:          "Paperless Billing",
  MultipleLines_Yes:             "Multiple Lines",
  StreamingTV_Yes:               "Streaming TV",
  StreamingMovies_Yes:           "Streaming Movies",
};

function updatePrediction() {
  if (!model) return;

  const vals = readFormValues();
  let z = model.intercept;
  const contribs = [];

  model.features.forEach((feat, i) => {
    const raw     = vals[feat] ?? 0;
    const scaled  = (raw - model.scaler_mean[i]) / model.scaler_std[i];
    const contrib = scaled * model.coefficients[i];
    z += contrib;
    contribs.push({ feat, contrib });
  });

  const prob = sigmoid(z);
  const pct  = Math.round(prob * 100);

  // Update probability display
  const numEl   = document.getElementById("probNum");
  const barEl   = document.getElementById("probFill");
  const badgeEl = document.getElementById("riskBadge");

  numEl.textContent  = pct + "%";
  barEl.style.width  = pct + "%";

  let colour, label;
  if (pct < 30)       { colour = "#3d9970"; label = "Low Risk"; }
  else if (pct < 60)  { colour = "#e8a838"; label = "Moderate Risk"; }
  else                { colour = "#d94f4f"; label = "High Risk"; }

  numEl.style.color   = colour;
  badgeEl.textContent = label;
  badgeEl.style.color = colour;
  barEl.style.background = colour;

  // Top 5 drivers
  contribs.sort((a, b) => Math.abs(b.contrib) - Math.abs(a.contrib));
  const top = contribs.slice(0, 5);
  const maxAbs = Math.max(...top.map(c => Math.abs(c.contrib)), 0.01);

  document.getElementById("driverRows").innerHTML = top.map(c => {
    const pxWidth = Math.round((Math.abs(c.contrib) / maxAbs) * 70);
    const col     = c.contrib > 0 ? "#d94f4f" : "#3d9970";
    const dir     = c.contrib > 0 ? "↑ risk"  : "↓ risk";
    return `
      <div class="driver-row">
        <span class="driver-name">${FEAT_LABELS[c.feat] || c.feat}</span>
        <div class="driver-bar" style="width:${pxWidth}px;background:${col}"></div>
        <span class="driver-dir" style="color:${col}">${dir}</span>
      </div>`;
  }).join("");
}

document.addEventListener("DOMContentLoaded", () => {
  initPredictor();

  // Sliders
  const tenureSlider  = document.getElementById("tenureSlider");
  const chargesSlider = document.getElementById("chargesSlider");
  const tenureVal     = document.getElementById("tenureVal");
  const chargesVal    = document.getElementById("chargesVal");

  tenureSlider.addEventListener("input", () => {
    tenureVal.textContent = tenureSlider.value + " mo";
    updatePrediction();
  });

  chargesSlider.addEventListener("input", () => {
    chargesVal.textContent = "$" + chargesSlider.value;
    updatePrediction();
  });

  // Dropdowns
  ["contract", "internet"].forEach(id => {
    document.getElementById(id).addEventListener("change", updatePrediction);
  });

  // Checkboxes
  ["senior", "techSupport", "security", "paperless", "multiLine", "streamTV", "streamMovie"]
    .forEach(id => document.getElementById(id).addEventListener("change", updatePrediction));
});
