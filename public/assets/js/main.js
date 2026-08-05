const THEME_KEY = "ca-theme";

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const btn = document.getElementById("themeBtn");
  if (btn) btn.textContent = theme === "dark" ? "☀" : "☾";
  localStorage.setItem(THEME_KEY, theme);

  // Re-tint any already-rendered Plotly charts
  const textColor = theme === "dark" ? "#c0bdb8" : "#666";
  const grid      = theme === "dark" ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.10)";
  document.querySelectorAll("[data-plotly]").forEach(el => {
    if (el._fullLayout) {
      Plotly.relayout(el, {
        "font.color": textColor,
        "xaxis.gridcolor": grid,
        "yaxis.gridcolor": grid,
        "legend.font.color": textColor,
      });
    }
  });
}

function toggleTheme() {
  const cur = document.documentElement.dataset.theme || "light";
  applyTheme(cur === "dark" ? "light" : "dark");
}

// Mark the active nav link based on current page filename
function markActiveLink() {
  const page = location.pathname.split("/").pop() || "index.html";
  document.querySelectorAll(".nav-links a").forEach(a => {
    const href = a.getAttribute("href").split("/").pop();
    const match = href === page || (page === "" && href === "index.html");
    a.classList.toggle("active", match);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  applyTheme(localStorage.getItem(THEME_KEY) || "light");
  markActiveLink();
  const btn = document.getElementById("themeBtn");
  if (btn) btn.addEventListener("click", toggleTheme);
});
