// Fetch a Plotly figure JSON and render it into divId.
// Transparently handles theme colours at render time.
async function loadChart(divId, filename, extraLayout = {}) {
  const div = document.getElementById(divId);
  if (!div) return;

  try {
    const resp = await fetch("/data/" + filename);
    if (!resp.ok) throw new Error(resp.status);
    const fig = await resp.json();

    const isDark   = document.documentElement.dataset.theme === "dark";
    const textCol  = isDark ? "#c0bdb8" : "#666";
    const gridCol  = isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.10)";

    const layout = Object.assign({}, fig.layout, {
      font:     Object.assign({ color: textCol }, fig.layout.font || {}),
      autosize: true,
    });

    if (layout.xaxis) layout.xaxis.gridcolor = gridCol;
    if (layout.yaxis) layout.yaxis.gridcolor = gridCol;
    Object.assign(layout, extraLayout);

    div.dataset.plotly = "1";
    Plotly.react(divId, fig.data, layout, { responsive: true, displayModeBar: false });
  } catch (err) {
    div.innerHTML =
      '<p style="padding:20px;color:#aaa;font-size:0.83rem;">Chart data not available — run <code>python analysis/run_analysis.py</code> first.</p>';
  }
}
