const form = document.getElementById("launch-form");
const modelOptions = document.getElementById("model-options");
const quotaSelect = document.getElementById("quota-select");
const scenarioSelect = document.getElementById("scenario-select");
const modelInput = document.getElementById("model-input");
const launchStatus = document.getElementById("launch-status");
const sessionLabel = document.getElementById("session-label");
const timelineFeed = document.getElementById("timeline-feed");

let activeSource = null;
let catalog = null;

function formatNumber(value) {
  if (value === null || value === undefined || value === "") return "-";
  return Number(value).toLocaleString();
}

function formatSeconds(value) {
  if (value === null || value === undefined) return "-";
  if (value < 60) return `${Math.round(value)}s`;
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60);
  if (minutes < 60) return `${minutes}m ${seconds}s`;
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  return `${hours}h ${mins}m`;
}

function renderCatalog(data) {
  catalog = data;

  modelOptions.innerHTML = "";
  data.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.model_name;
    modelOptions.appendChild(option);
  });
  if (!modelInput.value && data.models[0]) {
    modelInput.value = data.models[0].model_name;
  }

  quotaSelect.innerHTML = data.quota_presets
    .map((preset) => `<option value="${preset.slug}">${preset.title}</option>`)
    .join("");

  scenarioSelect.innerHTML = data.scenarios
    .map((scenario) => `<option value="${scenario.slug}">${scenario.title}</option>`)
    .join("");

  renderSelectedQuota();
  renderSelectedScenario();
}

function renderSelectedQuota() {
  if (!catalog) return;
  const quota = catalog.quota_presets.find((item) => item.slug === quotaSelect.value);
  const quotaRoot = document.getElementById("quota-details");
  if (!quota) {
    quotaRoot.innerHTML = "";
    return;
  }
  quotaRoot.innerHTML = quota.instances.map((instance) => `
    <div class="quota-item">
      <strong>${instance.gpu_type} · ${instance.instance_type}</strong>
      <div class="quota-meta">${instance.gpus_per_instance} GPUs · ${instance.gpu_memory_gb} GB VRAM · $${instance.cost_per_instance_hour_usd}/hr</div>
    </div>
  `).join("");
}

function renderSelectedScenario() {
  if (!catalog) return;
  const scenario = catalog.scenarios.find((item) => item.slug === scenarioSelect.value);
  const root = document.getElementById("scenario-details");
  if (!scenario) {
    root.textContent = "";
    return;
  }
  const events = scenario.events.length
    ? scenario.events.map((event) => `${event.label} @ ${event.at_seconds}s`).join(" · ")
    : "No timed events";
  root.innerHTML = `
    <p>${scenario.description}</p>
    <p class="quota-meta">Initial replicas: ${scenario.initial_replicas} · Launch multiplier: ${scenario.launch_timing_multiplier}x</p>
    <p class="quota-meta">${events}</p>
  `;
}

function renderSession(snapshot) {
  const runtime = snapshot.runtime;
  sessionLabel.textContent = `${snapshot.session_id} · ${snapshot.model.model_name}`;

  document.getElementById("runtime-status").textContent = runtime.status;
  document.getElementById("launch-phase").textContent = runtime.launch_phase;
  document.getElementById("aggregate-tps").textContent = formatNumber(runtime.aggregate_tps);
  document.getElementById("eta-seconds").textContent = formatSeconds(runtime.eta_seconds);
  document.getElementById("active-replicas").textContent = formatNumber(runtime.active_replicas);
  document.getElementById("progress-percent").textContent = `${runtime.progress_pct}%`;
  document.getElementById("token-progress").textContent = `${formatNumber(runtime.tokens_completed)} / ${formatNumber(runtime.tokens_total)} tokens`;
  document.getElementById("progress-fill").style.width = `${runtime.progress_pct}%`;

  const modelDetails = document.getElementById("model-details");
  const modelRows = [
    ["Resolution", snapshot.model.source],
    ["Params", `${snapshot.model.num_params_billions}B`],
    ["Active Params", `${snapshot.model.active_params_billions}B`],
    ["Model Size", `${snapshot.model.model_size_gb.toFixed(1)} GB`],
    ["Family", snapshot.model.architecture_family],
    ["MoE", snapshot.model.is_moe ? "yes" : "no"],
  ];
  modelDetails.innerHTML = modelRows.map(([key, value]) => `
    <div><dt>${key}</dt><dd>${value}</dd></div>
  `).join("");

  const koiRoot = document.getElementById("koi-details");
  if (snapshot.koi && snapshot.koi.decision) {
    const decision = snapshot.koi.decision;
    const cfg = decision.config || {};
    koiRoot.innerHTML = `
      <p><strong>Decision:</strong> ${cfg.gpu_type || "-"} · TP ${cfg.tp || "-"} · PP ${cfg.pp || "-"}</p>
      <p class="quota-meta">Predicted TPS: ${formatNumber(decision.predicted_tps)} · Confidence: ${decision.confidence || "-"}</p>
      <p class="quota-meta">Decision ID: ${decision._decision_id || "-"}</p>
    `;
  } else if (snapshot.koi && snapshot.koi.error) {
    koiRoot.innerHTML = `<p>Koi unavailable: ${snapshot.koi.error}</p>`;
  } else {
    koiRoot.innerHTML = "<p>Live Koi decision not attached. Using demo runtime defaults.</p>";
  }

  const timingRoot = document.getElementById("launch-timing-grid");
  const timing = snapshot.launch_preview.launch_timing_s;
  timingRoot.innerHTML = Object.entries(timing).map(([phase, seconds]) => `
    <div class="timing-chip">
      <span class="metric-label">${phase.replaceAll("_", " ")}</span>
      <strong>${formatSeconds(seconds)}</strong>
    </div>
  `).join("");

  renderTimeline(runtime.events, runtime.status, runtime.launch_phase, runtime.elapsed_seconds);
}

function renderTimeline(events, status, launchPhase, elapsedSeconds) {
  const derived = [
    {
      event_id: "runtime-state",
      label: `Status: ${status}`,
      description: `Current launch phase is ${launchPhase}.`,
      at_seconds: elapsedSeconds,
    },
    ...events,
  ];

  timelineFeed.innerHTML = derived.map((event) => `
    <article class="timeline-event">
      <h4>${event.label}</h4>
      <p>${event.description}</p>
      <span class="timeline-meta">${formatSeconds(event.at_seconds || 0)}</span>
    </article>
  `).join("");
}

async function loadCatalog() {
  const response = await fetch("/demo/catalog");
  renderCatalog(await response.json());
}

function startStream(sessionId) {
  if (activeSource) {
    activeSource.close();
  }

  activeSource = new EventSource(`/demo/stream/${sessionId}`);
  activeSource.onmessage = (event) => {
    const snapshot = JSON.parse(event.data);
    renderSession(snapshot);
  };
  activeSource.onerror = () => {
    launchStatus.textContent = "Stream disconnected";
  };
}

async function launchSession(payload) {
  launchStatus.textContent = "Launching...";
  const response = await fetch("/demo/launch", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Launch failed");
  }
  const session = await response.json();
  renderSession(session);
  startStream(session.session_id);
  launchStatus.textContent = "Streaming";
}

quotaSelect.addEventListener("change", renderSelectedQuota);
scenarioSelect.addEventListener("change", renderSelectedScenario);

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = {
    model_name: modelInput.value,
    quota_preset: quotaSelect.value,
    scenario: scenarioSelect.value,
    avg_input_tokens: Number(document.getElementById("input-tokens").value),
    avg_output_tokens: Number(document.getElementById("output-tokens").value),
    total_chunks: Number(document.getElementById("total-chunks").value),
    slo_deadline_hours: Number(document.getElementById("slo-hours").value),
    cost_cap_usd: Number(document.getElementById("cost-cap").value),
  };

  try {
    await launchSession(payload);
  } catch (error) {
    launchStatus.textContent = error.message;
  }
});

loadCatalog().catch((error) => {
  launchStatus.textContent = error.message;
});
