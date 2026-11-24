"use strict";

const byId = (id) => document.getElementById(id);
const ui = {
  run: byId("run-label"),
  worm: byId("worm"),
  origin: byId("origin"),
  previous: byId("previous"),
  next: byId("next"),
  originLabel: byId("origin-label"),
  search: byId("search"),
  linkedOnly: byId("linked-only"),
  neurons: byId("neurons"),
  restore: byId("restore"),
  count: byId("selection-count"),
  groupName: byId("group-name"),
  saveGroup: byId("save-group"),
  save: byId("save"),
  groups: byId("groups"),
  apply: byId("apply"),
  delete: byId("delete"),
  groupStatus: byId("group-status"),
  status: byId("forecast-status"),
  table: byId("forecast-table"),
  probabilities: byId("probabilities"),
  model: byId("model"),
  details: byId("forecast-details"),
  canvas: byId("trajectory"),
  view3d: byId("trajectory-3d"),
  view2d: byId("trajectory-2d"),
  resetView: byId("trajectory-reset"),
  trajectoryHint: byId("trajectory-hint"),
  observedMotif: byId("observed-motif"),
  legend: byId("motif-legend"),
};
const defaultCamera = { yaw: 0.72, pitch: 0.50 };
const state = {
  run: null,
  worm: null,
  originIndex: 0,
  off: new Set(),
  groups: [],
  wormRequest: null,
  forecastRequest: null,
  forecastTimer: null,
  plottedPoints: [],
  trajectoryView: "3d",
  camera: { ...defaultCamera },
  rotation: null,
  rotated: false,
};
const palette = ["#187566", "#81488c", "#947025", "#3f628f", "#a55044", "#497137", "#786050", "#a24a72"];

function element(tag, value, className) {
  const node = document.createElement(tag);
  if (value !== undefined) node.textContent = value;
  if (className) node.className = className;
  return node;
}

function setStatus(message, failed = false) {
  ui.status.textContent = message;
  ui.status.classList.toggle("error", failed);
}

async function getJson(path, params = {}, signal) {
  const query = new URLSearchParams(params).toString();
  const response = await fetch(query ? `${path}?${query}` : path, { signal });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.error || `Request failed (${response.status})`);
  }
  return response.json();
}

function origin() {
  return state.worm?.origins[state.originIndex];
}

function cancelForecast() {
  clearTimeout(state.forecastTimer);
  state.forecastRequest?.abort();
  state.forecastRequest = null;
}

function clearForecast(message) {
  ui.model.hidden = true;
  ui.table.hidden = true;
  ui.table.setAttribute("aria-busy", "true");
  setStatus(message);
}

function queueForecast() {
  cancelForecast();
  if (!origin()) return;
  clearForecast("Calculating paired forecasts…");
  state.forecastTimer = setTimeout(renderForecast, 100);
}

function probabilityCell(motif, label, probability, style) {
  const cell = element("td");
  const container = element("div", undefined, "probability-entry");
  const bar = element("progress", undefined, style);
  bar.max = 1;
  bar.value = probability;
  bar.setAttribute("aria-label", `${motif}, ${label}`);
  container.append(bar, element("span", `${(probability * 100).toFixed(1)}%`));
  cell.append(container);
  return cell;
}

function differenceCell(delta) {
  const value = Math.abs(delta) < 0.005 ? 0 : delta;
  const sign = value < 0 ? "−" : value > 0 ? "+" : "";
  const cls = value < 0 ? " negative" : value > 0 ? " positive" : "";
  return element("td", `${sign}${Math.abs(value).toFixed(2)} pp`, `change${cls}`);
}

async function renderForecast() {
  const current = origin();
  if (!current) return;
  const controller = new AbortController();
  state.forecastRequest = controller;
  try {
    const result = await getJson("/api/intervene", {
      worm: state.worm.id,
      origin: current.sample,
      off: [...state.off].sort((a, b) => a - b).join(","),
    }, controller.signal);
    if (state.forecastRequest !== controller) return;
    const rows = result.motifs.map((motif, i) => {
      const row = element("tr");
      const name = element("th", motif);
      name.scope = "row";
      row.append(
        name,
        probabilityCell(motif, "all on", result.factual[i], "factual"),
        probabilityCell(motif, "disconnected", result.disconnected[i], "disconnected"),
        differenceCell(result.delta_pp[i]),
      );
      return row;
    });
    ui.probabilities.replaceChildren(...rows);
    ui.model.textContent = result.model === "local" ? "Trailing local fit" : "Pooled fit";
    ui.model.hidden = false;
    const links = result.removed_edges === 1 ? "edge" : "edges";
    setStatus(result.off.length
      ? `${result.removed_edges} outgoing ${links} removed in the joint simulation`
      : "All outgoing connections are on");
    ui.details.textContent = `${result.draws} paired draws · Horizon ${result.horizon_seconds.toFixed(2)} s · Terminal window ${result.target_seconds.toFixed(2)} s`;
    ui.table.hidden = false;
    ui.table.setAttribute("aria-busy", "false");
  } catch (error) {
    if (state.forecastRequest === controller && !controller.signal.aborted) {
      setStatus(error.message, true);
      ui.table.setAttribute("aria-busy", "false");
    }
  }
}

function updateOrigin(index) {
  if (index < 0 || index >= (state.worm?.origins.length ?? 0)) return;
  state.originIndex = index;
  const current = origin();
  const lastIndex = state.worm.origins.length - 1;
  ui.origin.value = index;
  ui.previous.disabled = index === 0;
  ui.next.disabled = index === lastIndex;
  ui.originLabel.textContent = `${current.seconds.toFixed(1)} s · sample ${current.sample} · ${index + 1} of ${lastIndex + 1}`;
  ui.origin.setAttribute("aria-valuetext", ui.originLabel.textContent);
  ui.observedMotif.textContent = `Recorded motif ${state.worm.motifs[current.observed_motif]}`;
  drawTrajectory();
  queueForecast();
}

function neuronItem(neuron) {
  const item = element("li", undefined, "neuron-item");
  const label = element("label");
  const name = neuron.name || neuron.raw_name || `ROI ${neuron.id}`;
  const aliases = neuron.raw_name && neuron.raw_name !== name ? ` · ${neuron.raw_name}` : "";
  const edges = `${neuron.outgoing_edges} outgoing ${neuron.outgoing_edges === 1 ? "edge" : "edges"}`;
  const metadata = `ROI ${neuron.id}${aliases} · ${edges}`;
  const toggle = element("input");
  toggle.type = "checkbox";
  toggle.setAttribute("role", "switch");
  toggle.setAttribute("aria-label", `Outgoing connections for ${name}, ROI ${neuron.id}`);
  toggle.checked = !state.off.has(neuron.id);
  const indicator = element("span", toggle.checked ? "On" : "Off");
  indicator.setAttribute("aria-hidden", "true");
  const control = element("span", undefined, "neuron-switch");
  control.append(toggle, indicator);
  toggle.addEventListener("change", () => {
    if (toggle.checked) state.off.delete(neuron.id);
    else state.off.add(neuron.id);
    indicator.textContent = toggle.checked ? "On" : "Off";
    updateSelectionCount();
    queueForecast();
  });
  label.append(element("span", name, "neuron-name"), element("span", metadata, "neuron-info"), control);
  item.append(label);
  return item;
}

function renderNeurons() {
  const query = ui.search.value.trim().toLowerCase();
  const matches = state.worm.neurons.filter((neuron) => (
    (!ui.linkedOnly.checked || neuron.outgoing_edges > 0)
    && [neuron.id, neuron.name, neuron.raw_name].some((value) => String(value).toLowerCase().includes(query))
  ));
  ui.neurons.replaceChildren(...(matches.length
    ? matches.map(neuronItem)
    : [element("li", "No matching neurons", "empty-list")]));
}

function updateSelectionCount() {
  const count = state.off.size;
  ui.count.textContent = count ? `${count} ${count === 1 ? "neuron" : "neurons"} off` : "All neurons on";
  ui.restore.disabled = count === 0;
  ui.groupName.disabled = count === 0;
  ui.save.disabled = count === 0;
}

function setOff(ids) {
  state.off = new Set(ids);
  renderNeurons();
  updateSelectionCount();
  queueForecast();
}

function groupKey() {
  return `effectome:disconnections:${JSON.stringify([state.run.run_id, state.worm.id])}`;
}

function loadGroups() {
  try {
    const options = JSON.parse(localStorage.getItem(groupKey()) || "[]");
    const known = new Set(state.worm.neurons.map((neuron) => neuron.id));
    return options.filter((group) => (
      typeof group?.name === "string" && Array.isArray(group.ids) && group.ids.length > 0
      && group.ids.every((id) => known.has(id))
    ));
  } catch {
    return [];
  }
}

function renderGroups(selected = "") {
  const options = state.groups.map((group) => new Option(group.name, group.name));
  ui.groups.replaceChildren(...(options.length ? options : [new Option("No saved groups", "")]));
  if (options.some((option) => option.value === selected)) ui.groups.value = selected;
  ui.groups.disabled = options.length === 0;
  ui.apply.disabled = options.length === 0;
  ui.delete.disabled = options.length === 0;
}

function persistGroups(message) {
  ui.groupStatus.textContent = message;
  try {
    localStorage.setItem(groupKey(), JSON.stringify(state.groups));
  } catch {
    ui.groupStatus.textContent += " Browser storage unavailable. Groups are available for this session.";
  }
}

function saveGroup(event) {
  event.preventDefault();
  const name = ui.groupName.value.trim();
  if (!state.worm || !name || !state.off.size) return;
  const group = { name, ids: [...state.off] };
  const index = state.groups.findIndex((saved) => saved.name === name);
  if (index >= 0) state.groups[index] = group;
  else state.groups.push(group);
  ui.groupName.value = "";
  renderGroups(name);
  persistGroups(`Saved ${name}`);
}

function deleteGroup() {
  const name = ui.groups.value;
  state.groups = state.groups.filter((group) => group.name !== name);
  renderGroups();
  persistGroups(`Deleted ${name}`);
}

function renderLegend() {
  const markers = state.worm.motifs.map((name, index) => {
    const dot = element("canvas");
    dot.width = dot.height = 10;
    dot.setAttribute("aria-hidden", "true");
    const context = dot.getContext("2d");
    if (context) {
      context.fillStyle = palette[index % palette.length];
      context.beginPath();
      context.arc(5, 5, 4, 0, Math.PI * 2);
      context.fill();
    }
    const marker = element("li");
    marker.append(dot, element("span", name));
    return marker;
  });
  ui.legend.replaceChildren(...markers);
}

function plotTwoDimensions(ctx, points, width, height, xMin, xMax, yMin, yMax) {
  const left = 36, right = width - 18, top = 18, bottom = height - 30;
  const scale = Math.min((right - left) / (xMax - xMin || 1), (bottom - top) / (yMax - yMin || 1));
  const positions = points.map((point) => [
    (left + right) / 2 + (point.code[0] - (xMin + xMax) / 2) * scale,
    (top + bottom) / 2 - (point.code[1] - (yMin + yMax) / 2) * scale,
  ]);
  ctx.strokeStyle = "#bdcdc4";
  ctx.beginPath();
  ctx.moveTo(left - 7, top - 6);
  ctx.lineTo(left - 7, bottom + 3);
  ctx.lineTo(right + 6, bottom + 3);
  ctx.stroke();
  ctx.font = "11px system-ui, sans-serif";
  ctx.fillStyle = "#516560";
  ctx.textAlign = "center";
  ctx.fillText("Code 1", width / 2, height - 8);
  ctx.save();
  ctx.translate(15, height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Code 2", 0, 0);
  ctx.restore();
  return positions;
}

function plotThreeDimensions(ctx, points, width, height, xMin, xMax, yMin, yMax) {
  const { yaw, pitch } = state.camera;
  const cosine = Math.cos(yaw), sine = Math.sin(yaw);
  const tiltCos = Math.cos(pitch), tiltSin = Math.sin(pitch);
  // Rotate around the time axis, then tilt the view for an orthographic projection.
  const rotate = (x, y, time) => {
    const depth = sine * x + cosine * y;
    return [cosine * x - sine * y, tiltCos * time + tiltSin * depth, tiltCos * depth - tiltSin * time];
  };
  const sides = [-0.5, 0.5];
  const corners = sides.flatMap((x) => sides.flatMap((y) => sides.map((time) => rotate(x, y, time))));
  const halfWidth = Math.max(...corners.map(([x]) => Math.abs(x)));
  const halfHeight = Math.max(...corners.map(([, y]) => Math.abs(y)));
  const scale = Math.min((width - 106) / (2 * halfWidth), (height - 63) / (2 * halfHeight));
  const project = (...coordinates) => {
    const [x, y, depth] = rotate(...coordinates);
    return [width / 2 + 12 + x * scale, height / 2 - y * scale, depth];
  };
  const first = points[0].seconds, last = points.at(-1).seconds;
  const connector = (from, to) => {
    ctx.beginPath();
    ctx.moveTo(...project(...from).slice(0, 2));
    ctx.lineTo(...project(...to).slice(0, 2));
    ctx.stroke();
  };
  ctx.strokeStyle = "#dbe6df";
  for (const tick of [-0.5, 0, 0.5]) {
    connector([tick, -0.5, -0.5], [tick, 0.5, -0.5]);
    connector([-0.5, tick, -0.5], [0.5, tick, -0.5]);
  }
  const base = [-0.5, -0.5, -0.5];
  const xAxis = [0.5, -0.5, -0.5];
  const yAxis = [-0.5, 0.5, -0.5];
  ctx.strokeStyle = "#8ca69a";
  for (const end of [xAxis, yAxis]) connector(base, end);
  ctx.fillStyle = "#516560";
  ctx.font = "11px system-ui, sans-serif";
  ctx.textAlign = "center";
  for (const [label, axis] of [["Code 1", xAxis], ["Code 2", yAxis]]) {
    const [x, y] = project(...axis);
    ctx.fillText(label, x, y + 16);
  }
  // Place the time scale beside the trajectory to keep its labels clear as the view rotates.
  const railX = width / 2 - halfWidth * scale;
  const [, topY] = project(0, 0, 0.5);
  const [, bottomY] = project(0, 0, -0.5);
  ctx.beginPath();
  ctx.moveTo(railX, topY);
  ctx.lineTo(railX, bottomY);
  ctx.textAlign = "right";
  ctx.fillText("Time (s)", railX, topY - 14);
  for (const [y, seconds] of [[bottomY, first], [topY, last]]) {
    ctx.moveTo(railX - 4, y);
    ctx.lineTo(railX, y);
    ctx.fillText(seconds.toFixed(1), railX - 7, y + 4);
  }
  ctx.stroke();

  // Both code axes use the same scale. The time axis spans the observed elapsed seconds.
  const span = Math.max(xMax - xMin, yMax - yMin) || 1;
  const midX = (xMin + xMax) / 2, midY = (yMin + yMax) / 2;
  return points.map((point) => project(
    (point.code[0] - midX) / span,
    (point.code[1] - midY) / span,
    (point.seconds - first) / (last - first || 1) - 0.5,
  ));
}

function drawTrajectory() {
  const points = state.worm?.origins;
  const hasData = Boolean(points?.length);
  const threeDimensions = state.trajectoryView === "3d";
  state.plottedPoints = [];
  ui.canvas.classList.toggle("has-data", hasData);
  ui.canvas.classList.toggle("is-3d", threeDimensions);
  ui.resetView.disabled = !hasData || !threeDimensions;
  const { width, height } = ui.canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  ui.canvas.width = Math.round(width * ratio);
  ui.canvas.height = Math.round(height * ratio);
  const ctx = ui.canvas.getContext("2d");
  if (!ctx || !hasData || width < 120 || height < 95) return;
  ctx.scale(ratio, ratio);
  const xs = points.map((point) => point.code[0]);
  const ys = points.map((point) => point.code[1]);
  const plot = threeDimensions ? plotThreeDimensions : plotTwoDimensions;
  state.plottedPoints = plot(ctx, points, width, height, Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys));

  ctx.strokeStyle = "#b0c3b8";
  ctx.beginPath();
  state.plottedPoints.forEach(([x, y], i) => {
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  const drawOrder = state.plottedPoints.map((_, index) => index);
  if (threeDimensions) {
    drawOrder.sort((a, b) => state.plottedPoints[a][2] - state.plottedPoints[b][2]);
  }
  drawOrder.forEach((i) => {
    const [x, y] = state.plottedPoints[i];
    ctx.fillStyle = palette[points[i].observed_motif % palette.length];
    ctx.beginPath();
    ctx.arc(x, y, 2.8, 0, Math.PI * 2);
    ctx.fill();
  });
  const [x, y] = state.plottedPoints[state.originIndex];
  ctx.strokeStyle = "#182927";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(x, y, 7, 0, Math.PI * 2);
  ctx.stroke();
}

function setTrajectoryView(view) {
  state.trajectoryView = view;
  const threeDimensions = view === "3d";
  ui.view3d.setAttribute("aria-pressed", threeDimensions);
  ui.view2d.setAttribute("aria-pressed", !threeDimensions);
  ui.trajectoryHint.textContent = threeDimensions
    ? "Drag or use arrow keys to rotate. Select a point to choose an origin."
    : "Select a point to choose a forecast origin.";
  ui.canvas.setAttribute("aria-label", threeDimensions
    ? "Motif trajectory with time as the third axis. The forecast origin is circled."
    : "Two-dimensional trajectory colored by recorded motif. The forecast origin is circled.");
  drawTrajectory();
}

function rotateTrajectory(yaw, pitch) {
  state.camera = { yaw, pitch: Math.max(0.12, Math.min(1.25, pitch)) };
  drawTrajectory();
}

function startRotation(event) {
  if (!event.isPrimary || event.button !== 0) return;
  state.rotated = false;
  if (state.trajectoryView !== "3d" || !state.worm) return;
  state.rotation = { id: event.pointerId, x: event.clientX, y: event.clientY, ...state.camera };
  ui.canvas.setPointerCapture(event.pointerId);
}

function moveRotation(event) {
  const start = state.rotation;
  if (!start || start.id !== event.pointerId) return;
  const dx = event.clientX - start.x, dy = event.clientY - start.y;
  if (!state.rotated && Math.hypot(dx, dy) < 4) return;
  state.rotated = true;
  ui.canvas.classList.add("is-rotating");
  rotateTrajectory(start.yaw + 0.012 * dx, start.pitch + 0.012 * dy);
}

function stopRotation(event) {
  if (state.rotation?.id !== event.pointerId) return;
  state.rotation = null;
  ui.canvas.classList.remove("is-rotating");
}

function pickOrigin(event) {
  if (state.rotated) {
    state.rotated = false;
    return;
  }
  const rect = ui.canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  let winner = -1;
  let nearest = 16 ** 2;
  for (const [index, [cx, cy]] of state.plottedPoints.entries()) {
    const distance = (x - cx) ** 2 + (y - cy) ** 2;
    if (distance < nearest) {
      nearest = distance;
      winner = index;
    }
  }
  if (winner >= 0) updateOrigin(winner);
}

async function loadWorm(id) {
  cancelForecast();
  state.wormRequest?.abort();
  const controller = new AbortController();
  state.wormRequest = controller;
  state.worm = null;
  state.originIndex = 0;
  state.off.clear();
  state.groups = [];
  ui.origin.disabled = ui.previous.disabled = ui.next.disabled = ui.linkedOnly.disabled = true;
  ui.search.disabled = ui.restore.disabled = ui.save.disabled = ui.groupName.disabled = true;
  ui.search.value = ui.groupStatus.textContent = ui.observedMotif.textContent = ui.count.textContent = "";
  ui.originLabel.textContent = "Loading origins…";
  ui.neurons.replaceChildren();
  ui.legend.replaceChildren();
  renderGroups();
  drawTrajectory();
  clearForecast("Loading worm…");
  try {
    const worm = await getJson("/api/worm", { worm: id }, controller.signal);
    if (state.wormRequest !== controller) return;
    state.worm = worm;
    state.groups = loadGroups();
    ui.origin.disabled = ui.search.disabled = ui.linkedOnly.disabled = false;
    ui.origin.max = worm.origins.length - 1;
    renderNeurons();
    renderGroups();
    renderLegend();
    updateSelectionCount();
    updateOrigin(0);
  } catch (error) {
    if (state.wormRequest === controller && !controller.signal.aborted) {
      ui.originLabel.textContent = "Origins unavailable";
      setStatus(error.message, true);
    }
  }
}

async function initialize() {
  try {
    state.run = await getJson("/api/run");
    ui.run.textContent = `Run ${state.run.run_id}`;
    ui.worm.replaceChildren(...state.run.worms.map((worm) => new Option(worm.label, worm.id)));
    if (!state.run.worms.length) {
      setStatus("No completed worm forecasts in this run.", true);
      return;
    }
    ui.worm.disabled = false;
    await loadWorm(ui.worm.value);
  } catch (error) {
    setStatus(error.message, true);
  }
}

ui.worm.addEventListener("change", () => loadWorm(ui.worm.value));
ui.origin.addEventListener("input", () => updateOrigin(Number(ui.origin.value)));
ui.previous.addEventListener("click", () => updateOrigin(state.originIndex - 1));
ui.next.addEventListener("click", () => updateOrigin(state.originIndex + 1));
ui.search.addEventListener("input", renderNeurons);
ui.linkedOnly.addEventListener("change", renderNeurons);
ui.restore.addEventListener("click", () => setOff([]));
ui.saveGroup.addEventListener("submit", saveGroup);
ui.apply.addEventListener("click", () => {
  const group = state.groups.find((option) => option.name === ui.groups.value);
  if (group) setOff(group.ids);
});
ui.delete.addEventListener("click", deleteGroup);
ui.view3d.addEventListener("click", () => setTrajectoryView("3d"));
ui.view2d.addEventListener("click", () => setTrajectoryView("2d"));
ui.resetView.addEventListener("click", () => rotateTrajectory(defaultCamera.yaw, defaultCamera.pitch));
ui.canvas.addEventListener("pointerdown", startRotation);
ui.canvas.addEventListener("pointermove", moveRotation);
ui.canvas.addEventListener("lostpointercapture", stopRotation);
ui.canvas.addEventListener("click", pickOrigin);
ui.canvas.addEventListener("keydown", (event) => {
  if (state.trajectoryView !== "3d" || !state.worm) return;
  const direction = {
    ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1],
  }[event.key];
  if (!direction) return;
  event.preventDefault();
  rotateTrajectory(state.camera.yaw + direction[0] * 0.12, state.camera.pitch + direction[1] * 0.12);
});
if ("ResizeObserver" in window) new ResizeObserver(drawTrajectory).observe(ui.canvas);
else window.addEventListener("resize", drawTrajectory);

initialize();
