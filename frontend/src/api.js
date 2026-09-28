import { API_BASE } from "./config";

async function requestJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    const detail = payload && payload.detail ? payload.detail : `HTTP ${response.status}`;
    throw new Error(detail);
  }
  return payload;
}

export function getHealth() {
  return requestJson("/health");
}

export function capture() {
  return requestJson("/capture", { method: "POST" });
}

export function analyzeCapture(captureId, force = false) {
  const body = new FormData();
  body.append("capture_id", captureId);
  return requestJson(`/analyze?force=${force ? "true" : "false"}`, { method: "POST", body });
}

export function analyzeFile(file, force = false) {
  const body = new FormData();
  body.append("file", file);
  return requestJson(`/analyze?force=${force ? "true" : "false"}`, { method: "POST", body });
}

export function listCaptures(limit = 20) {
  return requestJson(`/api/captures?limit=${limit}`);
}

export function getLatestTap() {
  return requestJson("/latest_tap");
}

export function viewfinderUrl() {
  return `${API_BASE}/viewfinder`;
}

export function snapshotUrl() {
  return `${API_BASE}/snapshot?t=${Date.now()}`;
}

export function imageUrl(captureId) {
  return `${API_BASE}/images/${captureId}`;
}

export function heatmapUrl(captureId) {
  return `${API_BASE}/heatmaps/${captureId}`;
}
