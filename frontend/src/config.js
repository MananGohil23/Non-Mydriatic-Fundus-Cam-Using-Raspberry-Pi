const rawBase = import.meta.env.VITE_API_BASE || "http://localhost:8001";

export const API_BASE = rawBase.replace(/\/+$/, "");

export const HEALTH_POLL_MS = 5000;
export const VIEWFINDER_RETRY_MS = 4000;
