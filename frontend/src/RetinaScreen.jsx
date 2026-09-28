import React, { useState, useEffect, useRef } from "react";
import {
  BatteryFull,
  Eye,
  FileText,
  Settings,
  HelpCircle,
  ArrowLeft,
  CheckCircle2,
  Scan,
  Wifi,
  WifiOff,
  Signal,
  Upload,
  AlertTriangle,
  Activity,
  Camera,
} from "lucide-react";
import {
  getHealth,
  capture,
  analyzeCapture,
  analyzeFile,
  listCaptures,
  viewfinderUrl,
  heatmapUrl,
  imageUrl,
} from "./api";
import { HEALTH_POLL_MS, VIEWFINDER_RETRY_MS } from "./config";

const BG_LIGHT = "#F8FAFC";
const ACCENT_SUCCESS = "#10B981";
const ACCENT_DANGER = "#EF4444";
const TEXT_MAIN = "#0F172A";

/* ---------- shared chrome ---------- */

function StatusBar({ health }) {
  const [clock, setClock] = useState(() => new Date());
  useEffect(() => {
    const t = setInterval(() => setClock(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const cameraOk = Boolean(health && health.camera && health.camera.reachable);
  const engine = health && health.model ? health.model.engine : "…";
  const timeLabel = clock.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  return (
    <div
      className="flex items-center justify-between px-6 h-12 flex-shrink-0 z-50 relative"
      style={{ background: 'linear-gradient(180deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0) 100%)' }}
    >
      <div className="flex items-center gap-4">
        <span className="text-[13px] font-medium text-slate-800 tracking-wide">{timeLabel}</span>
        <div className="flex items-center gap-1.5" title={cameraOk ? "Camera online" : "Camera offline"}>
          {cameraOk ? (
            <Wifi size={14} color={ACCENT_SUCCESS} />
          ) : (
            <WifiOff size={14} color={ACCENT_DANGER} />
          )}
          <Signal size={14} color={cameraOk ? TEXT_MAIN : "#CBD5E1"} />
        </div>
      </div>

      <div className="flex items-center gap-3 h-full">
        <div className="flex items-center gap-1.5">
          <span className="text-[12px] font-medium text-slate-500 capitalize">{engine} engine</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-[12px] font-medium text-slate-800">100%</span>
          <BatteryFull size={16} color={TEXT_MAIN} />
        </div>
      </div>
    </div>
  );
}

function ScreenHeader({ title, onBack, step }) {
  return (
    <div className="flex items-center justify-between px-8 py-5 flex-shrink-0 relative z-10">
      <div className="flex items-center gap-4 min-w-[160px]">
        {onBack ? (
          <button
            onClick={onBack}
            className="flex items-center justify-center w-10 h-10 rounded-full bg-slate-100 hover:bg-slate-200 transition-colors focus:outline-none focus-visible:ring-2 ring-emerald-500"
          >
            <ArrowLeft size={20} color={TEXT_MAIN} />
          </button>
        ) : (
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-full bg-gradient-to-tr from-emerald-400 to-teal-500 flex items-center justify-center shadow-lg shadow-emerald-500/20">
              <Eye size={18} color="#fff" />
            </div>
            <span className="text-xl font-semibold tracking-tight text-slate-800">RetinaScreen</span>
          </div>
        )}
      </div>

      <div className="absolute left-1/2 -translate-x-1/2">
        {onBack && <span className="text-lg font-medium tracking-wide text-slate-800">{title}</span>}
      </div>

      <div className="min-w-[160px] flex justify-end">
        {step && (
          <div className="px-4 py-1.5 rounded-full bg-emerald-50 border border-emerald-100 text-xs font-medium text-emerald-600">
            {step}
          </div>
        )}
      </div>
    </div>
  );
}

/* ---------- home ---------- */

function ModuleTile({ icon, label, description, onClick, primary }) {
  return (
    <button
      onClick={onClick}
      className={`group relative overflow-hidden flex flex-col items-start gap-4 p-6 rounded-2xl transition-all duration-300 ease-out text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 ${primary
        ? "w-full bg-white border border-emerald-100 hover:border-emerald-300 shadow-[0_8px_30px_rgba(16,185,129,0.1)] hover:shadow-[0_12px_40px_rgba(16,185,129,0.15)]"
        : "glass-panel hover:bg-slate-50/80 shadow-sm"
        }`}
    >
      <div className="absolute inset-0 bg-gradient-to-br from-emerald-50/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>

      <div
        className={`w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0 transition-transform duration-300 group-hover:scale-110 ${primary
          ? "bg-gradient-to-br from-emerald-400 to-teal-500 shadow-lg shadow-emerald-500/30"
          : "bg-slate-100 text-slate-700"
          }`}
      >
        {React.cloneElement(icon, { color: primary ? "#fff" : TEXT_MAIN, size: 24 })}
      </div>

      <div className="flex flex-col gap-1">
        <span className={`text-lg font-semibold ${primary ? "text-slate-900" : "text-slate-800"}`}>{label}</span>
        {description && <span className="text-sm text-slate-500 line-clamp-2">{description}</span>}
      </div>
    </button>
  );
}

function ReadinessChip({ ok, label, detail }) {
  return (
    <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-white border border-slate-200 shadow-sm">
      <div
        className="w-1.5 h-1.5 rounded-full"
        style={{ backgroundColor: ok ? ACCENT_SUCCESS : ACCENT_DANGER }}
      ></div>
      <span className="text-xs font-medium text-slate-600">
        {label}
        {detail ? <span className="text-slate-400"> · {detail}</span> : null}
      </span>
    </div>
  );
}

function HomeScreen({ goCapture, goResults, health }) {
  const cameraOk = Boolean(health && health.camera && health.camera.reachable);
  const modelOk = Boolean(health && health.model && health.model.available);
  const cameraSource = health && health.camera ? health.camera.source : null;
  const engine = health && health.model ? health.model.engine : null;

  return (
    <div className="flex flex-col h-full relative">
      <ScreenHeader />

      <div className="flex-1 flex flex-col items-center justify-center px-12 gap-6 relative z-10 -mt-12">
        <div className="w-full max-w-3xl">
          <ModuleTile
            primary
            icon={<Scan />}
            label="Initialize Screening"
            description="Activate the NoIR camera, align the pupil, and run the retinal analysis."
            onClick={goCapture}
          />
        </div>

        <div className="w-full max-w-3xl grid grid-cols-3 gap-5">
          <ModuleTile
            icon={<FileText />}
            label="Patient Records"
            description="Review past AI screening results and reports."
            onClick={goResults}
          />
          <ModuleTile
            icon={<Settings />}
            label="Calibration"
            description="System tuning and optical sensor alignment."
          />
          <ModuleTile icon={<HelpCircle />} label="Support" description="Documentation and usage guidelines." />
        </div>
      </div>

      <div className="pb-8 text-center relative z-10 flex flex-col items-center gap-3">
        <div className="flex items-center gap-2 flex-wrap justify-center">
          <ReadinessChip ok={cameraOk} label={cameraOk ? "Camera online" : "Camera offline"} detail={cameraSource} />
          <ReadinessChip ok={modelOk} label={modelOk ? "Model ready" : "Model unavailable"} detail={engine} />
        </div>
        <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-50 border border-emerald-100">
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></div>
          <span className="text-xs font-medium text-emerald-600">
            {health ? "System Ready" : "Connecting to backend…"}
          </span>
        </div>
      </div>

      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-emerald-400/5 rounded-full blur-[100px] pointer-events-none"></div>
    </div>
  );
}

/* ---------- capture ---------- */

function LiveViewfinder() {
  const [online, setOnline] = useState(true);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (online) return undefined;
    const timer = setTimeout(() => {
      setOnline(true);
      setAttempt((value) => value + 1);
    }, VIEWFINDER_RETRY_MS);
    return () => clearTimeout(timer);
  }, [online]);

  return (
    <>
      {online ? (
        <img
          key={attempt}
          src={`${viewfinderUrl()}?a=${attempt}`}
          onError={() => setOnline(false)}
          alt="Live retinal viewfinder"
          className="absolute inset-0 w-full h-full object-cover"
        />
      ) : (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-slate-900">
          <WifiOff size={32} className="text-rose-400" />
          <span className="text-xs font-mono uppercase tracking-widest text-rose-300">No camera signal</span>
          <span className="text-[11px] text-slate-400">Retrying…</span>
        </div>
      )}

      <div className="absolute inset-0 pointer-events-none">
        <div
          className="absolute inset-0 opacity-40"
          style={{
            backgroundImage:
              "linear-gradient(rgba(0,0,0,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,0,0,0.03) 1px, transparent 1px)",
            backgroundSize: "30px 30px",
          }}
        ></div>

        {[
          "top-6 left-6 border-t-2 border-l-2",
          "top-6 right-6 border-t-2 border-r-2",
          "bottom-6 left-6 border-b-2 border-l-2",
          "bottom-6 right-6 border-b-2 border-r-2",
        ].map((cls, i) => (
          <div key={i} className={`absolute w-10 h-10 ${cls} border-emerald-400 rounded-sm`} />
        ))}

        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-6 h-6 border border-emerald-400/70 rounded-full flex items-center justify-center bg-black/20">
          <div className="w-1 h-1 bg-emerald-400 rounded-full"></div>
        </div>

        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[280px] h-[280px] rounded-full border border-emerald-400/40 animate-pulse-ring"></div>

        <div className="absolute inset-0 overflow-hidden">
          <div className="w-full h-1 bg-emerald-400/60 shadow-[0_0_15px_rgba(16,185,129,0.8)] animate-scan-line"></div>
        </div>

        <div className="absolute bottom-4 left-6 flex items-center gap-2 bg-black/40 px-2 py-1 rounded-md backdrop-blur-sm">
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></div>
          <span className="text-[11px] font-mono text-emerald-300 uppercase tracking-wider">IR live</span>
        </div>
        <div className="absolute top-4 right-6 text-[11px] font-mono text-emerald-300 uppercase tracking-wider bg-black/40 px-2 py-1 rounded-md backdrop-blur-sm">
          Z-Axis: 0.00
        </div>
      </div>
    </>
  );
}

function CaptureScreen({ goHome, onResult }) {
  const [phase, setPhase] = useState("idle");
  const [flash, setFlash] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const run = async () => {
    setError(null);
    try {
      const captured = await capture();
      setPhase("analyzing");
      const analyzed = await analyzeCapture(captured.capture_id, false);
      onResult(analyzed);
    } catch (err) {
      setError(err.message || "Capture failed");
      setPhase("idle");
    }
  };

  const handleShutter = () => {
    if (phase !== "idle") return;
    setFlash(true);
    setTimeout(() => setFlash(false), 120);
    setPhase("capturing");
    run();
  };

  const handleUpload = async (event) => {
    const file = event.target.files && event.target.files[0];
    event.target.value = "";
    if (!file || phase !== "idle") return;
    setError(null);
    setPhase("analyzing");
    try {
      const analyzed = await analyzeFile(file, false);
      onResult(analyzed);
    } catch (err) {
      setError(err.message || "Analysis failed");
      setPhase("idle");
    }
  };

  const busy = phase !== "idle";

  return (
    <div className="flex flex-col h-full relative">
      <ScreenHeader
        title="Screening Active"
        onBack={busy ? null : goHome}
        step={phase === "analyzing" ? "Analyzing" : phase === "capturing" ? "Capturing" : "Alignment"}
      />

      <div className="flex-1 flex flex-col items-center justify-center gap-8">
        {busy ? (
          <div className="flex flex-col items-center gap-6">
            <div className="relative w-24 h-24">
              <div className="absolute inset-0 rounded-full border-4 border-slate-100"></div>
              <div className="absolute inset-0 rounded-full border-4 border-emerald-500 border-t-transparent animate-spin"></div>
              <div className="absolute inset-0 flex items-center justify-center bg-white rounded-full m-1">
                <Scan size={32} className="text-emerald-500 animate-pulse" />
              </div>
            </div>
            <div className="flex flex-col items-center gap-2">
              <span className="text-xl font-semibold text-slate-800 tracking-wide">
                {phase === "capturing" ? "Capturing Retina" : "Processing Retina"}
              </span>
              <span className="text-sm text-emerald-600 font-mono tracking-wider">RETFOUND_PIPELINE_ACTIVE</span>
            </div>
          </div>
        ) : (
          <>
            <div className="text-center space-y-1">
              <p className="text-lg font-medium text-slate-800">Center patient pupil in the frame</p>
              <p className="text-sm text-slate-500">IR viewfinder is active. White LED flashes only on capture.</p>
            </div>

            <div
              className="relative mx-auto rounded-3xl overflow-hidden shadow-xl shadow-slate-200 border border-slate-800 bg-slate-900"
              style={{ width: "420px", height: "420px" }}
            >
              <LiveViewfinder />
              {flash && (
                <div
                  className="absolute inset-0 bg-white z-50 transition-opacity duration-75"
                  style={{ opacity: 0.9 }}
                />
              )}
            </div>

            {error && (
              <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-rose-50 border border-rose-200 text-rose-600 text-sm">
                <AlertTriangle size={16} />
                <span>{error}</span>
                <button
                  onClick={handleShutter}
                  className="ml-2 underline underline-offset-2 font-medium hover:text-rose-700"
                >
                  Retry
                </button>
              </div>
            )}

            <div className="h-24 flex items-center justify-center gap-6">
              <button
                onClick={() => fileInputRef.current && fileInputRef.current.click()}
                className="flex items-center gap-2 px-5 py-3 rounded-full bg-white border border-slate-200 text-slate-600 text-sm font-medium hover:bg-slate-50 transition-colors shadow-sm"
                title="Analyze an existing image instead of the live camera"
              >
                <Upload size={16} />
                Upload image
              </button>

              <button
                onClick={handleShutter}
                className="group relative w-[72px] h-[72px] rounded-full flex items-center justify-center focus:outline-none shadow-lg shadow-emerald-500/20"
                aria-label="Capture image"
              >
                <div className="absolute inset-0 rounded-full border-4 border-emerald-500 opacity-80 group-hover:opacity-100 group-hover:scale-105 transition-all"></div>
                <div className="w-[58px] h-[58px] bg-emerald-500 rounded-full group-hover:scale-95 transition-transform group-active:scale-90 flex items-center justify-center">
                  <div className="w-[50px] h-[50px] bg-white rounded-full"></div>
                </div>
              </button>

              <button
                onClick={goHome}
                className="flex items-center gap-2 px-5 py-3 rounded-full bg-white border border-slate-200 text-slate-600 text-sm font-medium hover:bg-slate-50 transition-colors shadow-sm"
              >
                <Camera size={16} />
                Cancel
              </button>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleUpload}
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

/* ---------- result ---------- */

function QualityIssues({ quality }) {
  if (!quality) return null;
  const ISSUE_LABELS = {
    too_blurry: "Image too blurry",
    too_dark: "Image too dark",
    too_bright: "Image over-exposed",
    low_contrast: "Low contrast",
    no_retina_visible: "No retina detected in frame",
    no_background: "Framing too tight",
    empty_image: "Empty image",
  };
  return (
    <div className="flex flex-wrap gap-2">
      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-100 border border-slate-200 text-[11px] font-medium text-slate-600">
        Quality {quality.score != null ? `${quality.score}%` : "n/a"}
      </span>
      {quality.issues.map((issue) => (
        <span
          key={issue}
          className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-amber-50 border border-amber-200 text-[11px] font-medium text-amber-700"
        >
          <AlertTriangle size={12} />
          {ISSUE_LABELS[issue] || issue}
        </span>
      ))}
    </div>
  );
}

function ResultScreen({ outcome, goHome, goRecords }) {
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [saved, setSaved] = useState(false);

  const analysis = outcome && outcome.analysis;
  const quality = outcome && outcome.quality;
  const rejected = !analysis;
  const isLow = !rejected && analysis.risk === "low";
  const captureId = outcome && outcome.capture_id;
  const hasHeatmap = Boolean(outcome && outcome.heatmap_url);

  const statusColor = rejected
    ? "from-amber-400 to-orange-500"
    : isLow
      ? "from-emerald-400 to-teal-500"
      : "from-rose-500 to-orange-500";
  const badgeClasses = rejected
    ? "bg-amber-50 text-amber-600 border-amber-200"
    : isLow
      ? "bg-emerald-50 text-emerald-600 border-emerald-200"
      : "bg-rose-50 text-rose-600 border-rose-200";
  const badgeText = rejected ? "RECAPTURE NEEDED" : isLow ? "NO REFERRAL NEEDED" : "SPECIALIST REVIEW";
  const statusTitle = rejected ? "Image Quality Rejected" : isLow ? "Low Risk Detected" : "Referable Anomaly";
  const glowColor = rejected ? "rgba(245,158,11,0.1)" : isLow ? "rgba(16,185,129,0.1)" : "rgba(244,63,94,0.1)";

  const displaySrc = hasHeatmap && showHeatmap ? heatmapUrl(captureId) : imageUrl(captureId);
  const displayName = analysis ? analysis.label : "n/a";

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => goRecords(), 900);
  };

  return (
    <div className="flex flex-col h-full relative">
      <ScreenHeader title="Analysis Complete" onBack={goHome} step="Results" />

      <div className="flex-1 flex items-center justify-center gap-16 px-16 relative z-10">
        <div className="relative flex-shrink-0 group">
          <div className="absolute -inset-1 rounded-full bg-gradient-to-tr from-emerald-500/10 to-teal-500/10 blur-xl"></div>
          <div
            className="relative rounded-full overflow-hidden shadow-2xl ring-4 ring-white bg-slate-900"
            style={{ width: "300px", height: "300px" }}
          >
            <img
              src={displaySrc}
              alt="Captured retinal image"
              className="absolute inset-0 w-full h-full object-cover"
              onError={(event) => {
                event.currentTarget.style.display = "none";
              }}
            />
            {hasHeatmap && showHeatmap && !rejected && (
              <div className="absolute bottom-2 left-1/2 -translate-x-1/2 flex items-center gap-1.5 bg-white/85 px-2.5 py-1 rounded-full shadow-sm">
                <Activity size={12} className="text-rose-500" />
                <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-600">
                  AI attention
                </span>
              </div>
            )}
          </div>

          {hasHeatmap && !rejected && (
            <button
              onClick={() => setShowHeatmap((value) => !value)}
              className="absolute -bottom-3 left-1/2 -translate-x-1/2 px-4 py-1.5 rounded-full bg-white border border-slate-200 shadow-sm text-[11px] font-semibold text-slate-600 hover:bg-slate-50"
            >
              {showHeatmap ? "Hide heatmap" : "Show heatmap"}
            </button>
          )}
        </div>

        <div className="flex flex-col max-w-md w-full">
          <div className="bg-white rounded-3xl p-8 relative overflow-hidden shadow-xl shadow-slate-200/50 border border-slate-100">
            <div className={`absolute top-0 left-0 w-1.5 h-full bg-gradient-to-b ${statusColor}`}></div>

            <div className="flex flex-col gap-5 pl-2">
              <div>
                <span className={`inline-block px-3 py-1 rounded-full border text-[11px] font-bold tracking-widest ${badgeClasses}`}>
                  {badgeText}
                </span>
                <h2 className="text-3xl font-bold text-slate-800 mt-3">{statusTitle}</h2>
                <p className="text-sm text-slate-500 mt-1">{analysis ? analysis.diagnosis : "Captured frame did not pass the quality gate."}</p>
              </div>

              {!rejected && (
                <div className="flex items-end gap-3 pb-5 border-b border-slate-100">
                  <span className={`text-6xl font-bold tracking-tighter text-transparent bg-clip-text bg-gradient-to-br ${statusColor}`}>
                    {analysis.confidence}%
                  </span>
                  <span className="text-sm font-medium text-slate-500 mb-2">Model Confidence</span>
                </div>
              )}

              {!rejected && analysis.probabilities && (
                <div className="flex flex-col gap-2">
                  {Object.entries(analysis.probabilities).map(([name, value]) => (
                    <div key={name} className="flex items-center gap-3">
                      <span className="text-xs font-medium text-slate-500 w-24 capitalize">{name}</span>
                      <div className="flex-1 h-1.5 rounded-full bg-slate-100 overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{ width: `${value}%`, backgroundColor: name === displayName ? ACCENT_SUCCESS : "#CBD5E1" }}
                        ></div>
                      </div>
                      <span className="text-xs font-semibold text-slate-600 w-12 text-right">{value}%</span>
                    </div>
                  ))}
                </div>
              )}

              <QualityIssues quality={quality} />

              <div className="flex items-start gap-3">
                <Settings size={18} className="text-slate-400 mt-0.5 flex-shrink-0" />
                <p className="text-sm text-slate-600 leading-relaxed">
                  Automated AI screening for triage only. It does not replace a formal clinical diagnosis by a
                  certified ophthalmologist.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="pb-10 pt-4 flex items-center justify-center gap-5 z-10 relative">
        {saved ? (
          <div className="flex items-center gap-3 px-8 py-4 rounded-full bg-emerald-50 border border-emerald-200 shadow-sm">
            <CheckCircle2 size={20} className="text-emerald-600" />
            <span className="text-sm font-semibold tracking-wide text-emerald-600">RECORD SAVED</span>
          </div>
        ) : (
          <>
            <button
              onClick={handleSave}
              className="px-10 py-4 rounded-full font-semibold text-white shadow-lg shadow-emerald-500/25 transition-all hover:scale-105 hover:shadow-emerald-500/40 focus:outline-none focus-visible:ring-2 ring-emerald-500"
              style={{ background: "linear-gradient(135deg, #10B981 0%, #059669 100%)" }}
            >
              Save to Records
            </button>
            <button
              onClick={goHome}
              className="px-10 py-4 rounded-full font-semibold text-slate-700 bg-white border border-slate-200 hover:bg-slate-50 transition-all focus:outline-none focus-visible:ring-2 ring-slate-400 shadow-sm"
            >
              Discard
            </button>
          </>
        )}
      </div>

      <div
        className="absolute bottom-0 right-0 w-[500px] h-[500px] rounded-full blur-[120px] pointer-events-none opacity-60"
        style={{ backgroundColor: glowColor }}
      ></div>
    </div>
  );
}

/* ---------- past results ---------- */

function ResultsListScreen({ goHome, goCapture }) {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;
    listCaptures(30)
      .then((payload) => {
        if (active) setRecords(payload.captures || []);
      })
      .catch((err) => {
        if (active) setError(err.message || "Failed to load records");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="flex flex-col h-full relative">
      <ScreenHeader title="Patient Records" onBack={goHome} />

      <div className="flex-1 px-12 py-6 overflow-hidden flex flex-col z-10">
        {loading ? (
          <div className="flex-1 flex items-center justify-center text-slate-400 text-sm">Loading records…</div>
        ) : error ? (
          <div className="flex-1 flex items-center justify-center text-rose-500 text-sm">{error}</div>
        ) : records.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center gap-5">
            <div className="w-16 h-16 rounded-2xl bg-slate-100 flex items-center justify-center">
              <FileText size={32} className="text-slate-400" />
            </div>
            <span className="text-slate-500 font-medium">No screening records found.</span>
            <button
              onClick={goCapture}
              className="mt-2 px-6 py-2.5 rounded-full text-sm font-semibold text-emerald-600 bg-emerald-5 hover:bg-emerald-100 transition-colors border border-emerald-100"
            >
              Start New Screening
            </button>
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto pr-2 rounded-2xl border border-slate-200 bg-white shadow-sm">
            <div className="grid grid-cols-5 px-8 py-4 border-b border-slate-100 text-xs font-bold text-slate-400 uppercase tracking-wider sticky top-0 bg-white/95 backdrop-blur z-20">
              <span className="col-span-2">Timestamp</span>
              <span>Source</span>
              <span>Status</span>
              <span className="text-right">Confidence</span>
            </div>
            <div className="divide-y divide-slate-50">
              {records.map((record) => {
                const analysis = record.analysis;
                const risk = analysis ? analysis.risk : "rejected";
                const color = risk === "rejected" ? "#F59E0B" : risk === "low" ? ACCENT_SUCCESS : ACCENT_DANGER;
                const label = risk === "rejected" ? "Recapture" : risk === "low" ? "Low Risk" : "Referable";
                return (
                  <div key={record.capture_id} className="grid grid-cols-5 items-center px-8 py-4 hover:bg-slate-50 transition-colors">
                    <span className="text-sm font-medium text-slate-700 col-span-2">
                      {new Date(record.created_at).toLocaleString()}
                    </span>
                    <span className="text-xs text-slate-500">
                      {record.quality ? `Q ${record.quality.score}%` : "—"}
                    </span>
                    <span className="flex items-center gap-2.5">
                      <span
                        className="w-2 h-2 rounded-full"
                        style={{ backgroundColor: color, boxShadow: `0 0 6px ${color}` }}
                      ></span>
                      <span className="text-sm font-semibold text-slate-600">{label}</span>
                    </span>
                    <span className="text-sm font-bold text-slate-800 text-right">
                      {analysis ? `${analysis.confidence}%` : "—"}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ---------- device shell + app ---------- */

export default function RetinaScreen() {
  const [screen, setScreen] = useState("home");
  const [outcome, setOutcome] = useState(null);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    let active = true;
    const poll = () => {
      getHealth()
        .then((payload) => {
          if (active) setHealth(payload);
        })
        .catch(() => {
          if (active) setHealth(null);
        });
    };
    poll();
    const timer = setInterval(poll, HEALTH_POLL_MS);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  const goHome = () => setScreen("home");
  const goCapture = () => setScreen("capture");
  const goResultsList = () => setScreen("resultsList");

  const handleResult = (payload) => {
    setOutcome(payload);
    setScreen("result");
  };

  return (
    <div className="w-full h-screen flex flex-col overflow-hidden relative" style={{ backgroundColor: BG_LIGHT }}>
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
        <div className="absolute -top-[20%] -left-[10%] w-[70vw] h-[70vw] max-w-[800px] max-h-[800px] bg-emerald-200/40 rounded-full blur-[120px]"></div>
        <div className="absolute top-[40%] -right-[10%] w-[60vw] h-[60vw] max-w-[600px] max-h-[600px] bg-teal-200/30 rounded-full blur-[120px]"></div>
        <div className="absolute -bottom-[20%] left-[20%] w-[80vw] h-[80vw] max-w-[900px] max-h-[900px] bg-emerald-100/50 rounded-full blur-[140px]"></div>
      </div>

      <div
        className="absolute inset-0 opacity-[0.03] pointer-events-none z-0 mix-blend-overlay"
        style={{
          backgroundImage:
            'url("data:image/svg+xml,%3Csvg viewBox=%220 0 200 200%22 xmlns=%22http://www.w3.org/2000/svg%22%3E%3Cfilter id=%22noiseFilter%22%3E%3CfeTurbulence type=%22fractalNoise%22 baseFrequency=%220.65%22 numOctaves=%223%22 stitchTiles=%22stitch%22/%3E%3C/filter%3E%3Crect width=%22100%25%22 height=%22100%25%22 filter=%22url(%23noiseFilter)%22/%3E%3C/svg%3E")',
        }}
      ></div>

      <StatusBar health={health} />
      <div className="flex-1 flex flex-col overflow-y-auto overflow-x-hidden relative z-10">
        {screen === "home" && <HomeScreen goCapture={goCapture} goResults={goResultsList} health={health} />}
        {screen === "capture" && <CaptureScreen goHome={goHome} onResult={handleResult} />}
        {screen === "result" && outcome && (
          <ResultScreen outcome={outcome} goHome={goHome} goRecords={goResultsList} />
        )}
        {screen === "resultsList" && <ResultsListScreen goHome={goHome} goCapture={goCapture} />}
      </div>
    </div>
  );
}
