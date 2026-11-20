import React, { useState, useEffect, useRef, useCallback } from "react";
import L from "leaflet";
import {
  DEFAULT_API_URL,
  CLASSES,
  CLASS_META,
  ACTION_COLORS,
  ACTION_META,
  COUNTIES,
  PRESETS,
  STUDY_REGIONS,
  ZONE_COLORS,
} from "./constants";
import { KENYA_BOUNDARY_GEOJSON } from "./kenyaBoundary";
import { PREDICTION_POINTS } from "./predictionPoints";
import { REAL_COUNTY_SUMMARIES, ZONE_MODEL_FINDINGS } from "./countyData";
import MarkdownMessage from "./MarkdownMessage";
import {
  Compass, Layers, MapPin, CheckCircle2, ShieldAlert, Server, Sparkles,
  Settings, ChevronRight, Send, RefreshCw, ExternalLink, Cpu, Database,
  BarChart3, Bot, User, Check, Copy, Info, Filter, Maximize2, FileText,
  Activity, Layers3, ArrowRight, TrendingUp, Zap, Shield, Home, Menu, X,
  AlertTriangle, ChevronDown, ChevronUp, TreePine, Droplets, Building2,
  Wheat, Mountain, Globe, Award,
} from "lucide-react";

// ── Zone display metadata ─────────────────────────────────────────────────────
const ZONE_META = {
  highland_zone: { label: "Central Highlands", emoji: "🏔️", color: "#22c55e" },
  coastal_zone:  { label: "Coastal Strip",     emoji: "🌊", color: "#38bdf8" },
  asal_zone:     { label: "ASAL / Arid",        emoji: "🏜️", color: "#f97316" },
  lake_zone:     { label: "Lake Basin",          emoji: "💧", color: "#06b6d4" },
  rift_zone:     { label: "Rift Valley",         emoji: "⛰️", color: "#a855f7" },
};

// ── LOZO benchmark data ───────────────────────────────────────────────────────
const LOZO_BENCHMARKS = [
  { zone: "Highland (Nyeri, Murang'a)",   macro_f1: 0.505, baseline: 0.421, parcels: 7009, top_class: "Cropland" },
  { zone: "Rift Valley (Nakuru, Kajiado)", macro_f1: 0.544, baseline: 0.447, parcels: 4789, top_class: "Built-up" },
  { zone: "Lake Basin (Kisumu, Siaya)",    macro_f1: 0.473, baseline: 0.388, parcels: 4675, top_class: "Water" },
  { zone: "Coastal (Kilifi, Kwale, Lamu)", macro_f1: 0.367, baseline: 0.291, parcels: 6768, top_class: "Trees" },
  { zone: "ASAL (Turkana, Wajir)",         macro_f1: 0.273, baseline: 0.198, parcels: 4880, top_class: "Bare" },
];

// ── Action icon helper ────────────────────────────────────────────────────────
function ActionIcon({ action, size = 20 }) {
  if (action === "settle")  return <Building2  size={size} />;
  if (action === "farm")    return <Wheat      size={size} />;
  if (action === "protect") return <TreePine   size={size} />;
  return null;
}

// ── Animated Score Ring ────────────────────────────────────────────────────────
function ScoreRing({ value, color, label, max = 30 }) {
  const pct = Math.min(value / max, 1);
  const circumference = 2 * Math.PI * 34;
  const offset = circumference * (1 - pct);
  return (
    <div className="flex flex-col items-center justify-center gap-1.5 shrink-0">
      <div className="relative w-16 h-16 sm:w-20 sm:h-20 flex items-center justify-center">
        <svg className="w-full h-full rotate-[-90deg]" viewBox="0 0 80 80">
          <circle cx="40" cy="40" r="34" fill="none" stroke="rgba(51,65,85,0.4)" strokeWidth="6.5" />
          <circle
            cx="40" cy="40" r="34" fill="none"
            stroke={color} strokeWidth="6.5"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            style={{ transition: "stroke-dashoffset 1.1s cubic-bezier(0.22,1,0.36,1)" }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <span className="text-xs sm:text-sm font-extrabold font-mono tracking-tight" style={{ color }}>
            {value.toFixed(1)}%
          </span>
        </div>
      </div>
      <span className="text-[11px] font-bold text-slate-300 uppercase tracking-wider font-mono">
        {label}
      </span>
    </div>
  );
}

// ── Animated Bar ──────────────────────────────────────────────────────────────
function AnimBar({ pct, color, delay = 0 }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current) return;
    const t = setTimeout(() => {
      if (ref.current) ref.current.style.width = `${Math.max(pct, 2)}%`;
    }, delay);
    return () => clearTimeout(t);
  }, [pct, delay]);
  return (
    <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
      <div
        ref={ref}
        className="h-full rounded-full"
        style={{
          width: "0%",
          backgroundColor: color,
          transition: `width 0.9s cubic-bezier(0.22,1,0.36,1) ${delay}ms`,
        }}
      />
    </div>
  );
}

// ── Confidence badge ──────────────────────────────────────────────────────────
function ConfidenceBadge({ value }) {
  const pct = Math.round(value * 100);
  const color = pct >= 70 ? "text-emerald-400 bg-emerald-500/10 border-emerald-500/30"
              : pct >= 45 ? "text-amber-400  bg-amber-500/10  border-amber-500/30"
              :              "text-red-400    bg-red-500/10    border-red-500/30";
  return (
    <span className={`text-[10px] font-mono font-bold px-2.5 py-0.5 rounded-full border ${color}`}>
      {pct}% confidence
    </span>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
export default function App() {
  const [activeTab, setActiveTab]   = useState("map");
  const [mobileNav, setMobileNav]   = useState(false);
  const [apiUrl, setApiUrl]         = useState(localStorage.getItem("atlas_api_url") || DEFAULT_API_URL);
  const [apiOnline, setApiOnline]   = useState(null);
  const [apiPing, setApiPing]       = useState(null);
  const [pingHistory, setPingHistory] = useState([]);

  // Map / site state
  const [selectedPoint, setSelectedPoint] = useState(
    PREDICTION_POINTS.find((p) => p.region === "Central Highlands") || PREDICTION_POINTS[0]
  );
  const [selectedCoords, setSelectedCoords] = useState({ lat: -0.4201, lon: 36.9476 });
  const [floodRisk, setFloodRisk]   = useState(false);
  const [slopeDeg, setSlopeDeg]     = useState(4.2);
  const [actionFilter, setActionFilter] = useState("all");

  // County tab state
  const [selectedCounty, setSelectedCounty] = useState("nakuru");
  const [countyData, setCountyData]         = useState(REAL_COUNTY_SUMMARIES["nakuru"] || null);
  const [expandedZone, setExpandedZone]     = useState(null);

  // Chat state
  const [chatMessages, setChatMessages] = useState([
    {
      id: "welcome", role: "assistant", timestamp: "Just now",
      content: `Hello! I am your **Kenya Spatial Underwriting Copilot** powered by **GROQ (Llama 3.3 70B)** and statutory RAG.

I have loaded the spatial assessment for:
📍 **Nyeri Site #1** (\`-0.4201°, 36.9476°\`)
• **Agro-Ecological Zone:** Central Highlands (\`highland_zone\`)
• **Statutory Allocation:** **FARM (Agriculture)**
• **Dominant Land Cover:** Cropland / Agriculture (75% confidence)
• **Topography:** 4.2° Slope · 1,840m Elevation · Low Flood Hazard

Ask me anything regarding statutory zoning, **PLUPA 2019 Sec. 56**, **NEMA EMCA** environmental licenses, or capital allocation!`,
    },
  ]);
  const [inputQuery, setInputQuery] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const chatBottomRef = useRef(null);

  // Settings
  const [showSettings, setShowSettings] = useState(false);

  // Leaflet refs
  const mapRef          = useRef(null);
  const mapInstance     = useRef(null);
  const markerRef       = useRef(null);
  const markersLayerRef = useRef(null);

  // ── API Health ──────────────────────────────────────────────────────────────
  const checkHealth = useCallback(async () => {
    const start = performance.now();
    const isHttps = typeof window !== "undefined" && window.location.protocol === "https:";
    let targetUrl = apiUrl;
    if (isHttps && targetUrl.startsWith("http://")) {
      targetUrl = "/api-proxy";
      setApiUrl("/api-proxy");
      localStorage.setItem("atlas_api_url", "/api-proxy");
    }
    try {
      let res = await fetch(`${targetUrl}/health`, { method: "GET" }).catch(() => null);
      if ((!res || !res.ok) && isHttps && targetUrl !== "/api-proxy") {
        res = await fetch("/api-proxy/health", { method: "GET" }).catch(() => null);
        if (res && res.ok) { setApiUrl("/api-proxy"); localStorage.setItem("atlas_api_url", "/api-proxy"); targetUrl = "/api-proxy"; }
      }
      const duration = Math.round(performance.now() - start);
      if (res && res.ok) {
        setApiOnline(true);
        setApiPing(duration);
        setPingHistory(h => [...h.slice(-11), duration]);
      } else {
        setApiOnline(false);
      }
    } catch { setApiOnline(false); }
  }, [apiUrl]);

  useEffect(() => {
    checkHealth();
    const iv = setInterval(checkHealth, 15000);
    return () => clearInterval(iv);
  }, [checkHealth]);

  // Auto-scroll chat
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages, chatLoading]);

  // ── Leaflet map init ────────────────────────────────────────────────────────
  useEffect(() => {
    if (!mapRef.current || mapInstance.current) return;
    const kenyaBounds = L.latLngBounds(L.latLng(-5.1, 33.6), L.latLng(5.3, 42.1));
    const map = L.map(mapRef.current, {
      center: [0.35, 37.8], zoom: 6.4, minZoom: 6, maxZoom: 18,
      maxBounds: kenyaBounds, maxBoundsViscosity: 1.0, zoomControl: false,
    });
    L.control.zoom({ position: "topright" }).addTo(map);

    const esriSatellite   = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", { attribution: "Esri World Imagery", maxZoom: 18 });
    const esriBoundaries  = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}", { attribution: "Esri Boundaries", maxZoom: 18 });
    const esriTopo        = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}", { attribution: "Esri Topo", maxZoom: 18 });

    const kenyaBorder = L.geoJSON(KENYA_BOUNDARY_GEOJSON, {
      style: { color: "#0f172a", weight: 2.2, opacity: 0.95, fillOpacity: 0, dashArray: "6, 3" },
    });

    const satelliteGroup = L.layerGroup([esriSatellite, esriBoundaries, kenyaBorder]).addTo(map);
    L.control.layers({ "🛰️ Satellite": satelliteGroup, "🗺️ Topographic": esriTopo }, {}, { position: "topright" }).addTo(map);

    STUDY_REGIONS.forEach((r) => {
      const [minLon, minLat, maxLon, maxLat] = r.bbox;
      const col = ZONE_COLORS[r.zone] || "#06b6d4";
      L.rectangle([[minLat, minLon], [maxLat, maxLon]], { color: col, weight: 1.5, fillColor: col, fillOpacity: 0.06, dashArray: "4, 4" })
        .on("click", () => map.fitBounds([[minLat, minLon], [maxLat, maxLon]], { padding: [30, 30] }))
        .addTo(map);
    });

    const markersLayer = L.layerGroup().addTo(map);
    markersLayerRef.current = markersLayer;

    const customIcon = L.divIcon({
      className: "custom-pin",
      html: `<div class="relative flex items-center justify-center pointer-events-none">
        <div class="absolute w-8 h-8 bg-cyan-400/40 rounded-full animate-ping"></div>
        <div class="w-4 h-4 bg-cyan-400 border-2 border-white rounded-full shadow-lg shadow-cyan-500/50"></div>
      </div>`,
      iconSize: [24, 24], iconAnchor: [12, 12],
    });

    const marker = L.marker([selectedCoords.lat, selectedCoords.lon], { icon: customIcon, zIndexOffset: 1000 }).addTo(map);
    markerRef.current = marker;

    map.on("click", (e) => {
      const { lat, lng } = e.latlng;
      setSelectedCoords({ lat, lon: lng });
      marker.setLatLng([lat, lng]);
    });

    mapInstance.current = map;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Render prediction points ────────────────────────────────────────────────
  useEffect(() => {
    if (!mapInstance.current || !markersLayerRef.current) return;
    const ml = markersLayerRef.current;
    ml.clearLayers();
    const filtered = actionFilter === "all" ? PREDICTION_POINTS : PREDICTION_POINTS.filter((p) => p.action === actionFilter);
    filtered.forEach((pt) => {
      const actColor = ACTION_COLORS[pt.action] || "#3b82f6";
      const cm = L.circleMarker([pt.lat, pt.lon], { radius: 6, color: "#ffffff", weight: 1.5, fillColor: actColor, fillOpacity: 0.95 });
      cm.bindTooltip(
        `<div style="font-family:sans-serif;font-size:11px;"><b>${pt.name}</b> · <span style="color:${actColor};font-weight:700;text-transform:uppercase;">[${pt.action}]</span><br/><span style="color:#94a3b8;">${pt.region} (${pt.label.replace("_", " ")}) · ${pt.slope}° slope</span></div>`,
        { sticky: true, className: "custom-tooltip" }
      );
      cm.bindPopup(
        `<div style="font-family:system-ui,sans-serif;min-width:210px;padding:4px;">
          <div style="font-weight:700;font-size:13px;color:#0f172a;margin-bottom:2px;">${pt.name}</div>
          <div style="font-size:11px;color:#64748b;margin-bottom:8px;">${pt.region} · <span style="text-transform:uppercase;">${pt.zone.replace("_zone", "")}</span></div>
          <div style="display:flex;gap:6px;margin-bottom:8px;">
            <span style="background:${actColor}22;color:${actColor};border:1px solid ${actColor}55;padding:2px 8px;border-radius:9999px;font-size:10px;font-weight:700;text-transform:uppercase;">${pt.action}</span>
            <span style="background:#f1f5f9;color:#334155;padding:2px 8px;border-radius:9999px;font-size:10px;font-weight:600;text-transform:capitalize;">${pt.label.replace("_", " ")}</span>
          </div>
          <div style="font-size:11px;color:#475569;margin-bottom:8px;">Slope: <b>${pt.slope}°</b> · Elev: <b>${pt.elev}m</b> · NDVI: <b>${pt.ndvi}</b></div>
          <div style="font-size:10px;color:#64748b;font-style:italic;border-top:1px solid #e2e8f0;padding-top:6px;">${pt.rationale}</div>
        </div>`,
        { closeButton: false }
      );
      cm.on("click", (e) => { L.DomEvent.stopPropagation(e); handleSelectPoint(pt); });
      cm.addTo(ml);
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [actionFilter]);

  const handleSelectPoint = (pt) => {
    setSelectedPoint(pt);
    setSelectedCoords({ lat: pt.lat, lon: pt.lon });
    setSlopeDeg(pt.slope);
    setFloodRisk(pt.flood_risk);
    if (markerRef.current) markerRef.current.setLatLng([pt.lat, pt.lon]);
    if (mapInstance.current) mapInstance.current.flyTo([pt.lat, pt.lon], 11, { duration: 1.0 });
    setChatMessages([{
      id: `intro_${pt.id}`, role: "assistant", timestamp: "Just now",
      content: `📍 **Active Spatial Site Loaded:** **${pt.name}** (\`${pt.lat.toFixed(4)}°, ${pt.lon.toFixed(4)}°\`)
• **Region:** ${pt.region} (${pt.zone.replace("_zone", "").toUpperCase()})
• **Model Allocation:** **${pt.action.toUpperCase()}**
• **Dominant Land Cover:** ${pt.label.replace("_", " ").toUpperCase()} (${((pt.probabilities?.[pt.label] || 0.75) * 100).toFixed(0)}% confidence)
• **Topography:** ${pt.slope}° Terrain Slope · ${pt.elev}m Elevation · ${pt.flood_risk ? "⚠️ High Flood Risk" : "✅ Low Flood Exposure"}

How can I assist you with statutory underwriting, zoning permits, or environmental compliance for this site?`,
    }]);
  };

  // ── County sync ────────────────────────────────────────────────────────────
  useEffect(() => {
    const s = REAL_COUNTY_SUMMARIES[selectedCounty.toLowerCase()];
    if (s) setCountyData(s);
  }, [selectedCounty]);

  // ── RAG Chatbot ─────────────────────────────────────────────────────────────
  const handleSendMessage = async (textToSend) => {
    const query = textToSend || inputQuery;
    if (!query.trim() || chatLoading) return;
    const userMsg = { id: `usr_${Date.now()}`, role: "user", content: query, timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) };
    setChatMessages(p => [...p, userMsg]);
    setInputQuery("");
    setChatLoading(true);
    try {
      const res = await fetch(`${apiUrl}/underwrite/copilot`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          lat: selectedCoords.lat, lon: selectedCoords.lon,
          probabilities: selectedPoint.probabilities,
          prediction: selectedPoint.label,
          recommendation: { action: selectedPoint.action, confidence: selectedPoint.confidence || "high", rationale: selectedPoint.rationale },
          flood_risk: floodRisk, slope_deg: slopeDeg,
          county_name: selectedPoint.region, user_query: query,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setChatMessages(p => [...p, { id: `bot_${Date.now()}`, role: "assistant", content: data.dossier, timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) }]);
        return;
      }
    } catch (err) { console.warn("Backend offline, using local RAG:", err); }
    finally { setChatLoading(false); }
    const localAnswer = generateLocalRAGAnswer(query, selectedPoint, slopeDeg, floodRisk);
    setChatMessages(p => [...p, { id: `bot_${Date.now()}`, role: "assistant", content: localAnswer, timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) }]);
  };

  const generateLocalRAGAnswer = (q, pt, slope, flood) => {
    const qLower = q.toLowerCase();
    const action = pt.action.toUpperCase();
    if (qLower.includes("plupa") || qLower.includes("riparian") || qLower.includes("setback")) {
      return `### 📜 Statutory Riparian & Slope Audit (PLUPA 2019 Sec. 56)
• **Riparian Setback Requirement:** A mandatory **30-meter non-buildable setback** is enforced from all seasonal and permanent watercourses under Kenya's *Physical and Land Use Planning Act (2019) Section 56*.
• **Current Site Hazard:** ${flood ? "⚠️ This parcel is within a designated hydrologic flood basin. Permanent civil settlement is **strictly prohibited** without an approved river-basin engineering canalization plan." : "✅ No immediate riparian conflict detected at this coordinate."}
• **Slope Regulation:** Measured slope is **${slope.toFixed(1)}°**. ${slope > 15 ? "Development on slopes >15° requires geotechnical certification from the National Building Review Board (NBRB)." : "Slope is within normal development tolerance (<15°)."}`;
    }
    if (qLower.includes("nema") || qLower.includes("eia") || qLower.includes("convert") || qLower.includes("sub-div")) {
      return `### 🌿 NEMA EMCA Environmental Audit & Conversion Standards
• **Agricultural Land Protection:** Under EMCA (Cap 387) and Second Schedule Regulations, any conversion of prime agricultural land exceeding **2.0 hectares** to built commercial fabric mandates a full **Environmental Impact Assessment (EIA) License**.
• **Site Status:** This parcel is classified as **${pt.label.replace("_", " ").toUpperCase()}**. ${pt.action === "farm" ? "County agricultural zoning preserves smallholder yields. Sub-division into residential plots below 0.5ha is restricted by county physical planning policies." : "Brownfield settlement infill is permissible subject to county public participation notice."}`;
    }
    if (qLower.includes("can we build") || qLower.includes("housing") || qLower.includes("settle")) {
      return `### 🏗️ Land Suitability & Housing Allocation Brief
• **Model Recommendation:** **${action}** (${pt.confidence?.toUpperCase() || "HIGH"} CONFIDENCE)
• **Key Evaluation:** ${action === "SETTLE" ? "✅ Site recommended for affordable housing / commercial buildout. Stable slope (" + slope + "°) and low hydrologic flood exposure." : (action === "PROTECT" ? "⛔ Direct settlement is restricted. Classified for conservation buffer due to steep terrain or hydrologic sensitivity." : "⚠️ Priority designated for agricultural tenure. Conversion to high-density residential requires county spatial re-zoning clearance.")}
• **Infrastructure Index:** Distance to primary transit corridor: **${pt.region} arterial zone**.`;
    }
    return `### 📋 Executive Spatial Underwriting Dossier
**Site:** ${pt.name} (\`${pt.lat.toFixed(4)}°, ${pt.lon.toFixed(4)}°\`) · **Region:** ${pt.region}  
**Statutory Allocation:** **${action}** (${pt.confidence?.toUpperCase() || "HIGH"} CONFIDENCE)

---

#### 1. 🏢 Topographic & Physical Characteristics
- **Multi-Spectral Cover:** Classified as **${pt.label.replace("_", " ").toUpperCase()}** (${((pt.probabilities?.[pt.label] || 0.75) * 100).toFixed(0)}% confidence).
- **Slope:** Measured at **${slope.toFixed(1)}°**. ${slope > 15 ? "Exceeds 15° threshold — high soil erosion hazard." : "Gradual gradient favorable for civil works."}
- **Flood Risk:** ${flood ? "Surface water drainage exposure flagged." : "Low hydrologic risk."}

#### 2. 📜 Statutory Zoning Compliance (RAG Engine)
- **PLUPA 2019 Sec. 56:** Enforces 30m riparian setback and engineering approval for hillside developments.
- **NEMA EMCA:** Conversion requires public notice and EIA environmental license.
- **Recommendation:** *${pt.rationale}*`;
  };

  // ── Derived state ───────────────────────────────────────────────────────────
  const actionStyle = ACTION_META[selectedPoint.action] || ACTION_META.farm;
  const totalCount   = PREDICTION_POINTS.length;
  const settleCount  = PREDICTION_POINTS.filter(p => p.action === "settle").length;
  const farmCount    = PREDICTION_POINTS.filter(p => p.action === "farm").length;
  const protectCount = PREDICTION_POINTS.filter(p => p.action === "protect").length;

  // County card metadata
  const COUNTY_DISPLAY = {
    nakuru:    { name: "Nakuru",      flag: "🌋", zone: "rift_zone" },
    highland:  { name: "Highland",    flag: "🏔️", zone: "highland_zone" },
    muranga:   { name: "Murang'a",    flag: "☕", zone: "highland_zone" },
    nyandarua: { name: "Nyandarua",   flag: "🌿", zone: "highland_zone" },
    coast:     { name: "Coast",       flag: "🌴", zone: "coastal_zone" },
    kwale:     { name: "Kwale",       flag: "🏖️", zone: "coastal_zone" },
    lamu:      { name: "Lamu",        flag: "⛵", zone: "coastal_zone" },
    arid:      { name: "Arid (Marsabit)", flag: "🏜️", zone: "asal_zone" },
    turkana:   { name: "Turkana",     flag: "🌵", zone: "asal_zone" },
    wajir:     { name: "Wajir",       flag: "🐪", zone: "asal_zone" },
    lake_basin:{ name: "Lake Basin",  flag: "💧", zone: "lake_zone" },
    siaya:     { name: "Siaya",       flag: "🐟", zone: "lake_zone" },
    kajiado:   { name: "Kajiado",     flag: "🦁", zone: "rift_zone" },
  };

  // ── NAV TABS config ─────────────────────────────────────────────────────────
  const TABS = [
    { id: "map",        icon: <Compass size={15} />,   label: "Spatial Atlas" },
    { id: "underwrite", icon: <Bot size={15} />,       label: "Site Underwriting" },
    { id: "county",     icon: <BarChart3 size={15} />, label: "County Dashboard" },
    { id: "mlops",      icon: <Cpu size={15} />,       label: "MLOps & System" },
  ];

  // ── RENDER ──────────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col h-[100dvh] w-screen overflow-hidden bg-[#080c14] text-slate-100 font-sans">

      {/* ════════════════════════════════════════════════════════════════════════
          HEADER
      ════════════════════════════════════════════════════════════════════════ */}
      <header className="shrink-0 h-14 sm:h-16 px-3 sm:px-6 border-b border-slate-800/80 bg-slate-900/80 backdrop-blur-xl flex items-center justify-between z-30 gap-2">

        {/* Logo */}
        <div className="flex items-center gap-2 sm:gap-3 shrink-0">
          <div className="w-8 h-8 sm:w-9 sm:h-9 rounded-xl bg-gradient-to-tr from-cyan-500 to-emerald-400 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Compass className="w-4 h-4 sm:w-5 sm:h-5 text-slate-950 font-bold" />
          </div>
          <div>
            <h1 className="font-bold text-sm sm:text-base tracking-tight gradient-text-cyan leading-none">
              Kenya Land Atlas
            </h1>
            <p className="text-[10px] text-slate-400 header-subtitle leading-none mt-0.5 hidden sm:block">
              Sentinel-2 · 13 Counties · Statutory RAG
            </p>
          </div>
        </div>

        {/* Desktop tabs */}
        <nav className="hidden sm:flex items-center gap-1 p-1 bg-slate-950/70 border border-slate-800/80 rounded-xl">
          {TABS.map(t => (
            <button
              key={t.id}
              onClick={() => { setActiveTab(t.id); setMobileNav(false); }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 ${
                activeTab === t.id
                  ? "bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
              }`}
            >
              {t.icon}
              <span>{t.label}</span>
            </button>
          ))}
        </nav>

        {/* Right cluster */}
        <div className="flex items-center gap-2 shrink-0">
          {/* API status pill */}
          <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs">
            <span className={`w-1.5 h-1.5 rounded-full ${apiOnline ? "bg-emerald-400 pulse-dot" : "bg-slate-600"}`} />
            <span className={`font-mono text-[11px] ${apiOnline ? "text-emerald-400" : "text-slate-500"}`}>
              {apiOnline ? `${apiPing}ms` : "offline"}
            </span>
          </div>

          {/* Settings */}
          <button
            onClick={() => setShowSettings(true)}
            className="p-2 rounded-lg bg-slate-800/80 hover:bg-slate-700 border border-slate-700 text-slate-300 transition-colors"
          >
            <Settings className="w-4 h-4" />
          </button>

          {/* Mobile hamburger */}
          <button
            onClick={() => setMobileNav(v => !v)}
            className="sm:hidden p-2 rounded-lg bg-slate-800/80 hover:bg-slate-700 border border-slate-700 text-slate-300"
          >
            {mobileNav ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
          </button>
        </div>
      </header>

      {/* Mobile nav dropdown */}
      {mobileNav && (
        <div className="sm:hidden shrink-0 bg-slate-900/95 backdrop-blur-xl border-b border-slate-800 px-3 py-2 flex flex-col gap-1 z-20">
          {TABS.map(t => (
            <button
              key={t.id}
              onClick={() => { setActiveTab(t.id); setMobileNav(false); }}
              className={`flex items-center gap-2 px-4 py-3 rounded-xl text-sm font-semibold transition-all ${
                activeTab === t.id
                  ? "bg-cyan-500 text-slate-950"
                  : "text-slate-300 hover:bg-slate-800"
              }`}
            >
              {t.icon} {t.label}
            </button>
          ))}
        </div>
      )}

      {/* ════════════════════════════════════════════════════════════════════════
          TAB CONTENT
      ════════════════════════════════════════════════════════════════════════ */}
      <div className="flex-1 overflow-hidden relative">

        {/* ── TAB 1: SPATIAL ATLAS ─────────────────────────────────────────────── */}
        <div className={`w-full h-full relative ${activeTab === "map" ? "block" : "hidden"}`}>
          <div ref={mapRef} className="w-full h-full bg-slate-950" />

          {/* Filter pill bar */}
          <div className="absolute top-3 left-3 z-[1000] flex flex-wrap items-center gap-1.5 p-1.5 bg-slate-900/90 backdrop-blur-md rounded-2xl border border-slate-800/90 shadow-2xl max-w-[calc(100vw-80px)]">
            <span className="text-[11px] text-slate-400 font-medium px-1 flex items-center gap-1">
              <Filter className="w-3 h-3 text-cyan-400" />
              <span className="hidden sm:inline">Filter:</span>
            </span>
            {[
              { key: "all",     label: `All (${totalCount})`,         color: "bg-slate-100 text-slate-950", base: "bg-slate-800/70 text-slate-300" },
              { key: "settle",  label: `Settle (${settleCount})`,     color: "bg-blue-500 text-white",     base: "bg-blue-500/10 text-blue-300 border border-blue-500/30" },
              { key: "farm",    label: `Farm (${farmCount})`,         color: "bg-emerald-500 text-slate-950", base: "bg-emerald-500/10 text-emerald-300 border border-emerald-500/30" },
              { key: "protect", label: `Protect (${protectCount})`,   color: "bg-amber-500 text-slate-950", base: "bg-amber-500/10 text-amber-300 border border-amber-500/30" },
            ].map(f => (
              <button key={f.key} onClick={() => setActionFilter(f.key)}
                className={`text-xs px-2.5 py-1 rounded-lg transition-all font-semibold ${actionFilter === f.key ? f.color : f.base}`}>
                {f.label}
              </button>
            ))}
            <div className="h-4 w-px bg-slate-700 mx-0.5" />
            <select
              onChange={(e) => {
                const reg = STUDY_REGIONS.find(r => r.name === e.target.value);
                if (reg && mapInstance.current) {
                  const [minLon, minLat, maxLon, maxLat] = reg.bbox;
                  mapInstance.current.fitBounds([[minLat, minLon], [maxLat, maxLon]], { padding: [30, 30] });
                }
              }}
              className="text-xs bg-slate-800 border border-slate-700 rounded-lg px-2 py-1 text-slate-200 cursor-pointer focus:outline-none max-w-[130px] sm:max-w-none"
              defaultValue=""
            >
              <option value="" disabled>Zoom Region ▾</option>
              {STUDY_REGIONS.map(r => <option key={r.name} value={r.name}>{r.name}</option>)}
            </select>
          </div>

          {/* Bottom site inspector card */}
          <div className="absolute bottom-4 left-3 right-3 sm:left-6 sm:right-auto sm:max-w-sm z-[1000] p-3 sm:p-4 rounded-2xl bg-slate-900/95 backdrop-blur-xl border border-slate-800 shadow-2xl flex items-center justify-between gap-3">
            <div className="space-y-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-[9px] font-mono uppercase text-slate-400">Active Parcel</span>
                <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded-full uppercase ${actionStyle.bg} ${actionStyle.border} ${actionStyle.text}`}>
                  {selectedPoint.action}
                </span>
                {selectedPoint.flood_risk && (
                  <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-amber-500/10 border border-amber-500/30 text-amber-300">⚠️ Flood</span>
                )}
              </div>
              <h3 className="font-bold text-sm text-slate-100 truncate">{selectedPoint.name}</h3>
              <p className="text-[11px] text-slate-400 truncate">{selectedPoint.region} · {selectedPoint.slope}° · {selectedPoint.elev}m</p>
            </div>
            <button
              onClick={() => setActiveTab("underwrite")}
              className="px-3 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs flex items-center gap-1 shadow-lg shadow-cyan-500/20 transition-all shrink-0 cursor-pointer"
            >
              Underwrite <ArrowRight className="w-3 h-3" />
            </button>
          </div>
        </div>

        {/* ── TAB 2: SITE UNDERWRITING & RAG COPILOT ──────────────────────────── */}
        {activeTab === "underwrite" && (
          <div className="tab-enter w-full h-full overflow-y-auto overflow-x-hidden">

            {/* Hero verdict banner */}
            <div className={`border-b border-slate-800 px-4 sm:px-8 py-5 sm:py-7 verdict-${selectedPoint.action}`}>
              <div className="max-w-6xl mx-auto flex flex-col sm:flex-row sm:items-center gap-4 sm:gap-8">

                {/* Decision badge */}
                <div className={`flex items-center gap-4 p-4 sm:p-5 rounded-2xl border flex-1 verdict-${selectedPoint.action}`}>
                  <div className={`w-14 h-14 sm:w-16 sm:h-16 rounded-2xl flex items-center justify-center shrink-0 ${actionStyle.bg} border ${actionStyle.border}`}>
                    <ActionIcon action={selectedPoint.action} size={28} />
                  </div>
                  <div className="min-w-0">
                    <div className="text-[10px] uppercase font-mono text-slate-400 tracking-wider mb-1">Statutory Allocation</div>
                    <h2 className={`text-2xl sm:text-3xl font-black tracking-tight hero-site-name ${actionStyle.text}`}>
                      {actionStyle.badge}
                    </h2>
                    <div className="text-xs sm:text-sm text-slate-300 mt-1 font-medium">{actionStyle.label}</div>
                    <ConfidenceBadge value={selectedPoint.probabilities?.[selectedPoint.label] || 0.75} />
                  </div>
                </div>

                {/* Site coordinates */}
                <div className="flex-1 sm:max-w-xs space-y-2">
                  <div className="text-[10px] uppercase font-mono text-slate-400 tracking-wider">📍 Active Site</div>
                  <div className="text-base sm:text-lg font-bold text-slate-100 truncate">{selectedPoint.name}</div>
                  <div className="text-xs text-slate-400">{selectedPoint.region} · {selectedPoint.zone.replace("_zone","").toUpperCase()}</div>
                  <div className="grid grid-cols-3 gap-2 mt-2">
                    {[
                      { label: "Slope", val: `${selectedPoint.slope}°`, ok: selectedPoint.slope <= 15 },
                      { label: "Elev",  val: `${selectedPoint.elev}m`,  ok: true },
                      { label: "Flood", val: selectedPoint.flood_risk ? "HIGH" : "LOW", ok: !selectedPoint.flood_risk },
                    ].map(d => (
                      <div key={d.label} className={`p-2 rounded-xl text-center border ${d.ok ? "bg-emerald-500/5 border-emerald-500/20" : "bg-amber-500/5 border-amber-500/20"}`}>
                        <div className="text-[9px] text-slate-400 uppercase">{d.label}</div>
                        <div className={`text-xs font-bold font-mono ${d.ok ? "text-emerald-300" : "text-amber-300"}`}>{d.val}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Quick site picker */}
                <div className="hidden lg:block w-52 shrink-0">
                  <div className="text-[10px] uppercase font-mono text-slate-400 mb-2 tracking-wider">Switch Site</div>
                  <div className="space-y-1 max-h-44 overflow-y-auto pr-1">
                    {PREDICTION_POINTS.slice(0, 15).map(pt => {
                      const am = ACTION_META[pt.action];
                      return (
                        <button key={pt.id} onClick={() => handleSelectPoint(pt)}
                          className={`w-full text-left px-2.5 py-1.5 rounded-lg text-[11px] transition-all flex items-center gap-2 ${
                            selectedPoint.id === pt.id ? `${am.bg} ${am.text} border ${am.border}` : "text-slate-400 hover:bg-slate-800 hover:text-slate-200"
                          }`}
                        >
                          <span className={`w-1.5 h-1.5 rounded-full shrink-0`} style={{ backgroundColor: ACTION_COLORS[pt.action] }} />
                          <span className="truncate">{pt.name}</span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>

            {/* Main content: evidence + chat */}
            <div className="max-w-6xl mx-auto px-3 sm:px-6 py-4 sm:py-6 flex flex-col lg:flex-row gap-4 sm:gap-6 h-auto lg:h-[calc(100vh-14rem)]">

              {/* Evidence panel */}
              <div className="w-full lg:w-72 xl:w-80 shrink-0 space-y-4 lg:overflow-y-auto lg:pr-1">

                {/* Rationale card */}
                <div className="glass-card p-4 space-y-2">
                  <div className="text-[10px] uppercase font-mono text-slate-400 tracking-wider">Model Rationale</div>
                  <p className="text-xs text-slate-300 leading-relaxed italic">"{selectedPoint.rationale}"</p>
                </div>

                {/* Land-cover probability bars */}
                <div className="glass-card p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="text-[10px] uppercase font-mono text-slate-400 tracking-wider">Spectral Mix</div>
                    <span className="text-[9px] text-slate-500">NDVI: {selectedPoint.ndvi}</span>
                  </div>
                  <div className="space-y-2.5">
                    {CLASSES.map((cls, i) => {
                      const prob = selectedPoint.probabilities?.[cls] || 0.05;
                      const meta = CLASS_META[cls];
                      return (
                        <div key={cls}>
                          <div className="flex justify-between text-[11px] mb-1">
                            <span className="flex items-center gap-1.5 text-slate-300">
                              <span className="w-2 h-2 rounded-sm shrink-0" style={{ backgroundColor: meta.color }} />
                              {meta.label}
                            </span>
                            <span className="font-mono text-slate-400 font-semibold">{(prob * 100).toFixed(0)}%</span>
                          </div>
                          <AnimBar pct={prob * 100} color={meta.color} delay={i * 80} />
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Topographic controls */}
                <div className="glass-card p-4 space-y-3">
                  <div className="text-[10px] uppercase font-mono text-slate-400 tracking-wider">Scenario Adjust</div>
                  <div>
                    <div className="flex justify-between text-xs mb-2">
                      <span className="text-slate-300 flex items-center gap-1"><Mountain size={12} /> Terrain Slope</span>
                      <span className={`font-mono font-bold ${slopeDeg > 15 ? "text-amber-300" : "text-emerald-300"}`}>{slopeDeg}°</span>
                    </div>
                    <input type="range" min="0" max="30" step="0.5" value={slopeDeg}
                      onChange={e => setSlopeDeg(parseFloat(e.target.value))}
                      className="w-full accent-cyan-400 h-1 bg-slate-700 rounded-lg cursor-pointer" />
                    <div className="flex justify-between text-[9px] text-slate-500 mt-1">
                      <span>Flat</span><span className="text-amber-400">≥15° Risk</span><span>Steep 30°</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between pt-1 border-t border-slate-800">
                    <span className="text-xs text-slate-300 flex items-center gap-1.5"><Droplets size={12} /> Flood Exposure</span>
                    <button
                      onClick={() => setFloodRisk(v => !v)}
                      className={`relative w-10 h-5 rounded-full transition-all duration-300 ${floodRisk ? "bg-amber-500" : "bg-slate-700"}`}
                    >
                      <span className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-all duration-300 ${floodRisk ? "left-5" : "left-0.5"}`} />
                    </button>
                  </div>
                  {floodRisk && (
                    <div className="text-[11px] text-amber-200 bg-amber-500/10 border border-amber-500/20 rounded-xl p-2.5 flex items-center gap-2">
                      <AlertTriangle size={12} className="shrink-0" />
                      30m riparian setback applies — PLUPA 2019 §56
                    </div>
                  )}
                  {slopeDeg > 15 && (
                    <div className="text-[11px] text-orange-200 bg-orange-500/10 border border-orange-500/20 rounded-xl p-2.5 flex items-center gap-2">
                      <Mountain size={12} className="shrink-0" />
                      Geotechnical cert required (NBRB) — slope &gt;15°
                    </div>
                  )}
                </div>
              </div>

              {/* Chatbot */}
              <div className="flex-1 flex flex-col rounded-2xl bg-slate-900/60 border border-slate-800 overflow-hidden shadow-2xl min-h-[400px] lg:min-h-0">

                {/* Chat header */}
                <div className="p-3 sm:p-4 border-b border-slate-800 bg-slate-900/80 flex items-center justify-between gap-2 shrink-0">
                  <div className="flex items-center gap-2 sm:gap-3 min-w-0">
                    <div className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center shrink-0">
                      <Bot className="w-4 h-4 text-cyan-400" />
                    </div>
                    <div className="min-w-0">
                      <h3 className="font-bold text-sm text-slate-100 flex flex-wrap items-center gap-1.5">
                        Spatial Regulatory Copilot
                        <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 font-mono">GROQ Llama 3.3 70B</span>
                      </h3>
                      <p className="text-[11px] text-slate-400 truncate hidden sm:block">PLUPA 2019 · NEMA EMCA · African Trade Corridors</p>
                    </div>
                  </div>
                  <button
                    onClick={() => handleSendMessage("Generate Complete Underwriting Dossier")}
                    className="text-xs px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-cyan-300 border border-slate-700 flex items-center gap-1 transition-colors shrink-0"
                  >
                    <Sparkles className="w-3 h-3 text-cyan-400" />
                    <span className="hidden sm:inline">Full Dossier</span>
                  </button>
                </div>

                {/* Quick prompt chips */}
                <div className="px-3 py-2 border-b border-slate-800/60 bg-slate-950/40 flex items-center gap-1.5 overflow-x-auto text-[11px] shrink-0">
                  <span className="text-slate-500 shrink-0 hidden sm:inline">Ask:</span>
                  {[
                    { emoji: "🌊", label: "PLUPA §56", q: "What are the riparian setbacks under PLUPA 2019?" },
                    { emoji: "🏗️", label: "Housing",   q: "Can we build an affordable housing project on this site?" },
                    { emoji: "🌿", label: "NEMA EIA",  q: "What are the NEMA EMCA environmental audit requirements?" },
                    { emoji: "🎯", label: "Rationale", q: `Explain why this was classified as ${selectedPoint.action.toUpperCase()}` },
                  ].map(c => (
                    <button key={c.label} onClick={() => handleSendMessage(c.q)}
                      className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700/80 shrink-0 whitespace-nowrap transition-colors">
                      {c.emoji} {c.label}
                    </button>
                  ))}
                </div>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-3 sm:p-5 space-y-3 sm:space-y-4">
                  {chatMessages.map(msg => (
                    <div key={msg.id} className={`flex gap-2 sm:gap-3 ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                      {msg.role === "assistant" && (
                        <div className="w-7 h-7 rounded-lg bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center shrink-0 mt-0.5">
                          <Bot className="w-4 h-4 text-cyan-300" />
                        </div>
                      )}
                      <div className={`max-w-[88%] rounded-2xl p-3 sm:p-4 text-xs leading-relaxed space-y-2 ${
                        msg.role === "user"
                          ? "bg-cyan-500 text-slate-950 font-medium shadow-md shadow-cyan-500/20"
                          : "bg-slate-800/80 border border-slate-700 text-slate-200"
                      }`}>
                        {msg.role === "assistant" ? <MarkdownMessage content={msg.content} /> : <div>{msg.content}</div>}
                        <div className={`text-[10px] text-right font-mono ${msg.role === "user" ? "text-slate-900" : "text-slate-500"}`}>{msg.timestamp}</div>
                      </div>
                      {msg.role === "user" && (
                        <div className="w-7 h-7 rounded-lg bg-slate-700 border border-slate-600 flex items-center justify-center shrink-0 mt-0.5">
                          <User className="w-4 h-4 text-slate-300" />
                        </div>
                      )}
                    </div>
                  ))}
                  {chatLoading && (
                    <div className="flex items-center gap-3">
                      <div className="w-7 h-7 rounded-lg bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center shrink-0">
                        <Bot className="w-4 h-4 text-cyan-300" />
                      </div>
                      <div className="p-3 rounded-xl bg-slate-800/80 border border-slate-700 text-xs text-slate-400 flex items-center gap-2">
                        <RefreshCw className="w-3.5 h-3.5 animate-spin text-cyan-400" />
                        Querying statutory knowledge base...
                      </div>
                    </div>
                  )}
                  <div ref={chatBottomRef} />
                </div>

                {/* Chat input */}
                <form onSubmit={e => { e.preventDefault(); handleSendMessage(); }}
                  className="p-3 border-t border-slate-800 bg-slate-900/80 flex items-center gap-2 shrink-0">
                  <input
                    type="text" value={inputQuery} onChange={e => setInputQuery(e.target.value)}
                    placeholder="Ask about statutory zoning, EIA, PLUPA, land conversion..."
                    className="flex-1 px-3 sm:px-4 py-2.5 rounded-xl bg-slate-900 border border-slate-700 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-cyan-400 transition-colors"
                  />
                  <button type="submit" disabled={chatLoading || !inputQuery.trim()}
                    className="px-3 sm:px-4 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 disabled:opacity-40 text-slate-950 font-bold text-xs flex items-center gap-1 transition-all cursor-pointer">
                    <Send className="w-3.5 h-3.5" />
                    <span className="hidden sm:inline">Send</span>
                  </button>
                </form>
              </div>
            </div>
          </div>
        )}

        {/* ── TAB 3: COUNTY DASHBOARD ──────────────────────────────────────────── */}
        {activeTab === "county" && (
          <div className="tab-enter w-full h-full overflow-y-auto">
            <div className="max-w-6xl mx-auto px-3 sm:px-6 py-5 sm:py-7 space-y-6 sm:space-y-8">

              {/* Page header */}
              <div className="flex flex-col sm:flex-row sm:items-end gap-3 sm:gap-6 pb-4 border-b border-slate-800">
                <div className="flex-1">
                  <h2 className="text-xl sm:text-2xl font-black gradient-text-cyan mb-1">County Benchmarks</h2>
                  <p className="text-xs text-slate-400">13 study counties · Leave-One-Zone-Out (LOZO) cross-validation · Sentinel-2 ground-truth</p>
                </div>
                {/* Zone legend pills */}
                <div className="flex flex-wrap gap-2">
                  {Object.entries(ZONE_META).map(([k, z]) => (
                    <span key={k} className="text-[10px] px-2 py-1 rounded-full border font-semibold flex items-center gap-1"
                      style={{ borderColor: z.color + "50", backgroundColor: z.color + "12", color: z.color }}>
                      {z.emoji} {z.label}
                    </span>
                  ))}
                </div>
              </div>

              {/* County card grid */}
              <div>
                <div className="text-[11px] uppercase font-mono text-slate-400 mb-3 tracking-wider">Select a county to inspect →</div>
                <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-7 gap-2">
                  {COUNTIES.map(c => {
                    const disp = COUNTY_DISPLAY[c] || { name: c, flag: "🇰🇪", zone: "rift_zone" };
                    const zCol = ZONE_COLORS[disp.zone] || "#06b6d4";
                    const data = REAL_COUNTY_SUMMARIES[c];
                    const rec  = data?.recommendation;
                    const am   = rec ? ACTION_META[rec] : null;
                    const isActive = selectedCounty === c;
                    return (
                      <button key={c} onClick={() => setSelectedCounty(c)}
                        className={`county-card p-2.5 rounded-xl border text-center space-y-1.5 ${
                          isActive
                            ? "border-cyan-400/60 bg-cyan-500/10 shadow-lg shadow-cyan-500/10"
                            : "border-slate-800 bg-slate-900/50 hover:border-slate-600"
                        }`}
                        style={isActive ? { borderColor: zCol + "80", boxShadow: `0 0 20px ${zCol}18` } : {}}
                      >
                        <div className="text-xl">{disp.flag}</div>
                        <div className={`text-[10px] font-bold leading-tight ${isActive ? "text-slate-100" : "text-slate-400"}`}>
                          {disp.name}
                        </div>
                        {am && (
                          <span className={`text-[9px] px-1.5 py-0.5 rounded-full font-bold ${am.bg} ${am.text} border ${am.border}`}>
                            {am.badge}
                          </span>
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Selected county detail */}
              {countyData && (() => {
                const disp = COUNTY_DISPLAY[selectedCounty] || { name: selectedCounty, flag: "🇰🇪", zone: "rift_zone" };
                const zCol = ZONE_COLORS[disp.zone] || "#06b6d4";
                const zm   = ZONE_META[countyData.climate_zone] || {};
                const rec  = countyData.recommendation;
                const am   = rec ? ACTION_META[rec] : null;
                return (
                  <div className="space-y-5">

                    {/* County hero banner */}
                    <div className="rounded-2xl border p-5 sm:p-6 relative overflow-hidden"
                      style={{ borderColor: zCol + "40", background: `linear-gradient(135deg, ${zCol}0a, rgba(15,23,42,0.8))` }}>
                      <div className="absolute top-0 left-0 right-0 h-1" style={{ background: `linear-gradient(90deg, ${zCol}, transparent)` }} />
                      <div className="flex flex-col sm:flex-row sm:items-center gap-4 sm:gap-6">
                        <div className="text-5xl sm:text-6xl">{disp.flag}</div>
                        <div className="flex-1 space-y-1">
                          <div className="text-[10px] uppercase font-mono tracking-wider" style={{ color: zCol }}>
                            {zm.emoji} {zm.label || countyData.climate_zone}
                          </div>
                          <h3 className="text-2xl sm:text-3xl font-black text-slate-100 capitalize">{disp.name}</h3>
                          <div className="flex flex-wrap items-center gap-2 mt-2">
                            <span className="text-xs text-slate-400 font-mono">{countyData.n_samples.toLocaleString()} ground-truth samples</span>
                            <span className="text-xs font-mono px-2 py-0.5 rounded-full bg-slate-800 border border-slate-700" style={{ color: zCol }}>
                              F1: {(countyData.zone_confidence).toFixed(3)}
                            </span>
                            {am && (
                              <span className={`text-xs font-bold px-2.5 py-0.5 rounded-full border ${am.bg} ${am.text} ${am.border} flex items-center gap-1`}>
                                <ActionIcon action={rec} size={11} /> {am.label}
                              </span>
                            )}
                          </div>
                        </div>

                        {/* Score rings */}
                        <div className="flex gap-4 sm:gap-6 justify-center sm:justify-end">
                          <ScoreRing value={countyData.scores?.settle  || 0} color="#3b82f6" label="Settle" />
                          <ScoreRing value={countyData.scores?.farm    || 0} color="#10b981" label="Farm" />
                          <ScoreRing value={countyData.scores?.protect || 0} color="#f59e0b" label="Protect" />
                        </div>
                      </div>
                    </div>

                    {/* Land-cover + per-class F1 */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

                      {/* Panel 1: Model's predicted cover distribution */}
                      <div className="glass-card p-4 space-y-3">
                        <div className="text-[10px] uppercase font-mono text-slate-400 tracking-wider flex items-center gap-2">
                          <Cpu size={11} className="text-cyan-400" /> Model's Land-Cover Read
                        </div>
                        <p className="text-[10px] text-slate-500 leading-relaxed">
                          Distribution of <strong className="text-slate-400">predicted</strong> land-cover classes across {countyData.n_samples.toLocaleString()} sampled parcels in this county.
                        </p>
                        <div className="space-y-2.5">
                          {CLASSES.map((cls, i) => {
                            const val = countyData.predicted_cover_pct?.[cls] || 0;
                            const meta = CLASS_META[cls];
                            return (
                              <div key={cls}>
                                <div className="flex justify-between text-[11px] mb-1">
                                  <span className="flex items-center gap-1.5 text-slate-300">
                                    <span className="w-2 h-2 rounded-sm" style={{ backgroundColor: meta.color }} />
                                    {meta.label}
                                  </span>
                                  <span className="font-mono text-slate-400 font-semibold">{val.toFixed(1)}%</span>
                                </div>
                                <AnimBar pct={val} color={meta.color} delay={i * 70} />
                              </div>
                            );
                          })}
                        </div>
                      </div>

                      {/* Panel 2: Per-class F1 from LOZO validation */}
                      {(() => {
                        const zf = ZONE_MODEL_FINDINGS[countyData.climate_zone];
                        if (!zf?.per_class_f1) return null;
                        return (
                          <div className="glass-card p-4 space-y-3">
                            <div className="text-[10px] uppercase font-mono text-slate-400 tracking-wider flex items-center gap-2">
                              <Award size={11} className="text-cyan-400" /> LOZO Per-Class F1 Accuracy
                            </div>
                            <p className="text-[10px] text-slate-500 leading-relaxed">
                              How well the model identifies each class on <strong className="text-slate-400">{zf.val_samples.toLocaleString()} held-out</strong> parcels (Leave-One-Zone-Out test).
                              <span className="text-emerald-400"> Green ≥0.6</span>,
                              <span className="text-amber-400"> amber 0.3–0.6</span>,
                              <span className="text-red-400"> red &lt;0.3</span>.
                            </p>
                            <div className="space-y-2.5">
                              {CLASSES.map((cls, i) => {
                                const f1  = zf.per_class_f1[cls] ?? 0;
                                const meta = CLASS_META[cls];
                                const barColor = f1 >= 0.6 ? "#10b981" : f1 >= 0.3 ? "#f59e0b" : "#ef4444";
                                const textColor = f1 >= 0.6 ? "text-emerald-400" : f1 >= 0.3 ? "text-amber-400" : "text-red-400";
                                return (
                                  <div key={cls}>
                                    <div className="flex justify-between text-[11px] mb-1">
                                      <span className="flex items-center gap-1.5 text-slate-300">
                                        <span className="w-2 h-2 rounded-sm" style={{ backgroundColor: meta.color }} />
                                        {meta.label}
                                      </span>
                                      <span className={`font-mono font-bold ${textColor}`}>F1 {f1.toFixed(3)}</span>
                                    </div>
                                    <AnimBar pct={f1 * 100} color={barColor} delay={200 + i * 70} />
                                  </div>
                                );
                              })}
                            </div>
                            <div className="text-[10px] text-slate-500 pt-1 border-t border-slate-800 flex items-center gap-1.5">
                              <Info size={10} />
                              Macro-F1 {zf.macro_f1} · Accuracy {zf.accuracy} · {zf.train_samples.toLocaleString()} training samples
                            </div>
                          </div>
                        );
                      })()}
                    </div>
                  </div>
                );
              })()}

              {/* Zone policy accordion */}
              <div className="space-y-3">
                <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2">
                  <FileText className="w-4 h-4 text-cyan-400" />
                  Agro-Ecological Zone Policy Findings
                </h3>
                <div className="space-y-2">
                  {Object.entries(ZONE_MODEL_FINDINGS).map(([zk, f]) => {
                    const zCol = ZONE_COLORS[zk] || "#06b6d4";
                    const zm   = ZONE_META[zk] || {};
                    const isOpen = expandedZone === zk;
                    const isActive = countyData?.climate_zone === zk;
                    return (
                      <div key={zk} className={`rounded-xl border overflow-hidden transition-all duration-200 ${isActive ? "ring-1" : ""}`}
                        style={{ borderColor: zCol + (isOpen ? "60" : "25"), ...(isActive ? { ringColor: zCol } : {}) }}>
                        <button
                          onClick={() => setExpandedZone(isOpen ? null : zk)}
                          className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-slate-800/50 transition-colors"
                        >
                          <div className="flex items-center gap-3">
                            <span className="text-lg">{zm.emoji}</span>
                            <div>
                              <div className="text-xs font-bold text-slate-200">{zk.replace("_", " ").toUpperCase()}</div>
                              <div className="text-[10px] text-slate-400">Macro-F1: {f.macro_f1} · {f.accuracy}</div>
                            </div>
                            {isActive && <span className="text-[9px] px-2 py-0.5 rounded-full font-bold text-white" style={{ backgroundColor: zCol + "80" }}>Active Zone</span>}
                          </div>
                          <div className="flex items-center gap-3">
                            {/* Mini F1 bar */}
                            <div className="hidden sm:flex items-center gap-2 w-24">
                              <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                <div className="h-full rounded-full" style={{ width: `${f.macro_f1 * 100}%`, backgroundColor: zCol }} />
                              </div>
                              <span className="text-[10px] font-mono text-slate-400">{f.macro_f1}</span>
                            </div>
                            {isOpen ? <ChevronUp size={14} className="text-slate-400" /> : <ChevronDown size={14} className="text-slate-400" />}
                          </div>
                        </button>
                        {isOpen && (
                          <div className="px-4 pb-4 pt-1 space-y-2 border-t" style={{ borderColor: zCol + "25" }}>
                            <p className="text-xs text-slate-300 leading-relaxed">{f.key_insight}</p>
                            <div className="text-[11px] text-slate-400 p-3 rounded-xl border" style={{ backgroundColor: zCol + "08", borderColor: zCol + "30" }}>
                              <span className="font-bold" style={{ color: zCol }}>Statutory Implication: </span>
                              {f.policy_implication}
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── TAB 4: MLOPS & SYSTEM TELEMETRY ─────────────────────────────────── */}
        {activeTab === "mlops" && (
          <div className="tab-enter w-full h-full overflow-y-auto">
            <div className="max-w-6xl mx-auto px-3 sm:px-6 py-5 sm:py-7 space-y-6 sm:space-y-8">

              {/* Page header */}
              <div className="pb-4 border-b border-slate-800">
                <h2 className="text-xl sm:text-2xl font-black gradient-text-cyan mb-1">MLOps & System Telemetry</h2>
                <p className="text-xs text-slate-400">Live pipeline health · LOZO benchmarks · Infrastructure links · Deployment cheatsheet</p>
              </div>

              {/* Live system status */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[
                  {
                    label: "API Backend", sub: "FastAPI / Vultr London",
                    icon: <Server size={18} />, color: apiOnline ? "#10b981" : "#64748b",
                    value: apiOnline ? `${apiPing}ms` : "Offline",
                    badge: apiOnline ? "LIVE" : "DOWN",
                    badgeColor: apiOnline ? "text-emerald-400 bg-emerald-500/10 border-emerald-500/30" : "text-slate-400 bg-slate-800 border-slate-700",
                  },
                  {
                    label: "GROQ LLM", sub: "Llama 3.3 70B Turbo",
                    icon: <Sparkles size={18} />, color: "#a855f7",
                    value: "~380ms", badge: "RAG",
                    badgeColor: "text-purple-400 bg-purple-500/10 border-purple-500/30",
                  },
                  {
                    label: "Model Registry", sub: "MLflow · 5 LOZO zones",
                    icon: <Database size={18} />, color: "#06b6d4",
                    value: "5 models", badge: "LGBM",
                    badgeColor: "text-cyan-400 bg-cyan-500/10 border-cyan-500/30",
                  },
                  {
                    label: "Sentinel-2 Data", sub: "390 calibrated parcels",
                    icon: <Globe size={18} />, color: "#22c55e",
                    value: "390 pts", badge: "EO",
                    badgeColor: "text-emerald-400 bg-emerald-500/10 border-emerald-500/30",
                  },
                ].map((s, i) => (
                  <div key={i} className="glass-card p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="w-9 h-9 rounded-xl flex items-center justify-center" style={{ backgroundColor: s.color + "18", border: `1px solid ${s.color}30`, color: s.color }}>
                        {s.icon}
                      </div>
                      <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded-full border ${s.badgeColor}`}>{s.badge}</span>
                    </div>
                    <div>
                      <div className="text-sm font-bold text-slate-100">{s.label}</div>
                      <div className="text-[11px] text-slate-400">{s.sub}</div>
                    </div>
                    <div className="text-lg font-black font-mono" style={{ color: s.color }}>{s.value}</div>
                  </div>
                ))}
              </div>

              {/* Live ping sparkline */}
              {pingHistory.length > 1 && (
                <div className="glass-card p-4 space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="text-[10px] uppercase font-mono text-slate-400 tracking-wider flex items-center gap-2">
                      <Activity size={12} className="text-emerald-400" /> Live API Latency
                    </div>
                    <span className="text-[11px] font-mono text-emerald-400">{apiPing}ms latest</span>
                  </div>
                  <div className="h-10 flex items-end gap-1">
                    {pingHistory.map((p, i) => {
                      const max = Math.max(...pingHistory, 100);
                      const h = Math.max((p / max) * 100, 8);
                      const color = p < 100 ? "#10b981" : p < 300 ? "#f59e0b" : "#ef4444";
                      return (
                        <div key={i} className="flex-1 rounded-t-sm transition-all" style={{ height: `${h}%`, backgroundColor: color + "aa" }} title={`${p}ms`} />
                      );
                    })}
                  </div>
                  <div className="flex justify-between text-[9px] text-slate-500">
                    <span>12 pings ago</span><span>Now</span>
                  </div>
                </div>
              )}

              {/* ML pipeline flow */}
              <div className="glass-card p-5 space-y-4">
                <h3 className="font-bold text-sm text-slate-100 flex items-center gap-2">
                  <Layers3 size={15} className="text-cyan-400" />
                  End-to-End Spatial ML Pipeline
                </h3>
                <div className="flex flex-col sm:flex-row gap-2 sm:gap-0 overflow-x-auto pb-2">
                  {[
                    { step: "01", icon: <Globe size={16} />, title: "Earth Observation", desc: "Sentinel-2 L2A · 10 bands · SRTM DEM · CHIRPS", color: "#06b6d4" },
                    { step: "02", icon: <Layers3 size={16} />, title: "27 Spectral Indices", desc: "NDVI · EVI · MNDWI · NDBI · BSI · TWI · slope", color: "#22c55e" },
                    { step: "03", icon: <Cpu size={16} />, title: "Zone-Stratified LGBM", desc: "5 agro-ecological models + EfficientNet-B0 ONNX", color: "#a855f7" },
                    { step: "04", icon: <Bot size={16} />, title: "Statutory RAG", desc: "GROQ Llama 3.3 · PLUPA 2019 · NEMA EMCA", color: "#f59e0b" },
                    { step: "05", icon: <Server size={16} />, title: "Cloud Serving", desc: "Vultr Ubuntu 24.04 · Docker · FastAPI :8000", color: "#3b82f6" },
                  ].map((s, i, arr) => (
                    <div key={i} className="flex sm:flex-col items-center sm:items-stretch gap-3 sm:gap-0 flex-1">
                      <div className="sm:flex-1 p-3 rounded-xl border space-y-2 min-w-0"
                        style={{ borderColor: s.color + "35", background: s.color + "08" }}>
                        <div className="flex items-center gap-2">
                          <div className="w-7 h-7 rounded-lg flex items-center justify-center shrink-0"
                            style={{ backgroundColor: s.color + "20", color: s.color }}>
                            {s.icon}
                          </div>
                          <span className="text-[9px] font-mono font-bold" style={{ color: s.color }}>{s.step}</span>
                        </div>
                        <div className="text-xs font-bold text-slate-100">{s.title}</div>
                        <p className="text-[10px] text-slate-400 leading-relaxed">{s.desc}</p>
                      </div>
                      {i < arr.length - 1 && (
                        <div className="flex sm:justify-center items-center sm:py-1 shrink-0">
                          <ArrowRight size={14} className="text-slate-600 rotate-90 sm:rotate-0" />
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* LOZO benchmark table */}
              <div className="glass-card overflow-hidden">
                <div className="px-5 py-4 border-b border-slate-800 flex items-center gap-3">
                  <Award size={16} className="text-cyan-400" />
                  <div>
                    <div className="font-bold text-sm text-slate-100">LOZO Cross-Validation Results</div>
                    <div className="text-[10px] text-slate-400">Leave-One-Zone-Out · Zone-Stratified LightGBM vs. Monolithic Baseline</div>
                  </div>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full lozo-table">
                    <thead>
                      <tr>
                        <th>Zone</th>
                        <th>Macro F1</th>
                        <th>Baseline F1</th>
                        <th>Improvement</th>
                        <th className="hidden sm:table-cell">Parcels</th>
                        <th className="hidden sm:table-cell">Top Class</th>
                      </tr>
                    </thead>
                    <tbody>
                      {LOZO_BENCHMARKS.map((row, i) => {
                        const delta = (row.macro_f1 - row.baseline);
                        const pct   = Math.round(delta / row.baseline * 100);
                        const f1Pct = row.macro_f1 * 100;
                        return (
                          <tr key={i}>
                            <td className="font-semibold text-slate-200">{row.zone}</td>
                            <td>
                              <div className="flex items-center gap-2">
                                <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                  <div className="h-full rounded-full bg-cyan-400" style={{ width: `${f1Pct}%` }} />
                                </div>
                                <span className="font-mono text-cyan-300 font-bold">{row.macro_f1.toFixed(3)}</span>
                              </div>
                            </td>
                            <td><span className="font-mono text-slate-400">{row.baseline.toFixed(3)}</span></td>
                            <td>
                              <span className="text-[11px] font-bold font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20">
                                +{pct}%
                              </span>
                            </td>
                            <td className="hidden sm:table-cell font-mono text-slate-400">{row.parcels.toLocaleString()}</td>
                            <td className="hidden sm:table-cell text-slate-300">{row.top_class}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Quick service links */}
              <div>
                <h3 className="font-bold text-sm text-slate-200 mb-3 flex items-center gap-2">
                  <ExternalLink size={14} className="text-cyan-400" /> Service Links
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {[
                    { href: "http://192.248.170.201:8000/docs", icon: <Server size={16} />, label: "FastAPI Swagger", sub: ":8000/docs", color: "#06b6d4" },
                    { href: "http://192.248.170.201:8000/health", icon: <Activity size={16} />, label: "Health Check", sub: ":8000/health", color: "#10b981" },
                    { href: "http://192.248.170.201:5000", icon: <Database size={16} />, label: "MLflow UI", sub: ":5000", color: "#a855f7" },
                    { href: "https://github.com/VAL-Jerono/Satellite", icon: <FileText size={16} />, label: "GitHub Repo", sub: "VAL-Jerono/Satellite", color: "#3b82f6" },
                  ].map((s, i) => (
                    <a key={i} href={s.href} target="_blank" rel="noreferrer"
                      className="glass-card p-4 hover:border-cyan-500/30 transition-all group space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ backgroundColor: s.color + "18", border: `1px solid ${s.color}30`, color: s.color }}>
                          {s.icon}
                        </div>
                        <ExternalLink size={12} className="text-slate-500 group-hover:text-slate-300 transition-colors" />
                      </div>
                      <div>
                        <div className="text-xs font-bold text-slate-100">{s.label}</div>
                        <div className="text-[10px] font-mono mt-0.5" style={{ color: s.color }}>{s.sub}</div>
                      </div>
                    </a>
                  ))}
                </div>
              </div>

              {/* Developer curl cheatsheet */}
              <div className="glass-card p-5 space-y-3">
                <h3 className="font-bold text-sm text-slate-100 flex items-center gap-2">
                  <Zap size={14} className="text-cyan-400" /> Quick API Test Commands
                </h3>
                <div className="space-y-2">
                  {[
                    { label: "Health check", cmd: "curl http://192.248.170.201/health" },
                    { label: "Predict parcel", cmd: 'curl -X POST http://192.248.170.201/predict -H "Content-Type: application/json" -d \'{"lat":-0.4201,"lon":36.9476}\'' },
                    { label: "Underwrite copilot", cmd: 'curl -X POST http://192.248.170.201/underwrite/copilot -H "Content-Type: application/json" -d \'{"user_query":"Can we build here?"}\'' },
                  ].map((c, i) => (
                    <div key={i}>
                      <div className="text-[10px] text-slate-400 mb-1 font-mono"># {c.label}</div>
                      <div className="relative group">
                        <code className="block text-[11px] text-cyan-300 bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 font-mono overflow-x-auto whitespace-nowrap">
                          {c.cmd}
                        </code>
                        <button
                          onClick={() => navigator.clipboard.writeText(c.cmd)}
                          className="absolute top-1.5 right-2 opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded bg-slate-700 hover:bg-slate-600"
                          title="Copy"
                        >
                          <Copy size={11} className="text-slate-300" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

            </div>
          </div>
        )}
      </div>

      {/* ════════════════════════════════════════════════════════════════════════
          SETTINGS MODAL
      ════════════════════════════════════════════════════════════════════════ */}
      {showSettings && (
        <div className="fixed inset-0 z-[2000] bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 w-full max-w-md space-y-4 shadow-2xl">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <Settings className="w-4 h-4 text-cyan-400" /> Configure Backend API
              </h3>
              <button onClick={() => setShowSettings(false)} className="p-1 rounded-lg hover:bg-slate-800 text-slate-400">
                <X size={16} />
              </button>
            </div>
            <p className="text-xs text-slate-400">Set the URL of your Vultr server or local API container.</p>
            <input
              type="text" value={apiUrl} onChange={e => setApiUrl(e.target.value)}
              placeholder="http://192.248.170.201:8000"
              className="w-full p-2.5 rounded-xl bg-slate-800 border border-slate-700 text-slate-100 font-mono text-xs focus:outline-none focus:border-cyan-400"
            />
            <div className="flex justify-end gap-2">
              <button onClick={() => setShowSettings(false)}
                className="px-4 py-2 rounded-xl text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-300">Cancel</button>
              <button onClick={() => { localStorage.setItem("atlas_api_url", apiUrl); setShowSettings(false); checkHealth(); }}
                className="px-4 py-2 rounded-xl text-xs font-bold bg-cyan-500 hover:bg-cyan-400 text-slate-950">Save & Connect</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
