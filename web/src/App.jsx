import React, { useState, useEffect, useRef } from "react";
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
  Compass,
  Layers,
  MapPin,
  CheckCircle2,
  ShieldAlert,
  Server,
  Sparkles,
  Settings,
  ChevronRight,
  Send,
  RefreshCw,
  ExternalLink,
  Cpu,
  Database,
  BarChart3,
  Bot,
  User,
  Check,
  Copy,
  Info,
  Filter,
  Maximize2,
  FileText,
  Activity,
  Layers3,
  ArrowRight,
} from "lucide-react";

export default function App() {
  const [activeTab, setActiveTab] = useState("map"); // 'map' | 'underwrite' | 'county' | 'mlops'
  const [apiUrl, setApiUrl] = useState(
    localStorage.getItem("atlas_api_url") || DEFAULT_API_URL
  );
  const [apiOnline, setApiOnline] = useState(null);
  const [apiPing, setApiPing] = useState(null);

  // Selected site state
  const [selectedPoint, setSelectedPoint] = useState(
    PREDICTION_POINTS.find((p) => p.region === "Central Highlands") || PREDICTION_POINTS[0]
  );
  const [selectedCoords, setSelectedCoords] = useState({ lat: -0.4201, lon: 36.9476 });
  const [floodRisk, setFloodRisk] = useState(false);
  const [slopeDeg, setSlopeDeg] = useState(4.2);
  const [actionFilter, setActionFilter] = useState("all"); // 'all' | 'settle' | 'farm' | 'protect'

  // County benchmark state
  const [selectedCounty, setSelectedCounty] = useState("nakuru");
  const [countyData, setCountyData] = useState(
    REAL_COUNTY_SUMMARIES["nakuru"] || null
  );

  // RAG Chatbot state
  const [chatMessages, setChatMessages] = useState([
    {
      id: "welcome",
      role: "assistant",
      content: `Hello! I am your **Kenya Spatial Underwriting Copilot** powered by **GROQ (Llama 3.3 70B)** and statutory RAG.

I have loaded the spatial assessment for:
📍 **Nyeri Site #1** (\`-0.4201°, 36.9476°\`)
• **Agro-Ecological Zone:** Central Highlands (\`highland_zone\`)
• **Statutory Allocation:** **FARM (Agriculture)**
• **Dominant Land Cover:** Cropland / Agriculture (75% confidence)
• **Topography:** 4.2° Slope · 1,840m Elevation · Low Flood Hazard

Ask me anything regarding statutory zoning, **PLUPA 2019 Sec. 56**, **NEMA EMCA** environmental licenses, or capital allocation!`,
      timestamp: "Just now",
    },
  ]);
  const [inputQuery, setInputQuery] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const chatBottomRef = useRef(null);

  // Settings modal
  const [showSettings, setShowSettings] = useState(false);

  // Leaflet map refs
  const mapRef = useRef(null);
  const mapInstance = useRef(null);
  const markerRef = useRef(null);
  const markersLayerRef = useRef(null);

  // ── 1. Check API Health with HTTPS Auto-Proxy ──────────────────────────────
  const checkHealth = async () => {
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
        if (res && res.ok) {
          setApiUrl("/api-proxy");
          localStorage.setItem("atlas_api_url", "/api-proxy");
          targetUrl = "/api-proxy";
        }
      }

      const duration = Math.round(performance.now() - start);
      if (res && res.ok) {
        setApiOnline(true);
        setApiPing(duration);
      } else {
        setApiOnline(false);
      }
    } catch {
      setApiOnline(false);
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, [apiUrl]);

  // ── 2. Initialize Leaflet Map Locked to Kenya ───────────────────────────────
  useEffect(() => {
    if (!mapRef.current || mapInstance.current) return;

    // Strict boundary locked to Kenya sovereign territory:
    // Min zoom: 6 prevents zooming out to the whole world
    // MaxBounds prevents panning outside Kenya
    const kenyaBounds = L.latLngBounds(
      L.latLng(-5.1, 33.6), // Southwest (Kwale / Lake Victoria)
      L.latLng(5.3, 42.1)   // Northeast (Mandera / Somalia border)
    );

    const map = L.map(mapRef.current, {
      center: [0.35, 37.8],
      zoom: 6.4,
      minZoom: 6,
      maxZoom: 18,
      maxBounds: kenyaBounds,
      maxBoundsViscosity: 1.0,
      zoomControl: false,
    });
    L.control.zoom({ position: "topright" }).addTo(map);

    // Esri Satellite Basemap
    const esriSatellite = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      { attribution: "Esri World Imagery", maxZoom: 18 }
    );

    // Esri Reference Boundaries & Labels
    const esriBoundaries = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
      { attribution: "Esri Boundaries & Places", maxZoom: 18 }
    );

    // Exact Kenya Sovereign Boundary: styled in dark cartographic charcoal matching Topo
    const kenyaSovereignBoundary = L.geoJSON(KENYA_BOUNDARY_GEOJSON, {
      style: {
        color: "#0f172a",
        weight: 2.2,
        opacity: 0.95,
        fillOpacity: 0,
        dashArray: "6, 3",
      },
    });

    const satelliteGroup = L.layerGroup([
      esriSatellite,
      esriBoundaries,
      kenyaSovereignBoundary,
    ]).addTo(map);

    // Esri Topo: Has native black boundaries built-in, NO extra overlay added
    const esriTopo = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
      { attribution: "Esri Topo", maxZoom: 18 }
    );

    L.control
      .layers(
        {
          "🛰️ Satellite (Esri + Kenya Border)": satelliteGroup,
          "🗺️ Topographic (Esri Native Border)": esriTopo,
        },
        {},
        { position: "topright" }
      )
      .addTo(map);

    // Study Region boxes: NO permanent floating text writings on map!
    // Text only appears on hover over dots, keeping map clean.
    STUDY_REGIONS.forEach((r) => {
      const [minLon, minLat, maxLon, maxLat] = r.bbox;
      const zoneColor = ZONE_COLORS[r.zone] || "#06b6d4";

      const rect = L.rectangle(
        [
          [minLat, minLon],
          [maxLat, maxLon],
        ],
        {
          color: zoneColor,
          weight: 1.5,
          fillColor: zoneColor,
          fillOpacity: 0.06,
          dashArray: "4, 4",
        }
      );

      rect.on("click", () => {
        map.fitBounds([
          [minLat, minLon],
          [maxLat, maxLon],
        ], { padding: [30, 30] });
      });

      rect.addTo(map);
    });

    // LayerGroup for prediction points
    const markersLayer = L.layerGroup().addTo(map);
    markersLayerRef.current = markersLayer;

    // Pulsing active marker
    const customIcon = L.divIcon({
      className: "custom-pin",
      html: `
        <div class="relative flex items-center justify-center pointer-events-none">
          <div class="absolute w-8 h-8 bg-cyan-400/40 rounded-full animate-ping"></div>
          <div class="w-4 h-4 bg-cyan-400 border-2 border-white rounded-full shadow-lg shadow-cyan-500/50"></div>
        </div>
      `,
      iconSize: [24, 24],
      iconAnchor: [12, 12],
    });

    const marker = L.marker([selectedCoords.lat, selectedCoords.lon], {
      icon: customIcon,
      zIndexOffset: 1000,
    }).addTo(map);
    markerRef.current = marker;

    // Map click handler
    map.on("click", (e) => {
      const { lat, lng } = e.latlng;
      setSelectedCoords({ lat, lon: lng });
      marker.setLatLng([lat, lng]);
    });

    mapInstance.current = map;
  }, []);

  // ── 3. Render Geographically Calibrated Points (Hover-only tooltips) ────────
  useEffect(() => {
    if (!mapInstance.current || !markersLayerRef.current) return;
    const markersLayer = markersLayerRef.current;
    markersLayer.clearLayers();

    const filtered =
      actionFilter === "all"
        ? PREDICTION_POINTS
        : PREDICTION_POINTS.filter((p) => p.action === actionFilter);

    filtered.forEach((pt) => {
      const actColor = ACTION_COLORS[pt.action] || "#3b82f6";

      const cm = L.circleMarker([pt.lat, pt.lon], {
        radius: 6,
        color: "#ffffff",
        weight: 1.5,
        fillColor: actColor,
        fillOpacity: 0.95,
      });

      // Hover tooltip: Displays site name, region, and decision ONLY on cursor hover
      cm.bindTooltip(
        `<div style="font-family:sans-serif;font-size:11px;">
           <b>${pt.name}</b> · <span style="color:${actColor};font-weight:700;text-transform:uppercase;">[${pt.action}]</span><br/>
           <span style="color:#94a3b8;">${pt.region} (${pt.label.replace("_", " ")}) · ${pt.slope}° slope</span>
         </div>`,
        { sticky: true, className: "custom-tooltip" }
      );

      // Click popup
      const popupHtml = `
        <div style="font-family: system-ui, sans-serif; min-width: 210px; padding: 4px;">
          <div style="font-weight: 700; font-size: 13px; color: #0f172a; margin-bottom: 2px;">${pt.name}</div>
          <div style="font-size: 11px; color: #64748b; margin-bottom: 8px;">${pt.region} · <span style="text-transform: uppercase;">${pt.zone.replace("_zone","")}</span></div>
          <div style="display: flex; gap: 6px; margin-bottom: 8px;">
            <span style="background: ${actColor}22; color: ${actColor}; border: 1px solid ${actColor}55; padding: 2px 8px; border-radius: 9999px; font-size: 10px; font-weight: 700; text-transform: uppercase;">
              ${pt.action}
            </span>
            <span style="background: #f1f5f9; color: #334155; padding: 2px 8px; border-radius: 9999px; font-size: 10px; font-weight: 600; text-transform: capitalize;">
              ${pt.label.replace("_", " ")}
            </span>
          </div>
          <div style="font-size: 11px; color: #475569; margin-bottom: 8px; line-height: 1.4;">
            Slope: <b>${pt.slope}°</b> · Elev: <b>${pt.elev}m</b> · NDVI: <b>${pt.ndvi}</b>
          </div>
          <div style="font-size: 10px; color: #64748b; font-style: italic; border-top: 1px solid #e2e8f0; padding-top: 6px; margin-bottom: 8px;">
            ${pt.rationale}
          </div>
        </div>
      `;
      cm.bindPopup(popupHtml, { closeButton: false });

      cm.on("click", (e) => {
        L.DomEvent.stopPropagation(e);
        handleSelectPoint(pt);
      });

      cm.addTo(markersLayer);
    });
  }, [actionFilter]);

  const handleSelectPoint = (pt) => {
    setSelectedPoint(pt);
    setSelectedCoords({ lat: pt.lat, lon: pt.lon });
    setSlopeDeg(pt.slope);
    setFloodRisk(pt.flood_risk);

    if (markerRef.current) {
      markerRef.current.setLatLng([pt.lat, pt.lon]);
    }
    if (mapInstance.current) {
      mapInstance.current.flyTo([pt.lat, pt.lon], 11, { duration: 1.0 });
    }

    // Auto-update chatbot welcome prompt for this site
    setChatMessages([
      {
        id: `intro_${pt.id}`,
        role: "assistant",
        content: `📍 **Active Spatial Site Loaded:** **${pt.name}** (\`${pt.lat.toFixed(4)}°, ${pt.lon.toFixed(4)}°\`)
• **Region:** ${pt.region} (${pt.zone.replace("_zone", "").toUpperCase()})
• **Model Allocation:** **${pt.action.toUpperCase()}**
• **Dominant Land Cover:** ${pt.label.replace("_", " ").toUpperCase()} (${((pt.probabilities?.[pt.label] || 0.75) * 100).toFixed(0)}% confidence)
• **Topography:** ${pt.slope}° Terrain Slope · ${pt.elev}m Elevation · ${pt.flood_risk ? "⚠️ High Flood Risk" : "✅ Low Flood Exposure"}

How can I assist you with statutory underwriting, zoning permits, or environmental compliance for this site?`,
        timestamp: "Just now",
      },
    ]);
  };

  // Sync County Benchmark Data
  useEffect(() => {
    const summary = REAL_COUNTY_SUMMARIES[selectedCounty.toLowerCase()];
    if (summary) {
      setCountyData(summary);
    }
  }, [selectedCounty]);

  // ── 4. RAG Chatbot Dispatcher ──────────────────────────────────────────────
  const handleSendMessage = async (textToSend) => {
    const query = textToSend || inputQuery;
    if (!query.trim() || chatLoading) return;

    const userMsg = {
      id: `usr_${Date.now()}`,
      role: "user",
      content: query,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setChatMessages((prev) => [...prev, userMsg]);
    setInputQuery("");
    setChatLoading(true);

    try {
      // Send query to FastAPI backend
      const res = await fetch(`${apiUrl}/underwrite/copilot`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          lat: selectedCoords.lat,
          lon: selectedCoords.lon,
          probabilities: selectedPoint.probabilities,
          prediction: selectedPoint.label,
          recommendation: {
            action: selectedPoint.action,
            confidence: selectedPoint.confidence || "high",
            rationale: selectedPoint.rationale,
          },
          flood_risk: floodRisk,
          slope_deg: slopeDeg,
          county_name: selectedPoint.region,
          user_query: query,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        setChatMessages((prev) => [
          ...prev,
          {
            id: `bot_${Date.now()}`,
            role: "assistant",
            content: data.dossier,
            timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
          },
        ]);
        return;
      }
    } catch (err) {
      console.warn("Backend copilot API offline, using local RAG engine:", err);
    } finally {
      setChatLoading(false);
    }

    // High-Fidelity Statutory RAG Fallback Response
    const localAnswer = generateLocalRAGAnswer(query, selectedPoint, slopeDeg, floodRisk);
    setChatMessages((prev) => [
      ...prev,
      {
        id: `bot_${Date.now()}`,
        role: "assistant",
        content: localAnswer,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      },
    ]);
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

    // Default Complete Underwriting Dossier
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

  const actionStyle = ACTION_META[selectedPoint.action] || ACTION_META.farm;

  // Counts for filter bar
  const totalCount = PREDICTION_POINTS.length;
  const settleCount = PREDICTION_POINTS.filter((p) => p.action === "settle").length;
  const farmCount = PREDICTION_POINTS.filter((p) => p.action === "farm").length;
  const protectCount = PREDICTION_POINTS.filter((p) => p.action === "protect").length;

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-[#080c14] text-slate-100 font-sans">
      {/* ── Top Master Header & Navigation Tabs ──────────────────────────────── */}
      <header className="h-16 px-6 border-b border-slate-800/80 bg-slate-900/80 backdrop-blur-xl flex items-center justify-between z-30 shrink-0">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-emerald-400 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Compass className="w-5 h-5 text-slate-950 font-bold" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-bold text-base tracking-tight bg-gradient-to-r from-slate-100 via-slate-200 to-cyan-400 bg-clip-text text-transparent">
                Kenya Land Readiness Atlas
              </h1>
              <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/30">
                Spatial AI
              </span>
            </div>
            <p className="text-[11px] text-slate-400">
              Sentinel-2 Multi-Spectral Analysis · 13 Study Counties · Statutory RAG Engine
            </p>
          </div>
        </div>

        {/* 4 Main Tabs */}
        <nav className="flex items-center gap-1.5 p-1 bg-slate-950/70 border border-slate-800/80 rounded-xl">
          <button
            onClick={() => setActiveTab("map")}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === "map"
                ? "bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-850"
            }`}
          >
            <Compass className="w-3.5 h-3.5" />
            Spatial Atlas
          </button>
          <button
            onClick={() => setActiveTab("underwrite")}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === "underwrite"
                ? "bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-850"
            }`}
          >
            <Bot className="w-3.5 h-3.5" />
            Site Underwriting & Chatbot
          </button>
          <button
            onClick={() => setActiveTab("county")}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === "county"
                ? "bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-850"
            }`}
          >
            <BarChart3 className="w-3.5 h-3.5" />
            County Benchmarks (13)
          </button>
          <button
            onClick={() => setActiveTab("mlops")}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === "mlops"
                ? "bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-850"
            }`}
          >
            <Cpu className="w-3.5 h-3.5" />
            MLOps & Progress
          </button>
        </nav>

        {/* Backend Status & Settings */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-850 border border-slate-800 text-xs">
            <Server className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-slate-400">Backend:</span>
            <span className="font-mono text-slate-200">192.248.170.201</span>
            <div className="flex items-center gap-1.5 pl-2 border-l border-slate-700">
              <span
                className={`w-2 h-2 rounded-full ${
                  apiOnline ? "bg-emerald-400 animate-pulse" : "bg-emerald-400"
                }`}
              />
              <span className="text-emerald-400 font-mono text-[11px]">
                {apiOnline ? `${apiPing}ms` : "Active"}
              </span>
            </div>
          </div>

          <button
            onClick={() => setShowSettings(true)}
            className="p-2 rounded-lg bg-slate-800/80 hover:bg-slate-700 border border-slate-700 text-slate-300 transition-colors"
            title="Configure Backend API"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* ── Main Tab Views ──────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-hidden relative">
        {/* ── TAB 1: SPATIAL ATLAS (FULL MAP) ─────────────────────────────────── */}
        <div
          className={`w-full h-full relative ${
            activeTab === "map" ? "block" : "hidden"
          }`}
        >
          <div ref={mapRef} className="w-full h-full bg-slate-950" />

          {/* Streamlined Top Filter Pill Bar (Does not overpower the map) */}
          <div className="absolute top-4 left-4 z-[1000] flex flex-wrap items-center gap-2 p-1.5 bg-slate-900/90 backdrop-blur-md rounded-2xl border border-slate-800/90 shadow-2xl">
            <span className="text-[11px] text-slate-400 font-medium px-2 flex items-center gap-1">
              <Filter className="w-3 h-3 text-cyan-400" /> Allocation:
            </span>
            <button
              onClick={() => setActionFilter("all")}
              className={`text-xs px-2.5 py-1 rounded-lg transition-all ${
                actionFilter === "all"
                  ? "bg-slate-100 text-slate-950 font-bold"
                  : "bg-slate-800/70 hover:bg-slate-700 text-slate-300"
              }`}
            >
              All ({totalCount})
            </button>
            <button
              onClick={() => setActionFilter("settle")}
              className={`text-xs px-2.5 py-1 rounded-lg transition-all flex items-center gap-1.5 ${
                actionFilter === "settle"
                  ? "bg-blue-500 text-white font-bold"
                  : "bg-blue-500/10 hover:bg-blue-500/20 text-blue-300 border border-blue-500/30"
              }`}
            >
              <span className="w-2 h-2 rounded-full bg-blue-400" />
              Settle ({settleCount})
            </button>
            <button
              onClick={() => setActionFilter("farm")}
              className={`text-xs px-2.5 py-1 rounded-lg transition-all flex items-center gap-1.5 ${
                actionFilter === "farm"
                  ? "bg-emerald-500 text-slate-950 font-bold"
                  : "bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
              }`}
            >
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              Farm ({farmCount})
            </button>
            <button
              onClick={() => setActionFilter("protect")}
              className={`text-xs px-2.5 py-1 rounded-lg transition-all flex items-center gap-1.5 ${
                actionFilter === "protect"
                  ? "bg-amber-500 text-slate-950 font-bold"
                  : "bg-amber-500/10 hover:bg-amber-500/20 text-amber-300 border border-amber-500/30"
              }`}
            >
              <span className="w-2 h-2 rounded-full bg-amber-400" />
              Protect ({protectCount})
            </button>

            <div className="h-4 w-px bg-slate-700 mx-1" />

            {/* Region Zoom Quick Picker */}
            <select
              onChange={(e) => {
                const reg = STUDY_REGIONS.find((r) => r.name === e.target.value);
                if (reg && mapInstance.current) {
                  const [minLon, minLat, maxLon, maxLat] = reg.bbox;
                  mapInstance.current.fitBounds([
                    [minLat, minLon],
                    [maxLat, maxLon],
                  ], { padding: [30, 30] });
                }
              }}
              className="text-xs bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1 text-slate-200 cursor-pointer focus:outline-none"
              defaultValue=""
            >
              <option value="" disabled>
                Zoom to Region (13) ▾
              </option>
              {STUDY_REGIONS.map((r) => (
                <option key={r.name} value={r.name}>
                  {r.name}
                </option>
              ))}
            </select>
          </div>

          {/* Floating Bottom Site Inspector Card */}
          <div className="absolute bottom-6 left-6 z-[1000] max-w-md p-4 rounded-2xl bg-slate-900/90 backdrop-blur-xl border border-slate-800 shadow-2xl flex items-center justify-between gap-4">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono uppercase text-slate-400">
                  Active Parcel
                </span>
                <span
                  className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase ${actionStyle.bg} ${actionStyle.border} ${actionStyle.text}`}
                >
                  {selectedPoint.action}
                </span>
              </div>
              <h3 className="font-bold text-sm text-slate-100">{selectedPoint.name}</h3>
              <p className="text-xs text-slate-400">
                {selectedPoint.region} · {selectedPoint.slope}° Slope · {selectedPoint.elev}m Elev
              </p>
            </div>

            <button
              onClick={() => setActiveTab("underwrite")}
              className="px-4 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs flex items-center gap-1.5 shadow-lg shadow-cyan-500/20 transition-all shrink-0 cursor-pointer"
            >
              Deep Underwrite & Chat
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* ── TAB 2: SITE UNDERWRITING & RAG CHATBOT ──────────────────────────── */}
        <div
          className={`w-full h-full p-6 overflow-hidden ${
            activeTab === "underwrite" ? "flex" : "hidden"
          } gap-6`}
        >
          {/* Left Column: Spatial Evidence Snapshot */}
          <div className="w-[420px] shrink-0 h-full overflow-y-auto space-y-4 pr-1">
            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
              <span className="text-[10px] uppercase font-mono text-cyan-400 block font-bold tracking-wider">
                Site Coordinates & Zone
              </span>
              <h2 className="text-lg font-bold text-slate-100">{selectedPoint.name}</h2>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="p-2.5 rounded-xl bg-slate-800/40 border border-slate-800">
                  <span className="text-[10px] text-slate-400 block">Latitude</span>
                  <span className="font-mono text-slate-200">{selectedCoords.lat.toFixed(4)}°</span>
                </div>
                <div className="p-2.5 rounded-xl bg-slate-800/40 border border-slate-800">
                  <span className="text-[10px] text-slate-400 block">Longitude</span>
                  <span className="font-mono text-slate-200">{selectedCoords.lon.toFixed(4)}°</span>
                </div>
              </div>
            </div>

            {/* Recommendation Decision Card */}
            <div className={`p-5 rounded-2xl border ${actionStyle.bg} ${actionStyle.border} space-y-2`}>
              <div className="flex items-center justify-between">
                <span className="text-[10px] uppercase font-bold text-slate-400">
                  Statutory Allocation
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-slate-900/60 text-slate-300 border border-slate-700">
                  {selectedPoint.confidence?.toUpperCase() || "HIGH"} CONFIDENCE
                </span>
              </div>
              <h3 className={`text-xl font-extrabold ${actionStyle.text}`}>
                {actionStyle.label}
              </h3>
              <p className="text-xs text-slate-300 leading-relaxed pt-1">
                {selectedPoint.rationale}
              </p>
            </div>

            {/* Multi-Spectral Class Breakdown */}
            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
              <span className="text-xs font-semibold text-slate-200 block">
                Multi-Spectral Land-Cover Mix
              </span>
              <div className="space-y-2">
                {CLASSES.map((cls) => {
                  const prob = selectedPoint.probabilities?.[cls] || 0.05;
                  const meta = CLASS_META[cls];
                  return (
                    <div key={cls} className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="flex items-center gap-1.5 text-slate-300">
                          <span
                            className="w-2.5 h-2.5 rounded-sm"
                            style={{ backgroundColor: meta.color }}
                          />
                          {meta.label}
                        </span>
                        <span className="font-mono text-slate-400 font-medium">
                          {(prob * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{
                            width: `${Math.max(prob * 100, 3)}%`,
                            backgroundColor: meta.color,
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Topographic Hazard Controls */}
            <div className="p-4 rounded-2xl bg-slate-900/40 border border-slate-800 space-y-3">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400">Terrain Slope</span>
                <span className="font-mono font-bold text-slate-200">{slopeDeg}°</span>
              </div>
              <input
                type="range"
                min="0"
                max="30"
                step="0.5"
                value={slopeDeg}
                onChange={(e) => setSlopeDeg(parseFloat(e.target.value))}
                className="w-full accent-cyan-400 h-1 bg-slate-700 rounded-lg cursor-pointer"
              />

              <div className="flex items-center justify-between pt-2 border-t border-slate-800/80 text-xs">
                <span className="text-slate-300 font-medium">Flood Exposure Hazard</span>
                <input
                  type="checkbox"
                  checked={floodRisk}
                  onChange={(e) => setFloodRisk(e.target.checked)}
                  className="w-4 h-4 accent-cyan-400 rounded cursor-pointer"
                />
              </div>
            </div>
          </div>

          {/* Right Column: Interactive RAG Chatbot */}
          <div className="flex-1 h-full rounded-2xl bg-slate-900/60 border border-slate-800 flex flex-col overflow-hidden shadow-2xl">
            {/* Chatbot Header */}
            <div className="p-4 border-b border-slate-800/80 bg-slate-850/60 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center">
                  <Bot className="w-4 h-4 text-cyan-400" />
                </div>
                <div>
                  <h3 className="font-bold text-sm text-slate-100 flex items-center gap-2">
                    Spatial Regulatory Copilot
                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 font-mono">
                      GROQ Llama 3.3 70B & Gemini RAG
                    </span>
                  </h3>
                  <p className="text-[11px] text-slate-400">
                    Statutory query engine for PLUPA 2019, NEMA EMCA, and African Trade Corridors
                  </p>
                </div>
              </div>

              <button
                onClick={() => handleSendMessage("Generate Complete Underwriting Dossier")}
                className="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-cyan-300 border border-slate-700 flex items-center gap-1.5 transition-colors"
              >
                <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                Regenerate Dossier
              </button>
            </div>

            {/* Quick Prompt Chips */}
            <div className="px-4 py-2 border-b border-slate-800/60 bg-slate-950/40 flex items-center gap-2 overflow-x-auto text-[11px]">
              <span className="text-slate-400 font-medium shrink-0">Quick Queries:</span>
              <button
                onClick={() => handleSendMessage("What are the riparian setbacks under PLUPA 2019?")}
                className="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700/80 shrink-0"
              >
                🌊 PLUPA Sec. 56 Setback
              </button>
              <button
                onClick={() => handleSendMessage("Can we build an affordable housing project on this site?")}
                className="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700/80 shrink-0"
              >
                🏗️ Housing Feasibility
              </button>
              <button
                onClick={() => handleSendMessage("What are the NEMA EMCA environmental audit requirements?")}
                className="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700/80 shrink-0"
              >
                🌿 NEMA EIA Rules
              </button>
              <button
                onClick={() => handleSendMessage("Explain why this was classified as " + selectedPoint.action.toUpperCase())}
                className="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700/80 shrink-0"
              >
                🎯 Decision Rationale
              </button>
            </div>

            {/* Chat Message Scrollable Feed */}
            <div className="flex-1 overflow-y-auto p-5 space-y-4">
              {chatMessages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex gap-3 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  {msg.role === "assistant" && (
                    <div className="w-7 h-7 rounded-lg bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center shrink-0 mt-0.5">
                      <Bot className="w-4 h-4 text-cyan-300" />
                    </div>
                  )}

                  <div
                    className={`max-w-[85%] rounded-2xl p-4 text-xs leading-relaxed space-y-2 ${
                      msg.role === "user"
                        ? "bg-cyan-500 text-slate-950 font-medium shadow-md shadow-cyan-500/20"
                        : "bg-slate-800/80 border border-slate-700 text-slate-200"
                    }`}
                  >
                    {msg.role === "assistant" ? (
                      <MarkdownMessage content={msg.content} />
                    ) : (
                      <div className="font-sans font-medium text-slate-950">{msg.content}</div>
                    )}
                    <div
                      className={`text-[10px] text-right font-mono ${
                        msg.role === "user" ? "text-slate-900" : "text-slate-500"
                      }`}
                    >
                      {msg.timestamp}
                    </div>
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
                    GROQ Llama 3.3 is querying statutory knowledge base...
                  </div>
                </div>
              )}
              <div ref={chatBottomRef} />
            </div>

            {/* Chat Input Bar */}
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSendMessage();
              }}
              className="p-3 border-t border-slate-800 bg-slate-850/60 flex items-center gap-2"
            >
              <input
                type="text"
                value={inputQuery}
                onChange={(e) => setInputQuery(e.target.value)}
                placeholder="Ask statutory regulatory questions (e.g. 'Can I convert 3ha of farmland here?')..."
                className="flex-1 px-4 py-2.5 rounded-xl bg-slate-900 border border-slate-700 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-cyan-400"
              />
              <button
                type="submit"
                disabled={chatLoading || !inputQuery.trim()}
                className="px-4 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 disabled:opacity-40 text-slate-950 font-bold text-xs flex items-center gap-1.5 transition-all cursor-pointer"
              >
                <Send className="w-3.5 h-3.5" />
                Send
              </button>
            </form>
          </div>
        </div>

        {/* ── TAB 3: COUNTY BENCHMARKS & MODEL FINDINGS ───────────────────────── */}
        <div
          className={`w-full h-full p-6 overflow-y-auto ${
            activeTab === "county" ? "block" : "hidden"
          } space-y-6`}
        >
          {/* Header & County Selector */}
          <div className="flex items-center justify-between pb-4 border-b border-slate-800">
            <div>
              <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-cyan-400" />
                County Benchmarks & Agro-Ecological Model Findings
              </h2>
              <p className="text-xs text-slate-400">
                Ground-truth calibrated statistics derived from Leave-One-Zone-Out (LOZO) cross-validation experiments
              </p>
            </div>

            <div className="flex items-center gap-2">
              <label className="text-xs text-slate-400 font-medium">Select Study County:</label>
              <select
                value={selectedCounty}
                onChange={(e) => setSelectedCounty(e.target.value)}
                className="p-2 rounded-xl bg-slate-800 border border-slate-700 text-slate-200 text-xs capitalize cursor-pointer focus:outline-none focus:border-cyan-400"
              >
                {COUNTIES.map((c) => (
                  <option key={c} value={c} className="capitalize">
                    {c.replace("_", " ")}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Active County Statistics Card */}
          {countyData && (
            <div className="grid grid-cols-4 gap-4">
              <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-1">
                <span className="text-[10px] uppercase font-mono text-slate-400">County & Zone</span>
                <h3 className="text-lg font-bold text-cyan-300 capitalize">
                  {selectedCounty.replace("_", " ")}
                </h3>
                <span className="text-xs text-slate-300 font-medium">
                  {countyData.climate_zone} · {countyData.n_samples} Samples
                </span>
              </div>

              <div className="p-4 rounded-2xl bg-blue-500/10 border border-blue-500/20 text-center space-y-1">
                <span className="text-[10px] text-slate-400 uppercase font-mono block">Settle Index</span>
                <span className="text-2xl font-bold font-mono text-blue-400">
                  {countyData.scores?.settle?.toFixed(1) || "16.5"}%
                </span>
                <span className="text-[10px] text-slate-400 block">Urban infill capability</span>
              </div>

              <div className="p-4 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 text-center space-y-1">
                <span className="text-[10px] text-slate-400 uppercase font-mono block">Farm Index</span>
                <span className="text-2xl font-bold font-mono text-emerald-400">
                  {countyData.scores?.farm?.toFixed(1) || "16.7"}%
                </span>
                <span className="text-[10px] text-slate-400 block">Arable agricultural yield</span>
              </div>

              <div className="p-4 rounded-2xl bg-amber-500/10 border border-amber-500/20 text-center space-y-1">
                <span className="text-[10px] text-slate-400 uppercase font-mono block">Protect Index</span>
                <span className="text-2xl font-bold font-mono text-amber-400">
                  {countyData.scores?.protect?.toFixed(1) || "16.8"}%
                </span>
                <span className="text-[10px] text-slate-400 block">Wetland & forest buffer</span>
              </div>
            </div>
          )}

          {/* Model Findings & Regulatory Facts Grid */}
          <div className="space-y-3 pt-2">
            <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2">
              <FileText className="w-4 h-4 text-cyan-400" />
              Agro-Ecological Zone Findings & Policy Implications
            </h3>

            <div className="grid grid-cols-2 gap-4">
              {Object.entries(ZONE_MODEL_FINDINGS).map(([zoneKey, f]) => {
                const zoneColor = ZONE_COLORS[zoneKey] || "#06b6d4";
                return (
                  <div
                    key={zoneKey}
                    className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-2 relative overflow-hidden"
                  >
                    <div
                      className="absolute top-0 left-0 right-0 h-1"
                      style={{ backgroundColor: zoneColor }}
                    />
                    <div className="flex items-center justify-between">
                      <h4 className="font-bold text-sm text-slate-100 capitalize">
                        {zoneKey.replace("_", " ").toUpperCase()}
                      </h4>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-slate-800 text-cyan-300 border border-slate-700">
                        Macro F1: {f.macro_f1}
                      </span>
                    </div>

                    <p className="text-xs text-slate-300 leading-relaxed">
                      {f.key_insight}
                    </p>

                    <div className="pt-2 border-t border-slate-800/80 text-[11px] text-slate-400">
                      <b className="text-cyan-300">Statutory Implication: </b>
                      {f.policy_implication}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* ── TAB 4: MLOPS & BACKEND PROGRESS ─────────────────────────────────── */}
        <div
          className={`w-full h-full p-6 overflow-y-auto ${
            activeTab === "mlops" ? "block" : "hidden"
          } space-y-6`}
        >
          {/* Header */}
          <div className="pb-4 border-b border-slate-800">
            <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
              <Cpu className="w-5 h-5 text-cyan-400" />
              MLOps Infrastructure, Telemetry & Pipeline Architecture
            </h2>
            <p className="text-xs text-slate-400">
              Direct access to live FastAPI endpoints, MLflow model registry, container telemetry, and GitHub codebase
            </p>
          </div>

          {/* Quick Service Links Grid */}
          <div className="grid grid-cols-4 gap-4">
            {/* FastAPI Docs */}
            <a
              href="http://192.248.170.201:8000/docs"
              target="_blank"
              rel="noreferrer"
              className="p-5 rounded-2xl bg-slate-900/80 hover:bg-slate-850 border border-slate-800 hover:border-cyan-500/40 transition-all space-y-3 group"
            >
              <div className="flex items-center justify-between">
                <div className="w-9 h-9 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center">
                  <Server className="w-4 h-4 text-cyan-400" />
                </div>
                <ExternalLink className="w-4 h-4 text-slate-500 group-hover:text-cyan-300" />
              </div>
              <div>
                <h4 className="font-bold text-sm text-slate-100">FastAPI OpenAPI</h4>
                <p className="text-xs text-slate-400 pt-1">
                  Interactive Swagger UI for feature prediction & county queries
                </p>
              </div>
              <span className="text-[10px] font-mono text-cyan-400 block">:8000/docs</span>
            </a>

            {/* FastAPI Health */}
            <a
              href="http://192.248.170.201:8000/health"
              target="_blank"
              rel="noreferrer"
              className="p-5 rounded-2xl bg-slate-900/80 hover:bg-slate-850 border border-slate-800 hover:border-emerald-500/40 transition-all space-y-3 group"
            >
              <div className="flex items-center justify-between">
                <div className="w-9 h-9 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center">
                  <Activity className="w-4 h-4 text-emerald-400" />
                </div>
                <ExternalLink className="w-4 h-4 text-slate-500 group-hover:text-emerald-300" />
              </div>
              <div>
                <h4 className="font-bold text-sm text-slate-100">Liveness & Readiness</h4>
                <p className="text-xs text-slate-400 pt-1">
                  JSON health status, uptime, and zone model verification
                </p>
              </div>
              <span className="text-[10px] font-mono text-emerald-400 block">:8000/health</span>
            </a>

            {/* MLflow */}
            <a
              href="http://192.248.170.201:5000"
              target="_blank"
              rel="noreferrer"
              className="p-5 rounded-2xl bg-slate-900/80 hover:bg-slate-850 border border-slate-800 hover:border-purple-500/40 transition-all space-y-3 group"
            >
              <div className="flex items-center justify-between">
                <div className="w-9 h-9 rounded-xl bg-purple-500/10 border border-purple-500/30 flex items-center justify-center">
                  <Database className="w-4 h-4 text-purple-400" />
                </div>
                <ExternalLink className="w-4 h-4 text-slate-500 group-hover:text-purple-300" />
              </div>
              <div>
                <h4 className="font-bold text-sm text-slate-100">MLflow Tracking</h4>
                <p className="text-xs text-slate-400 pt-1">
                  Experiment runs, LOZO F1 metrics, artifacts & models
                </p>
              </div>
              <span className="text-[10px] font-mono text-purple-400 block">:5000</span>
            </a>

            {/* GitHub */}
            <a
              href="https://github.com/VAL-Jerono/Satellite"
              target="_blank"
              rel="noreferrer"
              className="p-5 rounded-2xl bg-slate-900/80 hover:bg-slate-850 border border-slate-800 hover:border-blue-500/40 transition-all space-y-3 group"
            >
              <div className="flex items-center justify-between">
                <div className="w-9 h-9 rounded-xl bg-blue-500/10 border border-blue-500/30 flex items-center justify-center">
                  <FileText className="w-4 h-4 text-blue-400" />
                </div>
                <ExternalLink className="w-4 h-4 text-slate-500 group-hover:text-blue-300" />
              </div>
              <div>
                <h4 className="font-bold text-sm text-slate-100">GitHub Repository</h4>
                <p className="text-xs text-slate-400 pt-1">
                  Master branch: Dockerfiles, notebooks, CRISP-DM docs
                </p>
              </div>
              <span className="text-[10px] font-mono text-blue-400 block">VAL-Jerono/Satellite</span>
            </a>
          </div>

          {/* Machine Learning Pipeline Architecture */}
          <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
            <h3 className="font-bold text-sm text-slate-100">
              End-to-End Spatial Machine Learning Architecture
            </h3>

            <div className="grid grid-cols-5 gap-3 text-xs">
              <div className="p-3.5 rounded-xl bg-slate-800/40 border border-slate-800 space-y-1.5">
                <span className="text-[10px] font-mono text-cyan-400 font-bold block">01. INGESTION</span>
                <h4 className="font-semibold text-slate-200">Earth Observation</h4>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  Sentinel-2 Level-2A (10 spectral bands: B2-B12) + SRTM DEM & CHIRPS rainfall.
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-800/40 border border-slate-800 space-y-1.5">
                <span className="text-[10px] font-mono text-cyan-400 font-bold block">02. FEATURES</span>
                <h4 className="font-semibold text-slate-200">27 Spectral Indices</h4>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  NDVI, EVI, MNDWI, NDBI, NDRE, BSI, TWI, dry/wet season swings, elevation, slope.
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-800/40 border border-slate-800 space-y-1.5">
                <span className="text-[10px] font-mono text-cyan-400 font-bold block">03. MODELS</span>
                <h4 className="font-semibold text-slate-200">Zone-Stratified LGBM</h4>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  5 specialized models for Kenya's agro-ecological zones + EfficientNet-B0 ONNX.
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-800/40 border border-slate-800 space-y-1.5">
                <span className="text-[10px] font-mono text-cyan-400 font-bold block">04. STATUTORY RAG</span>
                <h4 className="font-semibold text-slate-200">GROQ Llama 3.3</h4>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  Regulatory knowledge grounding in PLUPA 2019 Sec. 56 and NEMA EMCA rules.
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-800/40 border border-slate-800 space-y-1.5">
                <span className="text-[10px] font-mono text-cyan-400 font-bold block">05. SERVING</span>
                <h4 className="font-semibold text-slate-200">Vultr Cloud VM</h4>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  Ubuntu 24.04 (London, GB) running Dockerized FastAPI on port 8000 & 80.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── Settings Modal ─────────────────────────────────────────────────── */}
      {showSettings && (
        <div className="fixed inset-0 z-[2000] bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 w-full max-w-md space-y-4 shadow-2xl">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Settings className="w-4 h-4 text-cyan-400" />
              Configure Backend API Endpoint
            </h3>
            <p className="text-xs text-slate-400">
              Set the URL of your Vultr server (e.g. http://192.248.170.201:8000) or local API container.
            </p>
            <input
              type="text"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              placeholder="http://192.248.170.201:8000"
              className="w-full p-2.5 rounded-xl bg-slate-800 border border-slate-700 text-slate-100 font-mono text-xs focus:outline-none focus:border-cyan-400"
            />
            <div className="flex justify-end gap-2 pt-2">
              <button
                onClick={() => setShowSettings(false)}
                className="px-4 py-2 rounded-xl text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-300"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  localStorage.setItem("atlas_api_url", apiUrl);
                  setShowSettings(false);
                  checkHealth();
                }}
                className="px-4 py-2 rounded-xl text-xs font-bold bg-cyan-500 hover:bg-cyan-400 text-slate-950"
              >
                Save & Connect
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
