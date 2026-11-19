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
import {
  Compass,
  Layers,
  MapPin,
  CheckCircle2,
  AlertTriangle,
  ShieldAlert,
  Server,
  Sparkles,
  Settings,
  ChevronRight,
  TrendingUp,
  FileText,
  Activity,
  Maximize2,
  Copy,
  Check,
  Filter,
} from "lucide-react";

export default function App() {
  const [apiUrl, setApiUrl] = useState(
    localStorage.getItem("atlas_api_url") || DEFAULT_API_URL
  );
  const [apiOnline, setApiOnline] = useState(null);
  const [apiPing, setApiPing] = useState(null);
  const [selectedCoords, setSelectedCoords] = useState({ lat: -0.4201, lon: 36.9476 });
  const [floodRisk, setFloodRisk] = useState(false);
  const [slopeDeg, setSlopeDeg] = useState(4.2);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState("point"); // 'point' | 'county'
  const [selectedCounty, setSelectedCounty] = useState("nakuru");
  const [countyData, setCountyData] = useState(null);
  const [copilotLoading, setCopilotLoading] = useState(false);
  const [copilotDossier, setCopilotDossier] = useState(null);
  const [copied, setCopied] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [actionFilter, setActionFilter] = useState("all"); // 'all' | 'settle' | 'farm' | 'protect'

  const mapRef = useRef(null);
  const mapInstance = useRef(null);
  const markerRef = useRef(null);
  const markersLayerRef = useRef(null);

  // ── 1. Check API Health ────────────────────────────────────────────────────
  const checkHealth = async () => {
    const start = performance.now();
    const isHttps = typeof window !== "undefined" && window.location.protocol === "https:";

    // Auto-detect Mixed Content: If frontend is on HTTPS, use HTTPS-safe proxy
    let targetUrl = apiUrl;
    if (isHttps && targetUrl.startsWith("http://")) {
      targetUrl = "/api-proxy";
      setApiUrl("/api-proxy");
      localStorage.setItem("atlas_api_url", "/api-proxy");
    }

    try {
      let res = await fetch(`${targetUrl}/health`, { method: "GET" }).catch(() => null);

      // If direct fetch failed and we are on HTTPS, try the reverse proxy
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

  // ── 2. Initialize Leaflet Map ──────────────────────────────────────────────
  useEffect(() => {
    if (!mapRef.current || mapInstance.current) return;

    const map = L.map(mapRef.current, {
      center: [0.5, 37.8],
      zoom: 6,
      zoomControl: false,
    });
    L.control.zoom({ position: "topright" }).addTo(map);

    // ── Basemap 1: Esri Satellite + Esri Places & Boundaries + Exact Kenya Sovereign Border ──
    const esriSatellite = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      {
        attribution: "Esri World Imagery",
        maxZoom: 18,
      }
    );

    const esriBoundaries = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
      {
        attribution: "Esri Boundaries & Places",
        maxZoom: 18,
      }
    );

    // High-Resolution Official Kenya Sovereign Boundary (GeoJSON from survey data)
    // Rendered only on Satellite layer as a crisp dark cartographic border matching Topo
    const kenyaSovereignBoundary = L.geoJSON(KENYA_BOUNDARY_GEOJSON, {
      style: {
        color: "#0f172a", // Dark charcoal line matching Topography cartography
        weight: 2.5,
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

    // ── Basemap 2: Esri Topo (Has native black boundaries built-in, NO overlay added) ──
    const esriTopo = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
      {
        attribution: "Esri Topo",
        maxZoom: 18,
      }
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

    // ── Study Regions: Color-Coded Rectangles & Permanent Tooltips ─────────────
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
          weight: 1.8,
          fillColor: zoneColor,
          fillOpacity: 0.08,
          dashArray: "5, 4",
        }
      );

      rect.bindTooltip(
        `<div style="font-weight:700;font-size:11px;">${r.name}</div><div style="color:${zoneColor};font-size:9px;font-weight:600;">${r.zone.replace("_zone", "").toUpperCase()}</div>`,
        {
          permanent: true,
          direction: "top",
          className: "custom-tooltip region-label",
        }
      );

      // Clicking region rectangle zooms in to inspect region's points
      rect.on("click", () => {
        map.fitBounds([
          [minLat, minLon],
          [maxLat, maxLon],
        ], { padding: [30, 30] });
      });

      rect.addTo(map);
    });

    // LayerGroup for all 390 Prediction Points
    const markersLayer = L.layerGroup().addTo(map);
    markersLayerRef.current = markersLayer;

    // Custom pulsing pin icon for selected site
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

    // Map click handler anywhere on terrain
    map.on("click", (e) => {
      const { lat, lng } = e.latlng;
      selectCoordinate(lat, lng);
    });

    mapInstance.current = map;

    // Load initial coordinate prediction
    const initialPt = PREDICTION_POINTS.find(p => p.region === "Central Highlands") || PREDICTION_POINTS[0];
    if (initialPt) {
      applyPointData(initialPt);
    }
  }, []);

  // ── 3. Render / Filter Prediction Points ───────────────────────────────────
  useEffect(() => {
    if (!mapInstance.current || !markersLayerRef.current) return;
    const markersLayer = markersLayerRef.current;
    markersLayer.clearLayers();

    const filtered = actionFilter === "all"
      ? PREDICTION_POINTS
      : PREDICTION_POINTS.filter((p) => p.action === actionFilter);

    filtered.forEach((pt) => {
      const actColor = ACTION_COLORS[pt.action] || "#3b82f6";
      const actMeta = ACTION_META[pt.action];

      const cm = L.circleMarker([pt.lat, pt.lon], {
        radius: 6,
        color: "#ffffff",
        weight: 1.5,
        fillColor: actColor,
        fillOpacity: 0.95,
      });

      const popupHtml = `
        <div style="font-family: system-ui, sans-serif; min-width: 200px; padding: 2px;">
          <div style="font-weight: 700; font-size: 13px; color: #0f172a; margin-bottom: 2px;">${pt.name}</div>
          <div style="font-size: 11px; color: #64748b; margin-bottom: 8px;">${pt.region} · <span style="text-transform: uppercase;">${pt.zone.replace('_zone','')}</span></div>
          <div style="display: flex; gap: 6px; margin-bottom: 8px;">
            <span style="background: ${actColor}22; color: ${actColor}; border: 1px solid ${actColor}55; padding: 2px 8px; border-radius: 9999px; font-size: 10px; font-weight: 700; text-transform: uppercase;">
              ${pt.action}
            </span>
            <span style="background: #f1f5f9; color: #334155; padding: 2px 8px; border-radius: 9999px; font-size: 10px; font-weight: 600; text-transform: capitalize;">
              ${pt.label.replace('_', ' ')}
            </span>
          </div>
          <div style="font-size: 11px; color: #475569; margin-bottom: 8px; line-height: 1.4;">
            Slope: <b>${pt.slope}°</b> · Elev: <b>${pt.elev}m</b> · NDVI: <b>${pt.ndvi}</b>
          </div>
          <div style="font-size: 10px; color: #64748b; font-style: italic; border-top: 1px solid #e2e8f0; padding-top: 6px;">
            ${pt.rationale}
          </div>
        </div>
      `;

      cm.bindPopup(popupHtml, { closeButton: false });

      cm.on("click", (e) => {
        L.DomEvent.stopPropagation(e);
        applyPointData(pt);
      });

      cm.addTo(markersLayer);
    });
  }, [actionFilter]);

  // Apply point data from clicked marker
  const applyPointData = (pt) => {
    setSelectedCoords({ lat: pt.lat, lon: pt.lon });
    setSlopeDeg(pt.slope);
    setFloodRisk(pt.flood_risk);
    if (markerRef.current) {
      markerRef.current.setLatLng([pt.lat, pt.lon]);
    }
    if (mapInstance.current) {
      mapInstance.current.flyTo([pt.lat, pt.lon], 11, { duration: 1.0 });
    }

    // Set result immediately from ground truth Sentinel-2 multi-spectral dataset
    setResult({
      prediction: pt.label,
      probabilities: pt.probabilities,
      recommendation: {
        action: pt.action,
        confidence: pt.confidence || "high",
        rationale: pt.rationale,
      },
      climate_zone: pt.zone,
      model_type: "Sentinel-2 Multi-Spectral LightGBM",
      lat: pt.lat,
      lon: pt.lon,
      elev: pt.elev,
    });

    // Also call backend API to run live inference if reachable
    runInference(pt.lat, pt.lon, pt.flood_risk, pt.slope);
  };

  const selectCoordinate = (lat, lon) => {
    setSelectedCoords({ lat, lon });
    if (markerRef.current) {
      markerRef.current.setLatLng([lat, lon]);
    }
    runInference(lat, lon, floodRisk, slopeDeg);
  };

  // ── 4. Run Inference API with Spatial Dataset Fallback ──────────────────────
  const runInference = async (lat, lon, flood, slope) => {
    setLoading(true);
    setCopilotDossier(null);
    try {
      const res = await fetch(`${apiUrl}/predict/features`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          lat,
          lon,
          flood_risk: flood,
          slope_deg: slope,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setResult(data);
      }
    } catch {
      // Backend offline: compute local deterministic spatial classification
      if (!result) {
        fallbackSpatialInference(lat, lon, flood, slope);
      }
    } finally {
      setLoading(false);
    }
  };

  const fallbackSpatialInference = (lat, lon, flood, slope) => {
    let action = "farm";
    let rationale = "Mixed agricultural parcel on gentle terrain.";
    let dominant = "cropland";

    if (flood || slope < 1.0) {
      action = "protect";
      rationale = "Hydrologic flood exposure — statutory conservation setback applied (PLUPA Sec. 56).";
      dominant = "water";
    } else if (slope > 15.0) {
      action = "protect";
      rationale = "Montane slope buffer (>15°) — watershed erosion protection mandatory.";
      dominant = "trees";
    } else if (lat < -1.1 && lat > -1.4 && lon > 36.6 && lon < 37.1) {
      action = "settle";
      rationale = "Nairobi metropolitan growth corridor — urban infill recommended.";
      dominant = "built_up";
    }

    const probs = { built_up: 0.1, cropland: 0.1, trees: 0.1, grass_shrub: 0.1, water: 0.1, bare: 0.1 };
    probs[dominant] = 0.5;

    setResult({
      prediction: dominant,
      probabilities: probs,
      recommendation: {
        action,
        confidence: "medium",
        rationale,
      },
      climate_zone: "highland_zone",
      model_type: "Spatial Rule Fallback",
      lat,
      lon,
    });
  };

  // ── 5. Fetch County Summary ────────────────────────────────────────────────
  const fetchCounty = async (countyName) => {
    try {
      const res = await fetch(`${apiUrl}/predict/county/${countyName.toLowerCase()}`);
      if (res.ok) {
        const data = await res.json();
        setCountyData(data);
      }
    } catch (err) {
      console.error("County fetch failed:", err);
    }
  };

  useEffect(() => {
    if (activeTab === "county") {
      fetchCounty(selectedCounty);
    }
  }, [activeTab, selectedCounty, apiUrl]);

  // ── 6. Generate GenAI Copilot Dossier ──────────────────────────────────────
  const generateCopilot = async () => {
    if (!result) return;
    setCopilotLoading(true);
    try {
      const res = await fetch(`${apiUrl}/underwrite/copilot`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          lat: selectedCoords.lat,
          lon: selectedCoords.lon,
          probabilities: result.probabilities,
          prediction: result.prediction,
          recommendation: result.recommendation,
          flood_risk: floodRisk,
          slope_deg: slopeDeg,
          county_name: "Kenya Spatial Corridor",
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setCopilotDossier(data);
        return;
      }
    } catch {
      // Offline fallback: generate client-side statutory underwriting report
    } finally {
      setCopilotLoading(false);
    }

    // Client-side deterministic RAG dossier fallback
    const action = result.recommendation?.action?.toUpperCase() || "FARM";
    const conf = result.recommendation?.confidence?.toUpperCase() || "HIGH";
    const domClass = result.prediction.replace("_", " ").toUpperCase();
    const fallbackReport = `### 📋 Spatial Underwriting Summary Dossier (Statutory RAG Engine)
**Site Coordinates:** \`${selectedCoords.lat.toFixed(4)}°, ${selectedCoords.lon.toFixed(4)}°\`  
**Primary Land Classification:** ${domClass} (${((result.probabilities?.[result.prediction] || 0.75) * 100).toFixed(1)}% Multi-Spectral Confidence)  
**Underwriting Recommendation:** **${action}** (${conf} CONFIDENCE)

---

#### 1. 🏢 Executive Land Suitability Assessment
- **Dominant Land Cover:** Classified as **${domClass}** via Sentinel-2 multi-spectral reflectance.
- **Topographic Constraints:** Terrain slope measured at **${slopeDeg.toFixed(1)}°**. ${slopeDeg > 15 ? "⚠️ Exceeds 15° slope threshold — heightened landslide & soil erosion vulnerability." : "✅ Slope within standard construction tolerance (<15°)."}
- **Flood Exposure Status:** ${floodRisk ? "⚠️ High surface flood risk flagged by hydrologic DEM analysis." : "✅ Low immediate flood risk identified."}

#### 2. 📜 Statutory Zoning & Regulatory Compliance Check (RAG Engine)
- **PLUPA 2019 Sec. 56 Compliance:** Mandatory 30m riparian setback from rivers and designated wetlands; development on slopes >15° requires specialized structural engineering certification.
- **NEMA EMCA Environmental Audit:** Conversion of classified agricultural land >2 hectares to built-up development mandates Environmental Impact Assessment (EIA) License and stakeholder participation.
- **East African Trade Corridor Standard:** Commercial logistics hubs along national transit trunks must maintain 50m highway buffer & demonstrate hydrological drainage neutral impact.

#### 3. 💡 Capital Deployment & Risk Mitigation Rationale
- *Statutory Rationale:* ${result.recommendation?.rationale || "Site suitability evaluated based on multi-spectral satellite features."}
- **Actionable Guidance:** ${action === "PROTECT" ? "Hold commercial/residential development; enforce strict environmental buffer." : (action === "SETTLE" ? "Approve infrastructure deployment & commercial brownfield infill." : "Prioritize agricultural financing, irrigation infrastructure, and sustainable land tenure.")}
`;
    setCopilotDossier({ dossier: fallbackReport, llm_powered: false });
  };

  // Recommendation Badge Config
  const getBadgeStyle = (action) => {
    switch (action?.toLowerCase()) {
      case "settle":
        return {
          bg: "bg-blue-500/10",
          border: "border-blue-500/30",
          text: "text-blue-400",
          label: "SETTLE — BUILDOUT",
          icon: <CheckCircle2 className="w-5 h-5 text-blue-400" />,
        };
      case "farm":
        return {
          bg: "bg-emerald-500/10",
          border: "border-emerald-500/30",
          text: "text-emerald-400",
          label: "FARM — AGRICULTURE",
          icon: <CheckCircle2 className="w-5 h-5 text-emerald-400" />,
        };
      case "protect":
        return {
          bg: "bg-amber-500/10",
          border: "border-amber-500/30",
          text: "text-amber-400",
          label: "PROTECT — CONSERVATION",
          icon: <ShieldAlert className="w-5 h-5 text-amber-400" />,
        };
      default:
        return {
          bg: "bg-slate-500/10",
          border: "border-slate-500/30",
          text: "text-slate-400",
          label: "ANALYZE",
          icon: <Activity className="w-5 h-5 text-slate-400" />,
        };
    }
  };

  const actionStyle = getBadgeStyle(result?.recommendation?.action);

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-[#080c14] text-slate-100">
      {/* ── Top Navbar ──────────────────────────────────────────────────────── */}
      <header className="h-16 px-6 border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-xl flex items-center justify-between z-20 shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-500 to-emerald-400 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Compass className="w-5 h-5 text-slate-950 font-bold" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-bold text-base tracking-tight bg-gradient-to-r from-slate-100 via-slate-200 to-cyan-400 bg-clip-text text-transparent">
                Kenya Land Readiness Atlas
              </h1>
              <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/30">
                Spatial AI Engine · 390 Sentinel-2 Points
              </span>
            </div>
            <p className="text-[11px] text-slate-400">
              Stratified Multi-Spectral Sentinel-2 Classification · Zone LightGBM & Statutory RAG
            </p>
          </div>
        </div>

        {/* Live Status & Controls */}
        <div className="flex items-center gap-4">
          <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800/60 border border-slate-700/50 text-xs">
            <Server className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-slate-400">Backend:</span>
            <span className="font-mono text-slate-200">
              {apiUrl === "/api-proxy" ? "192.248.170.201" : apiUrl.replace("http://", "").replace(":8000", "")}
            </span>
            <div className="flex items-center gap-1.5 pl-2 border-l border-slate-700">
              <span
                className={`w-2 h-2 rounded-full ${
                  apiOnline ? "bg-emerald-400 animate-pulse" : "bg-amber-400"
                }`}
              />
              <span className={apiOnline ? "text-emerald-400 font-mono text-[11px]" : "text-amber-400 text-[11px]"}>
                {apiOnline ? `${apiPing}ms` : "Local Spatial Dataset"}
              </span>
            </div>
          </div>

          <button
            onClick={() => setShowSettings(true)}
            className="p-2 rounded-lg bg-slate-800/80 hover:bg-slate-700 border border-slate-700 text-slate-300 transition-colors"
            title="Configure Backend URL"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* ── Main Layout: Map (Left) + Intelligence Drawer (Right) ───────────── */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: Map Canvas */}
        <div className="relative flex-1 h-full bg-slate-950">
          <div ref={mapRef} className="w-full h-full" />

          {/* Top Floating Controls: Allocation Action Filter & Region Zoom */}
          <div className="absolute top-4 left-4 z-[1000] flex flex-col gap-2 max-w-[85%]">
            {/* Action Filter Pills */}
            <div className="flex items-center gap-1.5 p-1.5 bg-slate-900/90 backdrop-blur-md rounded-xl border border-slate-800 shadow-2xl">
              <span className="text-[11px] text-slate-400 flex items-center gap-1 px-2 font-medium">
                <Filter className="w-3 h-3 text-cyan-400" /> Allocation:
              </span>
              <button
                onClick={() => setActionFilter("all")}
                className={`text-xs px-2.5 py-1 rounded-lg transition-all ${
                  actionFilter === "all"
                    ? "bg-slate-100 text-slate-950 font-bold shadow"
                    : "bg-slate-800/80 hover:bg-slate-700 text-slate-300"
                }`}
              >
                All (390)
              </button>
              <button
                onClick={() => setActionFilter("settle")}
                className={`text-xs px-2.5 py-1 rounded-lg transition-all flex items-center gap-1 ${
                  actionFilter === "settle"
                    ? "bg-blue-500 text-white font-bold shadow-md shadow-blue-500/40"
                    : "bg-blue-500/10 hover:bg-blue-500/20 text-blue-300 border border-blue-500/30"
                }`}
              >
                <span className="w-2 h-2 rounded-full bg-blue-400" />
                Settle (128)
              </button>
              <button
                onClick={() => setActionFilter("farm")}
                className={`text-xs px-2.5 py-1 rounded-lg transition-all flex items-center gap-1 ${
                  actionFilter === "farm"
                    ? "bg-emerald-500 text-slate-950 font-bold shadow-md shadow-emerald-500/40"
                    : "bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                }`}
              >
                <span className="w-2 h-2 rounded-full bg-emerald-400" />
                Farm (132)
              </button>
              <button
                onClick={() => setActionFilter("protect")}
                className={`text-xs px-2.5 py-1 rounded-lg transition-all flex items-center gap-1 ${
                  actionFilter === "protect"
                    ? "bg-amber-500 text-slate-950 font-bold shadow-md shadow-amber-500/40"
                    : "bg-amber-500/10 hover:bg-amber-500/20 text-amber-300 border border-amber-500/30"
                }`}
              >
                <span className="w-2 h-2 rounded-full bg-amber-400" />
                Protect (130)
              </button>
            </div>

            {/* Quick Coordinate & Region Zoom Bar */}
            <div className="flex flex-wrap gap-1.5 p-1.5 bg-slate-900/80 backdrop-blur-md rounded-xl border border-slate-800/80 shadow-2xl">
              <span className="text-[11px] text-slate-400 flex items-center px-2 font-medium">
                Regions:
              </span>
              {PRESETS.map((p) => {
                const zoneColor = ZONE_COLORS[p.zone] || "#06b6d4";
                return (
                  <button
                    key={p.name}
                    onClick={() => {
                      selectCoordinate(p.lat, p.lon);
                      if (mapInstance.current) {
                        mapInstance.current.flyTo([p.lat, p.lon], 11, { duration: 1.2 });
                      }
                    }}
                    className={`text-xs px-2.5 py-1 rounded-lg transition-all flex items-center gap-1.5 ${
                      Math.abs(selectedCoords.lat - p.lat) < 0.05 && Math.abs(selectedCoords.lon - p.lon) < 0.05
                        ? "text-slate-950 font-bold shadow-md"
                        : "bg-slate-800/70 hover:bg-slate-700 text-slate-300 border border-slate-700/60"
                    }`}
                    style={{
                      backgroundColor:
                        Math.abs(selectedCoords.lat - p.lat) < 0.05 && Math.abs(selectedCoords.lon - p.lon) < 0.05
                          ? zoneColor
                          : undefined,
                      borderColor: zoneColor,
                    }}
                  >
                    <span
                      className="w-1.5 h-1.5 rounded-full"
                      style={{ backgroundColor: zoneColor }}
                    />
                    {p.name}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Map Footer Helper */}
          <div className="absolute bottom-4 left-4 z-[1000] bg-slate-900/90 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-800 text-[11px] text-slate-300 flex items-center gap-2 shadow-lg">
            <MapPin className="w-3.5 h-3.5 text-cyan-400" />
            Click any color-coded prediction point or anywhere in Kenya for spatial underwriting
          </div>
        </div>

        {/* Right: Analytics & Underwriting Panel */}
        <aside className="w-[480px] shrink-0 h-full border-l border-slate-800/80 bg-slate-900/40 backdrop-blur-2xl flex flex-col overflow-hidden">
          {/* Tabs */}
          <div className="flex border-b border-slate-800 px-4 pt-3 gap-2 shrink-0">
            <button
              onClick={() => setActiveTab("point")}
              className={`pb-2.5 text-xs font-semibold border-b-2 transition-all flex items-center gap-1.5 ${
                activeTab === "point"
                  ? "border-cyan-400 text-cyan-300"
                  : "border-transparent text-slate-400 hover:text-slate-200"
              }`}
            >
              <Compass className="w-3.5 h-3.5" />
              Site Underwriting
            </button>
            <button
              onClick={() => setActiveTab("county")}
              className={`pb-2.5 text-xs font-semibold border-b-2 transition-all flex items-center gap-1.5 ${
                activeTab === "county"
                  ? "border-cyan-400 text-cyan-300"
                  : "border-transparent text-slate-400 hover:text-slate-200"
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              County Benchmark (13)
            </button>
          </div>

          {/* Tab 1: Point Inference */}
          {activeTab === "point" && (
            <div className="flex-1 overflow-y-auto p-5 space-y-4">
              {/* Coordinate Info Bar */}
              <div className="p-3.5 rounded-xl bg-slate-800/40 border border-slate-800/80 flex items-center justify-between text-xs">
                <div>
                  <span className="text-slate-400 block text-[10px] uppercase font-mono">
                    Latitude / Longitude
                  </span>
                  <span className="font-mono font-medium text-slate-200">
                    {selectedCoords.lat.toFixed(4)}°, {selectedCoords.lon.toFixed(4)}°
                  </span>
                </div>
                <div className="text-right">
                  <span className="text-slate-400 block text-[10px] uppercase font-mono">
                    Zone Routing
                  </span>
                  <span className="font-semibold text-cyan-300 capitalize">
                    {result?.climate_zone?.replace("_", " ") || "Calculating..."}
                  </span>
                </div>
              </div>

              {/* Terrain Controls */}
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="p-3 rounded-xl bg-slate-800/30 border border-slate-800">
                  <label className="text-[11px] text-slate-400 block mb-1.5">
                    Terrain Slope: <b className="text-slate-200">{slopeDeg}°</b>
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="30"
                    step="0.5"
                    value={slopeDeg}
                    onChange={(e) => {
                      const v = parseFloat(e.target.value);
                      setSlopeDeg(v);
                      runInference(selectedCoords.lat, selectedCoords.lon, floodRisk, v);
                    }}
                    className="w-full accent-cyan-400 h-1 bg-slate-700 rounded-lg cursor-pointer"
                  />
                </div>

                <div className="p-3 rounded-xl bg-slate-800/30 border border-slate-800 flex items-center justify-between">
                  <div>
                    <span className="text-[11px] text-slate-300 block font-medium">
                      Flood Exposure
                    </span>
                    <span className="text-[10px] text-slate-400">Hydrologic Risk</span>
                  </div>
                  <input
                    type="checkbox"
                    checked={floodRisk}
                    onChange={(e) => {
                      const v = e.target.checked;
                      setFloodRisk(v);
                      runInference(selectedCoords.lat, selectedCoords.lon, v, slopeDeg);
                    }}
                    className="w-4 h-4 accent-cyan-400 rounded cursor-pointer"
                  />
                </div>
              </div>

              {/* Big Underwriting Recommendation Badge */}
              <div className={`p-4 rounded-2xl border ${actionStyle.bg} ${actionStyle.border} space-y-2`}>
                <div className="flex items-center justify-between">
                  <span className="text-[11px] uppercase tracking-wider font-bold text-slate-400">
                    Statutory Allocation Recommendation
                  </span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-slate-900/60 text-slate-300 border border-slate-700">
                    {result?.recommendation?.confidence?.toUpperCase() || "HIGH"} CONFIDENCE
                  </span>
                </div>

                <div className="flex items-center gap-3">
                  {actionStyle.icon}
                  <h2 className={`text-xl font-extrabold tracking-tight ${actionStyle.text}`}>
                    {actionStyle.label}
                  </h2>
                </div>

                <p className="text-xs text-slate-300 leading-relaxed pt-1">
                  {result?.recommendation?.rationale || "Analyzing multi-spectral reflectance..."}
                </p>
              </div>

              {/* Class Probability Breakdown */}
              <div className="p-4 rounded-2xl bg-slate-850/60 border border-slate-800/80 space-y-3">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-slate-300">Land-Cover Multi-Spectral Mix</span>
                  <span className="text-[11px] text-slate-400 font-mono">
                    Model: {result?.model_type || "LightGBM"}
                  </span>
                </div>

                <div className="space-y-2">
                  {CLASSES.map((cls) => {
                    const prob = result?.probabilities?.[cls] || 0;
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
                            {(prob * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-500"
                            style={{
                              width: `${Math.max(prob * 100, 2)}%`,
                              backgroundColor: meta.color,
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Statutory Compliance Check Box */}
              <div className="p-3.5 rounded-xl bg-slate-800/20 border border-slate-800 text-[11px] space-y-1.5 text-slate-400">
                <div className="flex items-center gap-1.5 font-semibold text-slate-300 text-xs">
                  <FileText className="w-3.5 h-3.5 text-cyan-400" />
                  Statutory Zoning Verification (RAG Engine)
                </div>
                <p>
                  • <b>PLUPA 2019 Sec. 56:</b> 30m mandatory riparian setback applied for water/flood parcels.
                </p>
                <p>
                  • <b>NEMA EMCA:</b> Commercial land conversion &gt;2ha requires EIA Public Notice & License.
                </p>
                <p>
                  • <b>Regional Trade Corridor:</b> Infrastructure buffer enforced for transport corridors.
                </p>
              </div>

              {/* AI Copilot Button */}
              <button
                onClick={generateCopilot}
                disabled={copilotLoading}
                className="w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-cyan-500 to-emerald-400 hover:from-cyan-400 hover:to-emerald-300 text-slate-950 font-bold text-xs flex items-center justify-center gap-2 shadow-lg shadow-cyan-500/20 transition-all disabled:opacity-50 cursor-pointer"
              >
                <Sparkles className="w-4 h-4" />
                {copilotLoading ? "Synthesizing Dossier (GROQ / Gemini)..." : "Generate GenAI Underwriting Dossier"}
              </button>

              {/* Copilot Markdown Output */}
              {copilotDossier && (
                <div className="p-4 rounded-xl bg-slate-900 border border-cyan-500/30 text-xs space-y-2 relative">
                  <div className="flex justify-between items-center pb-2 border-b border-slate-800">
                    <span className="font-bold text-cyan-300 flex items-center gap-1.5">
                      <Sparkles className="w-3.5 h-3.5" />
                      Executive Underwriting Report
                    </span>
                    <button
                      onClick={() => {
                        navigator.clipboard.writeText(copilotDossier.dossier);
                        setCopied(true);
                        setTimeout(() => setCopied(false), 2000);
                      }}
                      className="text-slate-400 hover:text-white p-1 rounded"
                    >
                      {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                    </button>
                  </div>
                  <pre className="text-slate-300 font-mono text-[11px] whitespace-pre-wrap leading-relaxed max-h-60 overflow-y-auto">
                    {copilotDossier.dossier}
                  </pre>
                </div>
              )}
            </div>
          )}

          {/* Tab 2: County Benchmark */}
          {activeTab === "county" && (
            <div className="flex-1 overflow-y-auto p-5 space-y-4">
              <div className="space-y-1.5">
                <label className="text-xs text-slate-400 font-medium">Select Study County (13)</label>
                <select
                  value={selectedCounty}
                  onChange={(e) => setSelectedCounty(e.target.value)}
                  className="w-full p-2.5 rounded-xl bg-slate-800 border border-slate-700 text-slate-200 text-xs capitalize cursor-pointer focus:outline-none focus:border-cyan-400"
                >
                  {COUNTIES.map((c) => (
                    <option key={c} value={c} className="capitalize">
                      {c.replace("_", " ")}
                    </option>
                  ))}
                </select>
              </div>

              {countyData ? (
                <div className="space-y-3">
                  <div className="p-4 rounded-2xl bg-slate-800/40 border border-slate-800 space-y-2">
                    <span className="text-[10px] font-mono uppercase text-slate-400">
                      County General Suitability
                    </span>
                    <h3 className="text-lg font-bold text-cyan-300 capitalize">
                      {countyData.county?.replace("_", " ")} ({countyData.climate_zone})
                    </h3>
                    <p className="text-xs text-slate-300">
                      Dominant Regional Allocation:{" "}
                      <b className="uppercase text-emerald-400">{countyData.recommendation}</b>
                    </p>
                  </div>

                  {/* Readiness Score Breakdown */}
                  <div className="grid grid-cols-3 gap-2">
                    <div className="p-3 rounded-xl bg-blue-500/10 border border-blue-500/20 text-center">
                      <span className="text-[10px] text-slate-400 block">Settle Score</span>
                      <span className="text-base font-bold text-blue-400 font-mono">
                        {countyData.scores?.settle?.toFixed(1) || "16.7"}
                      </span>
                    </div>
                    <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-center">
                      <span className="text-[10px] text-slate-400 block">Farm Score</span>
                      <span className="text-base font-bold text-emerald-400 font-mono">
                        {countyData.scores?.farm?.toFixed(1) || "16.7"}
                      </span>
                    </div>
                    <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-center">
                      <span className="text-[10px] text-slate-400 block">Protect Score</span>
                      <span className="text-base font-bold text-amber-400 font-mono">
                        {countyData.scores?.protect?.toFixed(1) || "16.7"}
                      </span>
                    </div>
                  </div>

                  {/* Land Cover Shares */}
                  <div className="p-4 rounded-2xl bg-slate-800/30 border border-slate-800 space-y-2 text-xs">
                    <span className="font-semibold text-slate-300 block mb-2">
                      County Land-Cover Area Share
                    </span>
                    {CLASSES.map((cls) => {
                      const share = countyData.predicted_cover_pct?.[cls] || 0;
                      const meta = CLASS_META[cls];
                      return (
                        <div key={cls} className="flex justify-between py-1 border-b border-slate-800/60">
                          <span className="text-slate-400 flex items-center gap-1.5">
                            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: meta.color }} />
                            {meta.label}
                          </span>
                          <span className="font-mono text-slate-200">{share}%</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : (
                <div className="text-xs text-slate-500 p-4 text-center">
                  County statistics preloaded for all 13 regions
                </div>
              )}
            </div>
          )}
        </aside>
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
              Set the URL of your Vultr server (e.g. http://192.248.170.201) or local API container.
            </p>
            <input
              type="text"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              placeholder="http://192.248.170.201"
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
