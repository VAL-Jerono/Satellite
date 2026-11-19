import React, { useState, useEffect, useRef } from "react";
import L from "leaflet";
import {
  DEFAULT_API_URL,
  CLASSES,
  CLASS_META,
  COUNTIES,
  PRESETS,
  STUDY_REGIONS,
  ZONE_COLORS,
  REGION_POINTS,
} from "./constants";
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
} from "lucide-react";

// ── Kenya National Boundary (simplified polygon) ───────────────────────────
const KENYA_BOUNDARY = {
  type: "Feature",
  properties: { name: "Kenya" },
  geometry: {
    type: "Polygon",
    coordinates: [[
      [33.9, 4.6], [35.3, 4.7], [36.9, 4.4], [38.1, 3.9],
      [39.5, 3.4], [41.0, 3.1], [41.9, 2.0], [41.5, 0.0],
      [41.0, -1.6], [40.1, -2.5], [39.2, -4.7], [37.6, -4.5],
      [35.8, -3.1], [34.0, -1.0], [33.9, 0.5], [34.1, 1.8],
      [34.6, 3.5], [33.9, 4.6],
    ]],
  },
};

export default function App() {
  const [apiUrl, setApiUrl] = useState(
    localStorage.getItem("atlas_api_url") || DEFAULT_API_URL
  );
  const [apiOnline, setApiOnline] = useState(null);
  const [apiPing, setApiPing] = useState(null);
  const [selectedCoords, setSelectedCoords] = useState({ lat: -0.3031, lon: 36.08 });
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

  const mapRef = useRef(null);
  const mapInstance = useRef(null);
  const markerRef = useRef(null);

  // ── 1. Check API Health ────────────────────────────────────────────────────
  const checkHealth = async () => {
    const start = performance.now();
    try {
      const res = await fetch(`${apiUrl}/health`, { method: "GET" });
      const duration = Math.round(performance.now() - start);
      if (res.ok) {
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

    // Esri Satellite Basemap
    const esriSatellite = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      {
        attribution: "Esri World Imagery",
        maxZoom: 18,
      }
    ).addTo(map);

    // Esri Topo Basemap
    const esriTopo = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
      {
        attribution: "Esri Topo",
        maxZoom: 18,
      }
    );

    L.control
      .layers(
        { "🛰️ Satellite (Esri)": esriSatellite, "🗺️ Topographic": esriTopo },
        {},
        { position: "topright" }
      )
      .addTo(map);

    // ── Kenya National Boundary (always visible, not in layer switcher) ──────
    L.geoJSON(KENYA_BOUNDARY, {
      style: {
        color: "#00ff88",
        weight: 2,
        fillOpacity: 0,
        dashArray: "6,4",
      },
    }).addTo(map);

    // ── Study Regions: colored rectangles + permanent labels + circle markers ─
    STUDY_REGIONS.forEach((r) => {
      const [minLon, minLat, maxLon, maxLat] = r.bbox;
      const zoneColor = ZONE_COLORS[r.zone] || "#06b6d4";

      // Colored rectangle
      L.rectangle(
        [
          [minLat, minLon],
          [maxLat, maxLon],
        ],
        {
          color: zoneColor,
          weight: 1.5,
          fillColor: zoneColor,
          fillOpacity: 0.15,
          dashArray: "4, 4",
        }
      )
        .bindTooltip(`<b>${r.name}</b><br/><span style="color:${zoneColor}">${r.zone.replace("_zone", "").toUpperCase()}</span>`, {
          permanent: true,
          direction: "center",
          className: "custom-tooltip region-label",
        })
        .addTo(map);

      // Prediction circle markers for each point in this region
      const points = REGION_POINTS[r.name] || [];
      points.forEach((pt) => {
        const cm = L.circleMarker([pt.lat, pt.lon], {
          radius: 7,
          color: "white",
          weight: 1.5,
          fillColor: zoneColor,
          fillOpacity: 0.9,
        });

        cm.bindPopup(
          `<b>${pt.name}</b><br/><span style="color:${zoneColor}">${r.zone.replace("_zone", "").toUpperCase()}</span><br/>Click to analyze`
        );

        cm.on("click", () => {
          map.flyTo([pt.lat, pt.lon], 11, { duration: 1.2 });
          setSelectedCoords({ lat: pt.lat, lon: pt.lon });
        });

        cm.addTo(map);
      });
    });

    // Custom pulsing pin icon
    const customIcon = L.divIcon({
      className: "custom-pin",
      html: `
        <div class="relative flex items-center justify-center">
          <div class="absolute w-8 h-8 bg-cyan-400/30 rounded-full animate-ping"></div>
          <div class="w-4 h-4 bg-cyan-400 border-2 border-white rounded-full shadow-lg shadow-cyan-500/50"></div>
        </div>
      `,
      iconSize: [24, 24],
      iconAnchor: [12, 12],
    });

    // Initial marker
    const marker = L.marker([selectedCoords.lat, selectedCoords.lon], {
      icon: customIcon,
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

  // Sync marker when coordinates change
  useEffect(() => {
    if (markerRef.current) {
      markerRef.current.setLatLng([selectedCoords.lat, selectedCoords.lon]);
    }
    runInference(selectedCoords.lat, selectedCoords.lon, floodRisk, slopeDeg);
  }, [selectedCoords]);

  // ── 3. Run Inference API ───────────────────────────────────────────────────
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
    } catch (err) {
      console.error("Inference call failed:", err);
    } finally {
      setLoading(false);
    }
  };

  // ── 4. Fetch County Summary ────────────────────────────────────────────────
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

  // ── 5. Generate GenAI Copilot Dossier ──────────────────────────────────────
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
      }
    } catch (err) {
      console.error("Copilot error:", err);
    } finally {
      setCopilotLoading(false);
    }
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
                AI Spatial Engine
              </span>
            </div>
            <p className="text-[11px] text-slate-400">
              Sentinel-2 Multi-Spectral Analysis · Zone-Stratified LightGBM + Spectral CNN
            </p>
          </div>
        </div>

        {/* Live Status & Controls */}
        <div className="flex items-center gap-4">
          <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800/60 border border-slate-700/50 text-xs">
            <Server className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-slate-400">Backend:</span>
            <span className="font-mono text-slate-200">
              {apiUrl.replace("http://", "").replace(":8000", "")}
            </span>
            <div className="flex items-center gap-1.5 pl-2 border-l border-slate-700">
              <span
                className={`w-2 h-2 rounded-full ${
                  apiOnline ? "bg-emerald-400 animate-pulse" : "bg-red-400"
                }`}
              />
              <span className={apiOnline ? "text-emerald-400 font-mono text-[11px]" : "text-red-400 text-[11px]"}>
                {apiOnline ? `${apiPing}ms` : "Offline"}
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

          {/* Quick Coordinate Presets Pill Bar — zone-colored pills */}
          <div className="absolute top-4 left-4 z-[1000] flex flex-wrap gap-1.5 p-1.5 bg-slate-900/80 backdrop-blur-md rounded-xl border border-slate-800/80 shadow-2xl">
            <span className="text-[11px] text-slate-400 flex items-center px-2 font-medium">
              Presets:
            </span>
            {PRESETS.map((p) => {
              const zoneColor = ZONE_COLORS[p.zone] || "#06b6d4";
              const isActive = selectedCoords.lat === p.lat;
              return (
                <button
                  key={p.name}
                  onClick={() => {
                    setSelectedCoords({ lat: p.lat, lon: p.lon });
                    if (mapInstance.current) {
                      mapInstance.current.flyTo([p.lat, p.lon], 10, { duration: 1.2 });
                    }
                  }}
                  className={`text-xs px-2.5 py-1 rounded-lg transition-all flex items-center gap-1.5 ${
                    isActive
                      ? "text-slate-950 font-bold shadow-md"
                      : "bg-slate-800/60 hover:bg-slate-800 text-slate-300"
                  }`}
                  style={
                    isActive
                      ? { backgroundColor: zoneColor, boxShadow: `0 2px 10px ${zoneColor}50` }
                      : { borderColor: `${zoneColor}60`, border: `1px solid ${zoneColor}60` }
                  }
                >
                  <span
                    className="w-2 h-2 rounded-full shrink-0"
                    style={{ backgroundColor: zoneColor }}
                  />
                  {p.name}
                </button>
              );
            })}
          </div>

          {/* Map Footer Helper */}
          <div className="absolute bottom-4 left-4 z-[1000] bg-slate-900/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-800 text-[11px] text-slate-400 flex items-center gap-2">
            <MapPin className="w-3.5 h-3.5 text-cyan-400" />
            Click anywhere in Kenya to run instant multi-spectral inference
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
                    onChange={(e) => setSlopeDeg(parseFloat(e.target.value))}
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
                    onChange={(e) => setFloodRisk(e.target.checked)}
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
                    {result?.recommendation?.confidence?.toUpperCase()} CONFIDENCE
                  </span>
                </div>

                <div className="flex items-center gap-3">
                  {actionStyle.icon}
                  <h2 className={`text-xl font-extrabold tracking-tight ${actionStyle.text}`}>
                    {actionStyle.label}
                  </h2>
                </div>

                <p className="text-xs text-slate-300 leading-relaxed pt-1">
                  {result?.recommendation?.rationale}
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
                              width: `${prob * 100}%`,
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
                  • <b>NEMA EMCA:</b> Commercial land conversion &gt;2ha requires EIA Public Notice.
                </p>
              </div>

              {/* AI Copilot Button */}
              <button
                onClick={generateCopilot}
                disabled={copilotLoading}
                className="w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-cyan-500 to-emerald-400 hover:from-cyan-400 hover:to-emerald-300 text-slate-950 font-bold text-xs flex items-center justify-center gap-2 shadow-lg shadow-cyan-500/20 transition-all disabled:opacity-50"
              >
                <Sparkles className="w-4 h-4" />
                {copilotLoading ? "Synthesizing Dossier..." : "Generate GenAI Underwriting Dossier"}
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
                <div className="text-xs text-slate-500 p-4 text-center">Loading county data...</div>
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
              Set the URL of your Vultr server or local API container.
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
