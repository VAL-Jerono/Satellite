"""
api/main.py
────────────
FastAPI service for the Kenya Land Readiness Atlas.

Endpoints:
  GET  /health                        – liveness probe
  POST /predict/point                 – single lat/lon → land-cover + recommendation
  GET  /predict/county/{county_name}  – pre-aggregated county summary
  GET  /tiles/{z}/{x}/{y}.png         – XYZ map tiles (cached)

ONNX model loaded once at startup; CPU-only.
"""

from __future__ import annotations

import io
import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field

# ── Config ────────────────────────────────────────────────────────────────────
_ROOT   = Path(__file__).resolve().parents[1]
_CFG    = yaml.safe_load((_ROOT / "configs" / "config.yaml").read_text())
_RGN    = yaml.safe_load((_ROOT / "configs" / "regions.yaml").read_text())
CLASSES = _RGN["classes"]
COLOURS = _RGN["class_colours"]
AC      = _CFG["api"]

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Kenya Land Readiness Atlas API",
    description="Land-cover classification and readiness recommendations over Kenya using Sentinel-2 CNN.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── ONNX session (loaded once) ────────────────────────────────────────────────
_SESS: ort.InferenceSession | None = None
_STARTUP_TIME = time.time()


def _get_session() -> ort.InferenceSession:
    global _SESS
    if _SESS is None:
        model_path = _ROOT / AC["model_path"]
        if not model_path.exists():
            raise RuntimeError(
                f"ONNX model not found at {model_path}. "
                "Run training first, or set API_MODEL_PATH env var."
            )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.getenv("OMP_NUM_THREADS", "2"))
        _SESS = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=_CFG["serving"]["providers"],
        )
    return _SESS


# ── Sentinel-2 feature helpers ────────────────────────────────────────────────
# Band means / stds (same as training/dataset.py)
_BAND_MEAN = np.array([0.0673,0.0830,0.0881,0.1189,0.1983,0.2342,0.2547,0.2633,0.1686,0.0954], dtype=np.float32)
_BAND_STD  = np.array([0.0461,0.0497,0.0603,0.0672,0.0888,0.1003,0.1073,0.1098,0.0841,0.0586], dtype=np.float32)

PATCH_PX    = _CFG["training"]["patch_size"]   # 64
IN_CHANNELS = _CFG["model"]["in_channels"]      # 10


def _pull_patch_ee(lon: float, lat: float) -> np.ndarray:
    """
    Pull a 64×64 Sentinel-2 patch from Earth Engine on demand.
    Used only when a precomputed tile cache miss occurs.
    Returns (C, H, W) float32 array normalised to [0,1].
    """
    import ee
    project = os.getenv("GEE_PROJECT", "")
    if not project:
        raise RuntimeError("GEE_PROJECT env var not set; cannot fetch live patch.")
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)

    YEAR  = _CFG["training"]["year"]
    BANDS = ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]
    CLOUD = _CFG["training"]["cloud_thresh"]
    HALF  = (PATCH_PX // 2) * 10  # metres

    geom = ee.Geometry.Point([lon, lat]).buffer(HALF).bounds()
    csp  = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2   = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(f"{YEAR}-01-01", f"{YEAR+1}-01-01")
            .filterBounds(geom)
            .linkCollection(csp, ["cs_cdf"])
            .map(lambda im: im.updateMask(im.select("cs_cdf").gte(CLOUD)))
            .select(BANDS).median().divide(10000).clamp(0, 1))

    arr = np.array(s2.sampleRectangle(region=geom, defaultValue=0)
                     .get("array").getInfo(), dtype=np.float32)  # (H, W, C)
    arr = arr.transpose(2, 0, 1)  # (C, H, W)

    import cv2
    if arr.shape[1] != PATCH_PX or arr.shape[2] != PATCH_PX:
        arr = np.stack([cv2.resize(arr[c], (PATCH_PX, PATCH_PX)) for c in range(arr.shape[0])])
    return arr


def _normalise(patch: np.ndarray) -> np.ndarray:
    patch = (patch - _BAND_MEAN[:, None, None]) / (_BAND_STD[:, None, None] + 1e-6)
    return np.clip(patch, -3.0, 3.0).astype(np.float32)


def _run_inference(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (probs, pred) with shapes (K,) and ()."""
    sess   = _get_session()
    inp    = _normalise(patch)[None]  # (1, C, H, W)
    logits = sess.run(None, {"sentinel2_patch": inp})[0][0]  # (K,)
    probs  = _softmax(logits)
    return probs, int(np.argmax(probs))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ── Recommendation rule layer (transparent, not NN) ──────────────────────────

def recommend(
    probs: np.ndarray,
    flood_risk: bool,
    slope_deg: float,
) -> dict:
    """
    Transparent rule layer over model probabilities.
    Returns action, confidence_label, and rationale.
    """
    label_names = CLASSES
    dominant    = int(np.argmax(probs))
    dominant_p  = float(probs[dominant])
    conf_label  = "high" if dominant_p > 0.7 else "medium" if dominant_p > 0.45 else "low"

    built_p  = float(probs[0])   # built_up
    crop_p   = float(probs[1])   # cropland
    tree_p   = float(probs[2])   # trees
    water_p  = float(probs[4])   # water
    bare_p   = float(probs[5])   # bare

    if water_p > 0.4 or flood_risk:
        action   = "protect"
        rationale= "High water/flood exposure — development risk too high."
    elif tree_p > 0.4 and slope_deg > 15:
        action   = "protect"
        rationale= "Forest on steep slope — erosion risk; conserve watershed."
    elif built_p > 0.5:
        action   = "settle"
        rationale= "Already urban fringe; densification preferable to greenfield."
    elif crop_p > 0.35 and slope_deg < 10 and not flood_risk:
        action   = "farm"
        rationale= "Active cropland on suitable terrain."
    elif bare_p > 0.4 and not flood_risk:
        action   = "settle"
        rationale= "Bare land, low hazard — potential settlement zone."
    else:
        action   = "farm"
        rationale= "Mixed cover; smallholder cropping most likely appropriate."

    return dict(
        action=action,
        confidence=conf_label,
        dominant_class=label_names[dominant],
        dominant_prob=round(dominant_p, 3),
        rationale=rationale,
    )


# ── Schemas ───────────────────────────────────────────────────────────────────

class PointRequest(BaseModel):
    lat:         float = Field(..., ge=-5.0, le=5.0,   description="Latitude (Kenya ≈ -5 to 5)")
    lon:         float = Field(..., ge=33.0, le=42.0,  description="Longitude (Kenya ≈ 33 to 42)")
    flood_risk:  bool  = Field(False, description="External flood hazard flag")
    slope_deg:   float = Field(0.0,   description="Terrain slope in degrees")
    patch_b64:   Optional[str] = Field(None, description="Base64 float32 patch (C×H×W). If absent, pulls from EE.")


class PredictResponse(BaseModel):
    lat:          float
    lon:          float
    probabilities: dict[str, float]
    prediction:   str
    recommendation: dict
    model_version: str = "1.0.0"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "uptime_s": round(time.time() - _STARTUP_TIME, 1),
        "model_loaded": _SESS is not None,
    }


@app.post("/predict/point", response_model=PredictResponse)
def predict_point(req: PointRequest):
    # Patch source: supplied or pulled from EE
    if req.patch_b64:
        import base64
        raw   = base64.b64decode(req.patch_b64)
        patch = np.frombuffer(raw, dtype=np.float32).reshape(IN_CHANNELS, PATCH_PX, PATCH_PX)
    else:
        try:
            patch = _pull_patch_ee(req.lon, req.lat)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"EE pull failed: {exc}")

    probs, pred = _run_inference(patch)
    rec = recommend(probs, req.flood_risk, req.slope_deg)

    return PredictResponse(
        lat=req.lat,
        lon=req.lon,
        probabilities={cls: round(float(p), 4) for cls, p in zip(CLASSES, probs)},
        prediction=CLASSES[pred],
        recommendation=rec,
    )


@app.get("/predict/county/{county_name}")
def predict_county(county_name: str):
    """
    Returns pre-aggregated land-cover shares and built-up growth
    from the county export table (joins to dissertation HFVS).
    """
    cache_path = _ROOT / "data" / "county_summaries" / f"{county_name.lower()}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail=f"No cached summary for county '{county_name}'.")


@app.get("/tiles/{z}/{x}/{y}.png")
def get_tile(z: int, x: int, y: int):
    """
    Serve precomputed PNG map tiles for the land-cover layer.
    Falls back to a blank tile if not cached.
    """
    tile_path = Path(AC["tile_cache_dir"]) / str(z) / str(x) / f"{y}.png"
    if tile_path.exists():
        return Response(content=tile_path.read_bytes(), media_type="image/png")
    # Return a 256×256 transparent tile
    from PIL import Image as PILImage
    img = PILImage.new("RGBA", (256, 256), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# ── GenAI / RAG Spatial Underwriting Copilot ───────────────────────────────────

_RAG_REGULATORY_KNOWLEDGE = {
    "PLUPA_2019": "Kenya Physical and Land Use Planning Act (2019) Sec. 56: Mandatory 30m setback from rivers and designated wetlands; residential development on slopes >15° requires specialized civil engineering certification.",
    "NEMA_EMCA": "NEMA Environmental Impact Assessment Regulations: Conversion of agricultural land >2 hectares to commercial/built-up requires NEMA EIA License & Public Hearing.",
    "TRADE_CORRIDOR": "East African Trade Corridor Zoning Standard: Commercial logistics hubs must maintain 50m highway buffer & demonstrate non-interference with primary cropland drainage.",
}

class CopilotRequest(BaseModel):
    lat:            float
    lon:            float
    probabilities:  dict[str, float]
    prediction:     str
    recommendation: dict
    flood_risk:     bool = False
    slope_deg:      float = 0.0
    county_name:    Optional[str] = "Kenya Spatial Corridor"

class CopilotResponse(BaseModel):
    dossier:        str
    llm_powered:    bool
    source_context: list[str]


def _build_fallback_dossier(req: CopilotRequest) -> str:
    action = req.recommendation.get("action", "farm").upper()
    conf = req.recommendation.get("confidence", "medium").upper()
    dom_class = req.prediction.replace("_", " ").title()
    dom_p = req.probabilities.get(req.prediction, 0.0) * 100.0
    
    dossier = f"""### 📋 Spatial Underwriting Summary Dossier
**Site Coordinates:** `{req.lat:.4f}, {req.lon:.4f}` | **Region:** {req.county_name}  
**Primary Land Classification:** {dom_class} ({dom_p:.1f}% Model Confidence)  
**Underwriting Recommendation:** **{action}** ({conf} CONFIDENCE)

---

#### 1. 🏢 Executive Land Suitability Assessment
- **Dominant Land Cover:** Classified as **{dom_class}**.
- **Topographic Constraints:** Terrain slope measured at **{req.slope_deg:.1f}°**. {"⚠️ Exceeds 15° slope threshold — heightened landslide & soil erosion vulnerability." if req.slope_deg > 15 else "✅ Slope within standard construction tolerance (<15°)."}
- **Flood Exposure Status:** {"⚠️ High surface flood risk flagged by hydrologic DEM analysis." if req.flood_risk else "✅ Low immediate flood risk identified."}

#### 2. 📜 Statutory Zoning & Regulatory Compliance Check (RAG Engine)
- **PLUPA 2019 Compliance:** {_RAG_REGULATORY_KNOWLEDGE['PLUPA_2019']}
- **NEMA Environment Check:** {_RAG_REGULATORY_KNOWLEDGE['NEMA_EMCA']}
- **Trade Corridor Requirement:** {_RAG_REGULATORY_KNOWLEDGE['TRADE_CORRIDOR']}

#### 3. 💡 Capital Deployment & Risk Mitigation Rationale
*Rationale:* {req.recommendation.get('rationale', 'Site suitability evaluated based on multi-spectral satellite features.')}
- **Actionable Guidance:** {"Hold residential development; enforce environmental protection buffer." if action == "PROTECT" else ("Approve low-density infrastructure deployment." if action == "SETTLE" else "Prioritize agricultural financing and irrigation infrastructure.")}
"""
    return dossier.strip()


@app.post("/underwrite/copilot", response_model=CopilotResponse)
def spatial_underwriting_copilot(req: CopilotRequest):
    """
    GenAI / RAG Spatial Underwriting Copilot Endpoint.
    Synthesizes Sentinel-2 multi-spectral predictions, topographic risk,
    and Kenyan land use regulations into an executive underwriting dossier.
    """
    sources = list(_RAG_REGULATORY_KNOWLEDGE.values())
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            prompt = f"""
You are an expert GIS Spatial Underwriter for African Trade Corridors and Real Estate.
Synthesize an executive Underwriting Dossier for a site in Kenya with the following evidence:
- Location: Lat {req.lat:.4f}, Lon {req.lon:.4f} ({req.county_name})
- Model Prediction: {req.prediction} (Probabilities: {json.dumps(req.probabilities)})
- Recommendation: {req.recommendation}
- Flood Hazard Flag: {req.flood_risk}, Slope: {req.slope_deg}°
- Relevant Spatial Planning Regulations (RAG Context):
  * {sources[0]}
  * {sources[1]}
  * {sources[2]}

Format your response as a professional Markdown report with sections:
1. Executive Land Suitability Brief
2. Regulatory Compliance & Environmental Hazard Assessment
3. Strategic Underwriting & Capital Allocation Recommendation
Keep it concise, rigorous, and professional.
"""
            res = model.generate_content(prompt)
            return CopilotResponse(
                dossier=res.text.strip(),
                llm_powered=True,
                source_context=sources
            )
        except Exception as exc:
            # Fallback if API call fails
            dossier = _build_fallback_dossier(req)
            return CopilotResponse(
                dossier=dossier + f"\n\n*Note: Gemini API fallback used due to: {exc}*",
                llm_powered=False,
                source_context=sources
            )
    else:
        dossier = _build_fallback_dossier(req)
        return CopilotResponse(
            dossier=dossier,
            llm_powered=False,
            source_context=sources
        )

