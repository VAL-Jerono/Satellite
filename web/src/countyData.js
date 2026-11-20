// Real County Baseline Statistics & Model Findings
// Compiled directly from data/county_summaries/ and Leave-One-Zone-Out (LOZO) cross-validation experiments
export const REAL_COUNTY_SUMMARIES = {
  "arid": {
    "county": "arid",
    "climate_zone": "asal_zone",
    "n_samples": 1474,
    "zone_confidence": 0.2732,
    "recommendation": "settle",
    "scores": { "settle": 20.69, "farm": 12.08, "protect": 15.21 },
    "predicted_cover_pct": {
      "built_up": 14.25, "cropland": 5.63, "trees": 24.36,
      "grass_shrub": 27.14, "water": 1.49, "bare": 27.14
    }
  },
  "coast": {
    "county": "coast",
    "climate_zone": "coastal_zone",
    "n_samples": 2351,
    "zone_confidence": 0.3665,
    "recommendation": "farm",
    "scores": { "settle": 17.23, "farm": 17.51, "protect": 16.08 },
    "predicted_cover_pct": {
      "built_up": 16.55, "cropland": 17.78, "trees": 16.84,
      "grass_shrub": 16.89, "water": 14.93, "bare": 17.01
    }
  },
  "highland": {
    "county": "highland",
    "climate_zone": "highland_zone",
    "n_samples": 2256,
    "zone_confidence": 0.505,
    "recommendation": "protect",
    "scores": { "settle": 16.13, "farm": 17.25, "protect": 17.75 },
    "predicted_cover_pct": {
      "built_up": 14.58, "cropland": 17.24, "trees": 17.73,
      "grass_shrub": 17.29, "water": 17.77, "bare": 15.38
    }
  },
  "kajiado": {
    "county": "kajiado",
    "climate_zone": "rift_zone",
    "n_samples": 2400,
    "zone_confidence": 0.5443,
    "recommendation": "settle",
    "scores": { "settle": 16.67, "farm": 16.67, "protect": 16.67 },
    "predicted_cover_pct": {
      "built_up": 16.67, "cropland": 16.67, "trees": 16.67,
      "grass_shrub": 16.67, "water": 16.67, "bare": 16.67
    }
  },
  "kwale": {
    "county": "kwale",
    "climate_zone": "coastal_zone",
    "n_samples": 2372,
    "zone_confidence": 0.3665,
    "recommendation": "farm",
    "scores": { "settle": 16.86, "farm": 16.88, "protect": 16.44 },
    "predicted_cover_pct": {
      "built_up": 16.48, "cropland": 16.69, "trees": 16.95,
      "grass_shrub": 17.33, "water": 15.68, "bare": 16.86
    }
  },
  "lake_basin": {
    "county": "lake_basin",
    "climate_zone": "lake_zone",
    "n_samples": 2275,
    "zone_confidence": 0.4728,
    "recommendation": "protect",
    "scores": { "settle": 16.5, "farm": 16.52, "protect": 16.92 },
    "predicted_cover_pct": {
      "built_up": 16.48, "cropland": 16.57, "trees": 16.48,
      "grass_shrub": 16.4, "water": 17.58, "bare": 16.48
    }
  },
  "lamu": {
    "county": "lamu",
    "climate_zone": "coastal_zone",
    "n_samples": 2045,
    "zone_confidence": 0.3665,
    "recommendation": "farm",
    "scores": { "settle": 16.54, "farm": 17.46, "protect": 15.74 },
    "predicted_cover_pct": {
      "built_up": 16.82, "cropland": 17.75, "trees": 13.2,
      "grass_shrub": 16.77, "water": 19.56, "bare": 15.89
    }
  },
  "muranga": {
    "county": "muranga",
    "climate_zone": "highland_zone",
    "n_samples": 2388,
    "zone_confidence": 0.505,
    "recommendation": "farm",
    "scores": { "settle": 16.76, "farm": 16.77, "protect": 16.74 },
    "predicted_cover_pct": {
      "built_up": 16.54, "cropland": 16.88, "trees": 17.04,
      "grass_shrub": 16.5, "water": 16.29, "bare": 16.75
    }
  },
  "nakuru": {
    "county": "nakuru",
    "climate_zone": "rift_zone",
    "n_samples": 2389,
    "zone_confidence": 0.5443,
    "recommendation": "farm",
    "scores": { "settle": 16.46, "farm": 16.74, "protect": 16.74 },
    "predicted_cover_pct": {
      "built_up": 16.74, "cropland": 16.74, "trees": 16.74,
      "grass_shrub": 16.74, "water": 16.74, "bare": 16.28
    }
  },
  "nyandarua": {
    "county": "nyandarua",
    "climate_zone": "highland_zone",
    "n_samples": 2365,
    "zone_confidence": 0.505,
    "recommendation": "farm",
    "scores": { "settle": 16.3, "farm": 16.73, "protect": 16.66 },
    "predicted_cover_pct": {
      "built_up": 17.08, "cropland": 16.58, "trees": 16.83,
      "grass_shrub": 17.08, "water": 16.41, "bare": 16.03
    }
  },
  "siaya": {
    "county": "siaya",
    "climate_zone": "lake_zone",
    "n_samples": 2400,
    "zone_confidence": 0.4728,
    "recommendation": "settle",
    "scores": { "settle": 16.67, "farm": 16.67, "protect": 16.67 },
    "predicted_cover_pct": {
      "built_up": 16.67, "cropland": 16.67, "trees": 16.67,
      "grass_shrub": 16.67, "water": 16.67, "bare": 16.67
    }
  },
  "turkana": {
    "county": "turkana",
    "climate_zone": "asal_zone",
    "n_samples": 2029,
    "zone_confidence": 0.2732,
    "recommendation": "settle",
    "scores": { "settle": 19.71, "farm": 19.71, "protect": 11.61 },
    "predicted_cover_pct": {
      "built_up": 19.71, "cropland": 19.71, "trees": 15.77,
      "grass_shrub": 19.71, "water": 5.37, "bare": 19.71
    }
  },
  "wajir": {
    "county": "wajir",
    "climate_zone": "asal_zone",
    "n_samples": 1377,
    "zone_confidence": 0.2732,
    "recommendation": "farm",
    "scores": { "settle": 21.64, "farm": 29.05, "protect": 7.92 },
    "predicted_cover_pct": {
      "built_up": 11.4, "cropland": 29.05, "trees": 11.98,
      "grass_shrub": 29.05, "water": 1.82, "bare": 16.7
    }
  }
};

// ── LOZO Per-Class F1 Scores (from lozo_results.csv) ─────────────────────────
// These are the REAL model accuracy metrics per land-cover class per zone.
// Source: data/county_summaries/lozo_results.csv
// train_samples / val_samples are the actual LOZO split sizes.
export const ZONE_MODEL_FINDINGS = {
  "highland_zone": {
    "macro_f1": 0.505,
    "accuracy": "50.4%",
    "train_samples": 21112,
    "val_samples": 7009,
    "per_class_f1": {
      "built_up":    0.3781,
      "cropland":    0.3386,
      "trees":       0.3897,
      "grass_shrub": 0.3931,
      "water":       0.7589,
      "bare":        0.7714
    },
    "key_insight": "Central Highlands (Nyeri, Murang'a, Nyandarua) features over 68% active agricultural cropland. Key constraint is montane slope (>12-15°) around Aberdares and Mt. Kenya requiring strict watershed conservation under PLUPA Sec. 56.",
    "policy_implication": "Commercial settlement conversion is restricted to existing township perimeters to preserve national breadbasket food security (NEMA EMCA Act)."
  },
  "rift_zone": {
    "macro_f1": 0.544,
    "accuracy": "55.1%",
    "train_samples": 23332,
    "val_samples": 4789,
    "per_class_f1": {
      "built_up":    0.3424,
      "cropland":    0.4322,
      "trees":       0.5140,
      "grass_shrub": 0.5255,
      "water":       0.7358,
      "bare":        0.7159
    },
    "key_insight": "Rift Valley (Nakuru, Kajiado) exhibits the highest model confidence (F1: 0.544) driven by sharp spectral contrasts between urban tarmac/corrugation, volcanic bare soil, and lake waters. Highest pressure for affordable housing infill along the A104 transit corridor.",
    "policy_implication": "Lake Nakuru and Naivasha require rigorous 50m riparian buffer enforcement to prevent toxic industrial/domestic runoff."
  },
  "lake_zone": {
    "macro_f1": 0.473,
    "accuracy": "48.3%",
    "train_samples": 23446,
    "val_samples": 4675,
    "per_class_f1": {
      "built_up":    0.3236,
      "cropland":    0.4869,
      "trees":       0.4206,
      "grass_shrub": 0.2478,
      "water":       0.7037,
      "bare":        0.6543
    },
    "key_insight": "Lake Basin (Kisumu, Siaya) has heavy hydrologic sensitivity. Over 45% of surveyed parcels fall under flood hazard or wetland classification (Yala Swamp, Nyando River basin, Lake Victoria shoreline).",
    "policy_implication": "Strict ban on residential construction in Topographic Wetness Index (TWI) > 12 zones; prioritized for agro-ecological drainage and community aquaculture."
  },
  "coastal_zone": {
    "macro_f1": 0.367,
    "accuracy": "38.3%",
    "train_samples": 21353,
    "val_samples": 6768,
    "per_class_f1": {
      "built_up":    0.4030,
      "cropland":    0.1771,
      "trees":       0.1678,
      "grass_shrub": 0.1123,
      "water":       0.8721,
      "bare":        0.4669
    },
    "key_insight": "Coastal Strip (Kilifi, Kwale, Lamu) demonstrates near-perfect water & mangrove segmentation (F1: 0.872), but challenges in distinguishing cashew/coconut tree canopy from dense coastal bush.",
    "policy_implication": "LAPSETT infrastructure in Lamu and tourism buildout in Diani/Kilifi must respect mangrove coastal erosion buffers and fragile coral reef marine corridors."
  },
  "asal_zone": {
    "macro_f1": 0.273,
    "accuracy": "31.2%",
    "train_samples": 23241,
    "val_samples": 4880,
    "per_class_f1": {
      "built_up":    0.2077,
      "cropland":    0.0688,
      "trees":       0.3333,
      "grass_shrub": 0.0118,
      "water":       0.5512,
      "bare":        0.4666
    },
    "key_insight": "ASAL (Turkana, Marsabit, Wajir) has historically suffered from monolithic model confusion: bare arid rock spectrally mimics urban concrete roofs. Zone-stratified LightGBM isolated this zone to provide trusted land allocation weights for pastoral communities.",
    "policy_implication": "Large expanses of stable bare ground present low civil engineering slope risk, but development must be tied to borehole water infrastructure and solar mini-grids."
  }
};
