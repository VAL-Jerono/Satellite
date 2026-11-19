export const DEFAULT_API_URL =
  import.meta.env.VITE_API_URL || "http://192.248.170.201";

export const CLASSES = [
  "built_up",
  "cropland",
  "trees",
  "grass_shrub",
  "water",
  "bare",
];

export const CLASS_META = {
  built_up: { label: "Built-up / Urban", color: "#E05A3A", border: "border-red-500/30", bg: "bg-red-500/10" },
  cropland: { label: "Cropland / Agriculture", color: "#F5D76E", border: "border-yellow-500/30", bg: "bg-yellow-500/10" },
  trees: { label: "Tree Canopy / Forest", color: "#27AE60", border: "border-emerald-500/30", bg: "bg-emerald-500/10" },
  grass_shrub: { label: "Grassland & Shrub", color: "#A9D18E", border: "border-lime-500/30", bg: "bg-lime-500/10" },
  water: { label: "Open Water & Wetlands", color: "#2980B9", border: "border-blue-500/30", bg: "bg-blue-500/10" },
  bare: { label: "Bare Ground / Rock", color: "#C8A96E", border: "border-amber-500/30", bg: "bg-amber-500/10" },
};

export const COUNTIES = [
  "nakuru",
  "highland",
  "muranga",
  "nyandarua",
  "coast",
  "kwale",
  "lamu",
  "arid",
  "turkana",
  "wajir",
  "lake_basin",
  "siaya",
  "kajiado",
];

export const PRESETS = [
  { name: "Nyeri (Highland)", lat: -0.4201, lon: 36.9476, zone: "highland_zone" },
  { name: "Nakuru (Rift Valley)", lat: -0.3031, lon: 36.0800, zone: "rift_zone" },
  { name: "Kisumu (Lake Basin)", lat: -0.0917, lon: 34.7680, zone: "lake_zone" },
  { name: "Kilifi (Coast)", lat: -3.6305, lon: 39.8499, zone: "coastal_zone" },
  { name: "Turkana (ASAL)", lat: 3.1167, lon: 35.6000, zone: "asal_zone" },
  { name: "Nairobi (Urban)", lat: -1.2921, lon: 36.8219, zone: "rift_zone" },
];

export const STUDY_REGIONS = [
  { name: "Central Highlands", bbox: [36.30, -0.60, 36.80, -0.10], zone: "highland_zone" },
  { name: "Murang'a", bbox: [36.90, -0.90, 37.40, -0.40], zone: "highland_zone" },
  { name: "Nyandarua", bbox: [36.20, -0.80, 36.70, -0.20], zone: "highland_zone" },
  { name: "Coastal Strip (Kilifi)", bbox: [39.40, -3.90, 39.90, -3.40], zone: "coastal_zone" },
  { name: "Kwale", bbox: [39.10, -4.40, 39.65, -3.80], zone: "coastal_zone" },
  { name: "Lamu", bbox: [40.70, -2.40, 41.30, -1.80], zone: "coastal_zone" },
  { name: "Marsabit (Arid)", bbox: [37.50, 1.50, 38.00, 2.00], zone: "asal_zone" },
  { name: "Turkana", bbox: [35.20, 3.20, 35.80, 3.80], zone: "asal_zone" },
  { name: "Wajir", bbox: [40.20, 1.60, 40.80, 2.20], zone: "asal_zone" },
  { name: "Lake Basin (Kisumu)", bbox: [34.60, -0.30, 35.10, 0.20], zone: "lake_zone" },
  { name: "Siaya", bbox: [34.00, -0.30, 34.60, 0.30], zone: "lake_zone" },
  { name: "Nakuru", bbox: [35.80, -0.60, 36.30, -0.10], zone: "rift_zone" },
  { name: "Kajiado", bbox: [36.60, -2.00, 37.20, -1.40], zone: "rift_zone" },
];

export const ZONE_COLORS = {
  highland_zone: "#22c55e",   // green
  coastal_zone:  "#38bdf8",   // sky blue
  asal_zone:     "#f97316",   // orange
  lake_zone:     "#06b6d4",   // cyan
  rift_zone:     "#a855f7",   // purple
};

export const REGION_POINTS = {
  "Central Highlands": [
    { name: "Nyeri Town", lat: -0.4201, lon: 36.9476 },
    { name: "Mweiga", lat: -0.2667, lon: 36.9833 },
    { name: "Karatina", lat: -0.4833, lon: 37.1333 },
  ],
  "Murang'a": [
    { name: "Murang'a Town", lat: -0.7167, lon: 37.1500 },
    { name: "Kangema", lat: -0.6167, lon: 36.9833 },
    { name: "Maragua", lat: -0.7333, lon: 37.1333 },
  ],
  "Nyandarua": [
    { name: "Ol Kalou", lat: -0.2667, lon: 36.3833 },
    { name: "Engineer", lat: -0.4000, lon: 36.4833 },
    { name: "Kinangop", lat: -0.6333, lon: 36.6167 },
  ],
  "Coastal Strip (Kilifi)": [
    { name: "Kilifi Town", lat: -3.6305, lon: 39.8499 },
    { name: "Watamu", lat: -3.3667, lon: 39.9833 },
    { name: "Malindi", lat: -3.2167, lon: 40.1167 },
  ],
  "Kwale": [
    { name: "Kwale Town", lat: -4.1733, lon: 39.4522 },
    { name: "Diani", lat: -4.3167, lon: 39.5667 },
    { name: "Shimba Hills", lat: -4.2333, lon: 39.4167 },
  ],
  "Lamu": [
    { name: "Lamu Town", lat: -2.2694, lon: 40.9020 },
    { name: "Mpeketoni", lat: -2.1333, lon: 40.7000 },
    { name: "Witu", lat: -2.3833, lon: 40.4333 },
  ],
  "Marsabit (Arid)": [
    { name: "Marsabit Town", lat: 2.3333, lon: 37.9833 },
    { name: "Laisamis", lat: 1.6167, lon: 37.8000 },
    { name: "Moyale", lat: 3.5167, lon: 39.0500 },
  ],
  "Turkana": [
    { name: "Lodwar", lat: 3.1167, lon: 35.6000 },
    { name: "Lokichar", lat: 2.4167, lon: 35.6667 },
    { name: "Kalokol", lat: 3.5167, lon: 35.8333 },
  ],
  "Wajir": [
    { name: "Wajir Town", lat: 1.7500, lon: 40.0667 },
    { name: "Habaswein", lat: 1.0167, lon: 39.4833 },
    { name: "El Wak", lat: 2.8000, lon: 40.9333 },
  ],
  "Lake Basin (Kisumu)": [
    { name: "Kisumu City", lat: -0.0917, lon: 34.7680 },
    { name: "Ahero", lat: -0.1667, lon: 34.9167 },
    { name: "Muhoroni", lat: -0.1667, lon: 35.1833 },
  ],
  "Siaya": [
    { name: "Siaya Town", lat: 0.0603, lon: 34.2878 },
    { name: "Yala", lat: 0.0833, lon: 34.5333 },
    { name: "Bondo", lat: -0.1667, lon: 34.2667 },
  ],
  "Nakuru": [
    { name: "Nakuru CBD", lat: -0.3031, lon: 36.0800 },
    { name: "Naivasha", lat: -0.7167, lon: 36.4333 },
    { name: "Gilgil", lat: -0.5000, lon: 36.3167 },
  ],
  "Kajiado": [
    { name: "Kajiado Town", lat: -1.8522, lon: 36.7764 },
    { name: "Ngong", lat: -1.3667, lon: 36.6500 },
    { name: "Namanga", lat: -2.5500, lon: 36.7833 },
  ],
};
