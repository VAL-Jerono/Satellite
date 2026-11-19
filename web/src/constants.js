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
  { name: "Nyeri (Highland)", lat: -0.4201, lon: 36.9476 },
  { name: "Nakuru (Rift Valley)", lat: -0.3031, lon: 36.0800 },
  { name: "Kisumu (Lake Basin)", lat: -0.0917, lon: 34.7680 },
  { name: "Kilifi (Coast)", lat: -3.6305, lon: 39.8499 },
  { name: "Turkana (ASAL)", lat: 3.1167, lon: 35.6000 },
  { name: "Nairobi (Urban)", lat: -1.2921, lon: 36.8219 },
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

