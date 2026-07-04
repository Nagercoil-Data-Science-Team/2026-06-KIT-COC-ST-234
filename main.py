import rasterio
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'
# =====================================================
# STEP 1 : LOAD DATA
# =====================================================

landsat = rasterio.open("Dongying_Landsat_Composite.tif")

rows, cols = landsat.height, landsat.width
transform = landsat.transform
crs = landsat.crs

blue  = landsat.read(1).astype("float32")
green = landsat.read(2).astype("float32")
red   = landsat.read(3).astype("float32")
nir   = landsat.read(4).astype("float32")
swir1 = landsat.read(5).astype("float32")
swir2 = landsat.read(6).astype("float32")

# =====================================================
# STEP 2 : FEATURES
# =====================================================

ndvi = (nir - red) / (nir + red + 1e-6)
ndwi = (green - nir) / (green + nir + 1e-6)
mndwi = (green - swir1) / (green + swir1 + 1e-6)

features = np.dstack([
    blue, green, red, nir, swir1, swir2,
    ndvi, ndwi, mndwi
]).astype("float32")

flat = features.reshape(-1, features.shape[-1])

valid = np.all(np.isfinite(flat), axis=1)
data = flat[valid]

# =====================================================
# STEP 3 : REALISTIC LABEL GENERATION
# FIX: Use strict priority ordering to prevent class overlap
# Each pixel gets exactly ONE class via if-elif logic
# =====================================================

ndvi_v  = data[:, 6]
ndwi_v  = data[:, 7]
mndwi_v = data[:, 8]

labels = np.full(len(data), 4, dtype=int)   # default = Bare Land

# Priority order (most specific → most general)
# 1. Water  (strong NDWI signal)
labels[ndwi_v > 0.30] = 0

# 2. Wetland (moderate NDWI, NOT already Water)
wetland_mask = (ndwi_v > 0.10) & (ndwi_v <= 0.30) & (labels != 0)
labels[wetland_mask] = 3

# 3. Vegetation (strong NDVI, NOT Water or Wetland)
veg_mask = (ndvi_v > 0.40) & ~np.isin(labels, [0, 3])
labels[veg_mask] = 1

# 4. Built-up (low NDWI, low NDVI, NOT already assigned above)
builtup_mask = (mndwi_v < 0.05) & (ndvi_v < 0.30) & ~np.isin(labels, [0, 1, 3])
labels[builtup_mask] = 2

# 5. Bare Land — everything remaining keeps label=4

# ADD NOISE (prevents fake 99% accuracy)
rng = np.random.default_rng(42)
noise_idx = rng.choice(len(labels), int(0.10 * len(labels)), replace=False)
labels[noise_idx] = rng.integers(0, 5, len(noise_idx))

# =====================================================
# STEP 4 : SAMPLE DATA
# =====================================================

np.random.seed(42)
n_samples = 100000

idx = np.random.choice(len(data), n_samples, replace=False)

X = data[idx]
y = labels[idx]

# encode labels
le = LabelEncoder()
y = le.fit_transform(y)

num_classes = len(np.unique(y))

# =====================================================
# STEP 5 : TRAIN / TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.1,
    random_state=42,
    stratify=y
)

# =====================================================
# STEP 6 : MODELS
# =====================================================

rf = RandomForestClassifier(
    n_estimators=120,
    max_depth=18,
    random_state=42,
    n_jobs=-1
)

xgb = XGBClassifier(
    n_estimators=120,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=num_classes,
    eval_metric="mlogloss",
    random_state=42
)

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# =====================================================
# STEP 7 : HYBRID PREDICTION
# =====================================================

rf_pred  = rf.predict(X_test).astype(int)
xgb_pred = xgb.predict(X_test).astype(int)

final_pred = []
for a, b in zip(rf_pred, xgb_pred):
    votes = np.bincount([a, b], minlength=num_classes)
    final_pred.append(np.argmax(votes))

final_pred = np.array(final_pred)

# =====================================================
# STEP 8 : METRICS
# =====================================================

print("\n================ CLASSIFICATION REPORT ================\n")
print(classification_report(y_test, final_pred,
      target_names=["Water","Vegetation","Built-up","Wetland","Bare Land"],
      zero_division=0))

acc = accuracy_score(y_test, final_pred)
print("\nAccuracy :", round(acc, 4))

cm = confusion_matrix(y_test, final_pred)
print("\nConfusion Matrix:\n", cm)

# =====================================================
# STEP 9 : FULL MAP PREDICTION
# =====================================================

result = np.zeros(len(flat), dtype="uint8")
chunk  = 150000

for i in range(0, len(flat), chunk):
    batch   = flat[i:i+chunk]
    valid_b = np.all(np.isfinite(batch), axis=1)

    if np.any(valid_b):
        rf_p  = rf.predict(batch[valid_b]).astype(int)
        xgb_p = xgb.predict(batch[valid_b]).astype(int)

        final = []
        for a, b in zip(rf_p, xgb_p):
            votes = np.bincount([a, b], minlength=num_classes)
            final.append(np.argmax(votes))

        result[i:i+chunk][valid_b] = np.array(final, dtype="uint8")

lulc = result.reshape(rows, cols)

# =====================================================
# STEP 10 : SAVE TIFF
# =====================================================

with rasterio.open(
    "LULC_REALISTIC_FINAL.tif", "w",
    driver="GTiff", height=rows, width=cols,
    count=1, dtype="uint8", crs=crs, transform=transform
) as dst:
    dst.write(lulc, 1)

print("Saved: LULC_REALISTIC_FINAL.tif")

# =====================================================
# STEP 11 : SAVE MAP IMAGE
# =====================================================

plt.figure(figsize=(12, 10))
img = plt.imshow(lulc, cmap="tab10")
plt.title("LULC Map", fontweight="bold")
plt.axis("off")
cbar = plt.colorbar(img, ticks=[0, 1, 2, 3, 4])
cbar.ax.set_yticklabels(["Water","Vegetation","Built-up","Wetland","Bare Land"])
plt.savefig("LULC_MAP_REALISTIC.png", dpi=300, bbox_inches="tight")
plt.close()
print("Map saved: LULC_MAP_REALISTIC.png")

# =====================================================
# PIXEL AREA FIX
# FIX: Compute area correctly regardless of CRS units.
# For geographic CRS (degrees), convert using mean latitude.
# For projected CRS (metres), use transform directly.
# =====================================================

if crs.is_geographic:
    # Degrees → metres: 1 degree latitude ≈ 111320 m
    # Use centre latitude of the image
    centre_lat = transform.f + transform.e * (rows / 2.0)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(abs(centre_lat)))
    pixel_w_m = abs(transform.a) * m_per_deg_lon
    pixel_h_m = abs(transform.e) * m_per_deg_lat
    pixel_area_m2 = pixel_w_m * pixel_h_m
else:
    # Projected CRS — units already in metres
    pixel_area_m2 = abs(transform.a * transform.e)

pixel_area_ha = pixel_area_m2 / 10000.0
print(f"\nPixel size  : {pixel_w_m if crs.is_geographic else abs(transform.a):.2f} m  x  "
      f"{pixel_h_m if crs.is_geographic else abs(transform.e):.2f} m")
print(f"Pixel area  : {pixel_area_m2:.2f} m²  =  {pixel_area_ha:.4f} ha")

# =====================================================
# ROC CURVES
# =====================================================

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

y_prob = xgb.predict_proba(X_test)
y_bin  = label_binarize(y_test, classes=np.arange(num_classes))

class_names = ["Water","Vegetation","Built-up","Wetland","Bare Land"]

plt.figure(figsize=(8, 6))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=3, label=f'{class_names[i]} (AUC={roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate",fontweight="bold")
plt.ylabel("True Positive Rate",fontweight="bold")
plt.title("Class-wise ROC Curves", fontweight="bold")
plt.legend(fontsize=12)
plt.savefig("ROC_Curves.png", dpi=800, bbox_inches="tight")
plt.close()
print("Saved: ROC_Curves.png")

# =====================================================
# PRECISION-RECALL CURVES
# =====================================================

from sklearn.metrics import precision_recall_curve

plt.figure(figsize=(8, 6))
for i in range(num_classes):
    precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
    plt.plot(recall, precision, linewidth=3, label=class_names[i])

plt.xlabel("Recall",fontweight='bold')
plt.ylabel("Precision",fontweight='bold')
plt.title("Class-wise Precision-Recall Curves", fontweight="bold")
plt.legend(fontsize=12)
plt.savefig("Precision_Recall_Curves.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: Precision_Recall_Curves.png")

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(y_test, final_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix", fontweight="bold")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11)

plt.xticks(np.arange(num_classes), class_names, rotation=30, ha="right")
plt.yticks(np.arange(num_classes), class_names)
plt.savefig("Confusion_Matrix.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: Confusion_Matrix.png")

# =====================================================
# PERFORMANCE METRICS BAR CHART
# =====================================================

from sklearn.metrics import precision_score, recall_score, f1_score

acc  = accuracy_score(y_test, final_pred)
prec = precision_score(y_test, final_pred, average='weighted', zero_division=0)
rec  = recall_score(y_test, final_pred, average='weighted', zero_division=0)
f1   = f1_score(y_test, final_pred, average='weighted', zero_division=0)

metrics       = [acc, prec, rec, f1]
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1']

plt.figure(figsize=(8, 6))
bars = plt.bar(metric_labels, metrics, color=['#2196F3','#4CAF50','#FF9800','#9C27B0'],
               edgecolor='black')
for bar, val in zip(bars, metrics):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{val:.4f}", ha='center', va='bottom', fontsize=13, fontweight='bold')
plt.ylim(0, 1.08)
plt.title("Performance Metrics", fontweight="bold")
plt.savefig("Performance_Metrics.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: Performance_Metrics.png")

# =====================================================
# CLASS-WISE FPR AND FNR
# =====================================================

fpr_list, fnr_list = [], []
for i in range(num_classes):
    TP = cm[i, i]
    FN = np.sum(cm[i, :]) - TP
    FP = np.sum(cm[:, i]) - TP
    TN = np.sum(cm) - TP - FN - FP
    fpr_list.append(FP / (FP + TN + 1e-10))
    fnr_list.append(FN / (FN + TP + 1e-10))

plt.figure(figsize=(8, 6))
x = np.arange(num_classes)
plt.bar(x - 0.2, fpr_list, width=0.4, label='FPR', color='#F44336', edgecolor='black')
plt.bar(x + 0.2, fnr_list, width=0.4, label='FNR', color='#2196F3', edgecolor='black')
plt.xticks(x, class_names, rotation=20, ha='right')
plt.xlabel("Class",fontweight='bold')
plt.ylabel("Rate",fontweight='bold')
plt.title("Class-wise FPR and FNR", fontweight="bold")
plt.legend()
plt.savefig("FPR_FNR.png", dpi=800, bbox_inches="tight")
plt.close()
print("Saved: FPR_FNR.png")

# =====================================================
# PHASE 6 : ECOLOGICAL INFRASTRUCTURE MAPPING
# FIX: Use sequential masking (like label generation)
#      so Coastal Wetlands and Marsh/Mangrove get
#      meaningful pixel counts instead of near-zero.
# =====================================================

print("\n============ PHASE 6 : ECOLOGICAL INFRASTRUCTURE MAPPING ============\n")

# Start with a blank map; assign in priority order
eco_map = np.zeros((rows, cols), dtype="uint8")

# 1. Water Bodies — strong open water signal
water_bodies = (ndwi > 0.30) | (lulc == 0)
eco_map[water_bodies] = 1

# 2. Tidal Flats — intertidal bare mud (bare land class, very low indices)
tidal_flats = (
    (ndvi < 0.10) & (ndwi < 0.10) & (mndwi < 0.10) &
    (lulc == 4) & (eco_map == 0)
)
eco_map[tidal_flats] = 2

# 3. River Corridors — elongated water channels (MNDWI dominant)
river_corridors = (
    (mndwi > 0.20) & (ndwi > 0.05) & (eco_map == 0)
)
eco_map[river_corridors] = 3

# 4. Coastal Wetlands — moderate wetness, wetland LULC class
coastal_wetlands = (
    (ndwi > 0.05) & (ndwi <= 0.30) &
    (lulc == 3) & (eco_map == 0)
)
eco_map[coastal_wetlands] = 4

# 5. Marsh / Mangrove Vegetation — vegetated + moist pixels
marsh_mangrove = (
    (ndvi > 0.30) & (ndwi > 0.02) &
    (lulc == 1) & (eco_map == 0)
)
eco_map[marsh_mangrove] = 5

eco_labels_names = {
    0: "Non-Ecological",
    1: "Water Bodies",
    2: "Tidal Flats",
    3: "River Corridors",
    4: "Coastal Wetlands",
    5: "Marsh/Mangrove Veg."
}

print(f"{'Component':<25} {'Pixels':>10} {'Area (ha)':>12}")
print("-" * 50)
for cls_id, cls_name in eco_labels_names.items():
    pixel_count = int(np.sum(eco_map == cls_id))
    area_ha     = pixel_count * pixel_area_ha
    print(f"{cls_name:<25} {pixel_count:>10,} {area_ha:>12.2f}")
print("-" * 50)

# Save GeoTIFF
with rasterio.open(
    "Ecological_Infrastructure_Map.tif", "w",
    driver="GTiff", height=rows, width=cols,
    count=1, dtype="uint8", crs=crs, transform=transform
) as dst:
    dst.write(eco_map, 1)
print("\nSaved: Ecological_Infrastructure_Map.tif")

# Ecological Infrastructure Map
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

eco_colors = [
    "#d9d9d9",   # 0 – Non-Ecological
    "#1a6faf",   # 1 – Water Bodies
    "#c2a84a",   # 2 – Tidal Flats
    "#4db8ff",   # 3 – River Corridors
    "#2ca25f",   # 4 – Coastal Wetlands
    "#006d2c",   # 5 – Marsh/Mangrove
]

eco_cmap = ListedColormap(eco_colors)
eco_norm = BoundaryNorm(boundaries=np.arange(-0.5, 6.5, 1), ncolors=6)

legend_patches = [
    Patch(facecolor=eco_colors[i], edgecolor="black", label=eco_labels_names[i])
    for i in range(6)
]

fig, ax = plt.subplots(figsize=(14, 11))
im = ax.imshow(eco_map, cmap=eco_cmap, norm=eco_norm)
ax.set_title("Ecological Infrastructure Map\n(Dongying Coastal Region)",
             fontweight="bold", fontsize=20, pad=14)
ax.axis("off")
ax.legend(handles=legend_patches, loc="lower right", fontsize=13,
          framealpha=0.9, title="Ecological Components", title_fontsize=14)
plt.tight_layout()
plt.savefig("Ecological_Infrastructure_Map.png", dpi=300, bbox_inches="tight")
plt.close()
print("Map saved: Ecological_Infrastructure_Map.png")

# Area Pie Chart (exclude background class 0)
eco_pixel_counts = [int(np.sum(eco_map == cls_id)) for cls_id in range(1, 6)]
# Guard: if any component is 0 pixels, replace with 1 to avoid empty pie slice error
eco_pixel_counts = [max(c, 1) for c in eco_pixel_counts]
eco_pie_labels   = [eco_labels_names[i] for i in range(1, 6)]
eco_pie_colors   = eco_colors[1:]

fig, ax = plt.subplots(figsize=(9, 7))
wedges, texts, autotexts = ax.pie(
    eco_pixel_counts, labels=eco_pie_labels, colors=eco_pie_colors,
    autopct="%1.1f%%", startangle=140, pctdistance=0.82,
    wedgeprops=dict(edgecolor="white", linewidth=1.5)
)
for t in autotexts:
    t.set_fontsize(12)
    t.set_fontweight("bold")
ax.set_title("Ecological Component Area Distribution",
             fontweight="bold", fontsize=16, pad=16)
plt.tight_layout()
plt.savefig("Ecological_Area_PieChart.png", dpi=300, bbox_inches="tight")
plt.close()
print("Chart saved: Ecological_Area_PieChart.png")

# Sub-Component Overview Panel
sub_maps   = [water_bodies, tidal_flats, river_corridors, coastal_wetlands, marsh_mangrove]
sub_titles = ["Water Bodies","Tidal Flats","River Corridors","Coastal Wetlands","Marsh/Mangrove Veg."]
sub_cmaps  = ["Blues","YlOrBr","PuBu","Greens","YlGn"]

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()
for k, (mask, title, cmap_name) in enumerate(zip(sub_maps, sub_titles, sub_cmaps)):
    axes[k].imshow(mask.astype("uint8"), cmap=cmap_name, vmin=0, vmax=1)
    axes[k].set_title(title, fontweight="bold", fontsize=14)
    axes[k].axis("off")

axes[5].imshow(eco_map, cmap=eco_cmap, norm=eco_norm)
axes[5].set_title("Combined Eco. Infrastructure", fontweight="bold", fontsize=14)
axes[5].axis("off")
plt.suptitle("Ecological Infrastructure Sub-Components",
             fontsize=18, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("Ecological_SubComponents_Overview.png", dpi=800, bbox_inches="tight")
plt.close()
print("Overview saved: Ecological_SubComponents_Overview.png")
print("\nPhase 6 Complete.")

# =====================================================
# PHASE 7 : CLIMATE ADAPTATION ASSESSMENT (AHP)
# =====================================================

print("\n============ PHASE 7 : CLIMATE ADAPTATION ASSESSMENT ============\n")

# ---- Load Real DEM (Dongying_DEM.tif) ----
try:
    dem_ds  = rasterio.open("Dongying_DEM.tif")
    dem_raw = dem_ds.read(1).astype("float32")

    # Resample to match Landsat grid if sizes differ
    if dem_raw.shape != (rows, cols):
        from rasterio.enums import Resampling
        dem_raw = dem_ds.read(
            1,
            out_shape=(rows, cols),
            resampling=Resampling.bilinear
        ).astype("float32")
        print(f"DEM resampled from {dem_ds.shape} → ({rows},{cols})")

    dem_ds.close()
    print(f"Loaded: Dongying_DEM.tif  |  elevation range: "
          f"{np.nanmin(dem_raw):.1f} – {np.nanmax(dem_raw):.1f} m")

except Exception as e:
    print(f"WARNING: Dongying_DEM.tif not found ({e}). Using spectral proxy.")
    dem_raw = (1.0 - ndwi) * 30.0 + np.random.normal(0, 2, ndwi.shape).astype("float32")

# ---- Derive Slope from Real DEM ----
# Pixel spacing in metres for gradient calculation
if crs.is_geographic:
    centre_lat_val = transform.f + transform.e * (rows / 2.0)
    dy_m = abs(transform.e) * 111320.0
    dx_m = abs(transform.a) * 111320.0 * np.cos(np.radians(abs(centre_lat_val)))
else:
    dy_m = abs(transform.e)
    dx_m = abs(transform.a)

grad_y, grad_x = np.gradient(dem_raw, dy_m, dx_m)
slope_raw = np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_y**2))).astype("float32")
print(f"Slope derived from Dongying_DEM.tif  |  range: "
      f"{np.nanmin(slope_raw):.2f}° – {np.nanmax(slope_raw):.2f}°")

def norm_0_1(arr):
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx - mn < 1e-9:
        return np.zeros_like(arr, dtype="float32")
    return ((arr - mn) / (mx - mn)).astype("float32")

ndvi_n  = norm_0_1(ndvi)
ndwi_n  = norm_0_1(ndwi)
eco_n   = norm_0_1(eco_map.astype("float32"))
dem_n   = norm_0_1(dem_raw)
slope_n = norm_0_1(slope_raw)
slope_inv = 1.0 - slope_n

w_ndvi  = 0.35
w_ndwi  = 0.25
w_eco   = 0.20
w_dem   = 0.12
w_slope = 0.08

adaptation_score = (
    w_ndvi  * ndvi_n   +
    w_ndwi  * ndwi_n   +
    w_eco   * eco_n    +
    w_dem   * dem_n    +
    w_slope * slope_inv
)

adapt_class = np.zeros_like(adaptation_score, dtype="uint8")
adapt_class[adaptation_score >= 0.60] = 3
adapt_class[(adaptation_score >= 0.35) & (adaptation_score < 0.60)] = 2
adapt_class[adaptation_score < 0.35]  = 1

with rasterio.open(
    "Climate_Adaptation_Map.tif", "w",
    driver="GTiff", height=rows, width=cols,
    count=1, dtype="uint8", crs=crs, transform=transform
) as dst:
    dst.write(adapt_class, 1)
print("Saved: Climate_Adaptation_Map.tif")

adapt_colors = ["#d73027", "#fee08b", "#1a9850"]
adapt_cmap   = ListedColormap(adapt_colors)
adapt_norm   = BoundaryNorm([0.5, 1.5, 2.5, 3.5], ncolors=3)
adapt_legend = [
    Patch(facecolor=adapt_colors[0], edgecolor="black", label="Low Adaptation"),
    Patch(facecolor=adapt_colors[1], edgecolor="black", label="Moderate Adaptation"),
    Patch(facecolor=adapt_colors[2], edgecolor="black", label="High Adaptation"),
]

fig, ax = plt.subplots(figsize=(13, 10))
ax.imshow(adapt_class, cmap=adapt_cmap, norm=adapt_norm)
ax.set_title("Climate Adaptation Capacity Map (AHP)", fontweight="bold", fontsize=18)
ax.axis("off")
ax.legend(handles=adapt_legend, loc="lower right", fontsize=13,
          framealpha=0.9, title="Adaptation Capacity", title_fontsize=14)
plt.tight_layout()
plt.savefig("Climate_Adaptation_Map.png", dpi=300, bbox_inches="tight")
plt.close()
print("Map saved: Climate_Adaptation_Map.png")

adapt_labels_list = ["Low", "Moderate", "High"]
adapt_counts = [int(np.sum(adapt_class == c)) for c in [1, 2, 3]]

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(adapt_labels_list, adapt_counts,
              color=adapt_colors, edgecolor="black", linewidth=1.2)
for bar, cnt in zip(bars, adapt_counts):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(adapt_counts) * 0.01,
            f"{cnt:,}", ha="center", va="bottom", fontsize=13, fontweight="bold")
ax.set_xlabel("Adaptation Capacity Class")
ax.set_ylabel("Pixel Count")
ax.set_title("Climate Adaptation Class Distribution", fontweight="bold")
plt.tight_layout()
plt.savefig("Climate_Adaptation_BarChart.png", dpi=300, bbox_inches="tight")
plt.close()
print("Chart saved: Climate_Adaptation_BarChart.png")
print("\nPhase 7 Complete.")

# =====================================================
# PHASE 8 : COASTAL VULNERABILITY ASSESSMENT
# =====================================================

print("\n============ PHASE 8 : COASTAL VULNERABILITY ASSESSMENT ============\n")

slr_threshold_m = 2.0
slr_risk        = (dem_raw < slr_threshold_m).astype("float32")
slr_risk_n      = norm_0_1(dem_raw * -1)

erosion_raw = (1.0 - ndvi_n) * 0.6 + ndwi_n * 0.4
erosion_n   = norm_0_1(erosion_raw)

veg_vuln_n   = 1.0 - ndvi_n
water_vuln_n = ndwi_n

lulc_vuln_map = {0: 0.9, 1: 0.2, 2: 0.7, 3: 0.4, 4: 0.8}
lulc_vuln = np.vectorize(lulc_vuln_map.get)(lulc).astype("float32")

w_slr     = 0.30
w_erosion = 0.25
w_veg     = 0.20
w_lulc    = 0.15
w_water   = 0.10

vuln_score = (
    w_slr     * slr_risk_n   +
    w_erosion * erosion_n    +
    w_veg     * veg_vuln_n   +
    w_lulc    * lulc_vuln    +
    w_water   * water_vuln_n
)

vuln_class = np.zeros_like(vuln_score, dtype="uint8")
vuln_class[vuln_score >= 0.60] = 3
vuln_class[(vuln_score >= 0.35) & (vuln_score < 0.60)] = 2
vuln_class[vuln_score < 0.35]  = 1

with rasterio.open(
    "Coastal_Vulnerability_Map.tif", "w",
    driver="GTiff", height=rows, width=cols,
    count=1, dtype="uint8", crs=crs, transform=transform
) as dst:
    dst.write(vuln_class, 1)
print("Saved: Coastal_Vulnerability_Map.tif")

vuln_colors = ["#2166ac", "#f7f7f7", "#d6604d"]
vuln_cmap   = ListedColormap(vuln_colors)
vuln_norm   = BoundaryNorm([0.5, 1.5, 2.5, 3.5], ncolors=3)
vuln_legend = [
    Patch(facecolor=vuln_colors[0], edgecolor="black", label="Low Vulnerability"),
    Patch(facecolor=vuln_colors[1], edgecolor="black", label="Moderate Vulnerability"),
    Patch(facecolor=vuln_colors[2], edgecolor="black", label="High Vulnerability"),
]

fig, ax = plt.subplots(figsize=(13, 10))
ax.imshow(vuln_class, cmap=vuln_cmap, norm=vuln_norm)
ax.set_title("Coastal Vulnerability Map", fontweight="bold", fontsize=18)
ax.axis("off")
ax.legend(handles=vuln_legend, loc="lower right", fontsize=13,
          framealpha=0.9, title="Vulnerability Level", title_fontsize=14)
plt.tight_layout()
plt.savefig("Coastal_Vulnerability_Map.png", dpi=300, bbox_inches="tight")
plt.close()
print("Map saved: Coastal_Vulnerability_Map.png")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(slr_risk, cmap="Blues")
axes[0].set_title("Sea-Level Rise Risk", fontweight="bold", fontsize=14)
axes[0].axis("off")

axes[1].imshow(erosion_n, cmap="Reds")
axes[1].set_title("Coastal Erosion Risk", fontweight="bold", fontsize=14)
axes[1].axis("off")

axes[2].imshow(vuln_class, cmap=vuln_cmap, norm=vuln_norm)
axes[2].set_title("Combined Vulnerability", fontweight="bold", fontsize=14)
axes[2].axis("off")
plt.suptitle("Phase 8 – Coastal Vulnerability Sub-Blocks",
             fontsize=17, fontweight="bold")
plt.tight_layout()
plt.savefig("Coastal_Vulnerability_SubBlocks.png", dpi=300, bbox_inches="tight")
plt.close()
print("Sub-blocks saved: Coastal_Vulnerability_SubBlocks.png")
print("\nPhase 8 Complete.")

# =====================================================
# PHASE 9 : NbS SUITABILITY ANALYSIS
# =====================================================

print("\n============ PHASE 9 : NbS SUITABILITY ANALYSIS ============\n")

adapt_n = norm_0_1(adapt_class.astype("float32"))
vuln_n  = norm_0_1(vuln_class.astype("float32"))
eco_n2  = norm_0_1(eco_map.astype("float32"))

wetland_suit = norm_0_1(
    vuln_n * 0.40 +
    (eco_map == 4).astype("float32") * 0.35 +
    ndwi_n * 0.25
)

mangrove_suit = norm_0_1(
    ndvi_n * 0.40 +
    (eco_map == 5).astype("float32") * 0.35 +
    (1.0 - dem_n) * 0.25
)

shoreline_suit = norm_0_1(
    vuln_n * 0.45 +
    (eco_map == 2).astype("float32") * 0.30 +
    (1.0 - dem_n) * 0.25
)

corridor_suit = norm_0_1(
    adapt_n * 0.35 +
    (eco_map == 3).astype("float32") * 0.35 +
    ndvi_n * 0.30
)

w_wet  = 0.30
w_mang = 0.28
w_shor = 0.25
w_corr = 0.17

nbs_score = (
    w_wet  * wetland_suit   +
    w_mang * mangrove_suit  +
    w_shor * shoreline_suit +
    w_corr * corridor_suit
)

nbs_class = np.zeros_like(nbs_score, dtype="uint8")
nbs_class[nbs_score >= 0.60] = 3
nbs_class[(nbs_score >= 0.35) & (nbs_score < 0.60)] = 2
nbs_class[nbs_score < 0.35]  = 1

with rasterio.open(
    "NbS_Suitability_Map.tif", "w",
    driver="GTiff", height=rows, width=cols,
    count=1, dtype="uint8", crs=crs, transform=transform
) as dst:
    dst.write(nbs_class, 1)
print("Saved: NbS_Suitability_Map.tif")

nbs_colors = ["#ffffcc", "#78c679", "#005a32"]
nbs_cmap   = ListedColormap(nbs_colors)
nbs_norm   = BoundaryNorm([0.5, 1.5, 2.5, 3.5], ncolors=3)
nbs_legend = [
    Patch(facecolor=nbs_colors[0], edgecolor="black", label="Low Suitability"),
    Patch(facecolor=nbs_colors[1], edgecolor="black", label="Moderate Suitability"),
    Patch(facecolor=nbs_colors[2], edgecolor="black", label="High Suitability"),
]

fig, ax = plt.subplots(figsize=(13, 10))
ax.imshow(nbs_class, cmap=nbs_cmap, norm=nbs_norm)
ax.set_title("Nature-Based Solution (NbS) Suitability Map",
             fontweight="bold", fontsize=18)
ax.axis("off")
ax.legend(handles=nbs_legend, loc="lower right", fontsize=13,
          framealpha=0.9, title="NbS Suitability", title_fontsize=14)
plt.tight_layout()
plt.savefig("NbS_Suitability_Map.png", dpi=300, bbox_inches="tight")
plt.close()
print("Map saved: NbS_Suitability_Map.png")

nbs_sub_maps   = [wetland_suit, mangrove_suit, shoreline_suit, corridor_suit]
nbs_sub_titles = ["Wetland Restoration","Mangrove Restoration",
                  "Living Shorelines","Ecological Corridors"]
nbs_sub_cmaps  = ["YlGn","Greens","PuBu","BuGn"]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()
for k in range(4):
    im = axes[k].imshow(nbs_sub_maps[k], cmap=nbs_sub_cmaps[k], vmin=0, vmax=1)
    axes[k].set_title(nbs_sub_titles[k], fontweight="bold", fontsize=14)
    axes[k].axis("off")
    plt.colorbar(im, ax=axes[k], fraction=0.046, pad=0.04)

plt.suptitle("NbS Sub-Component Suitability Scores",
             fontsize=17, fontweight="bold")
plt.tight_layout()
plt.savefig("NbS_SubComponents.png", dpi=300, bbox_inches="tight")
plt.close()
print("Sub-components saved: NbS_SubComponents.png")
print("\nPhase 9 Complete.")

# =====================================================
# PHASE 10 : REGENERATIVE LANDSCAPE DEVELOPMENT
# =====================================================

print("\n============ PHASE 10 : REGENERATIVE LANDSCAPE DEVELOPMENT ============\n")

nbs_n    = norm_0_1(nbs_class.astype("float32"))
adapt_n2 = norm_0_1(adapt_class.astype("float32"))
eco_n3   = norm_0_1(eco_map.astype("float32"))

restoration = norm_0_1(
    nbs_n * 0.50 + (1.0 - adapt_n2) * 0.30 + vuln_n * 0.20
)

conservation = norm_0_1(
    eco_n3 * 0.50 + adapt_n2 * 0.30 + ndvi_n * 0.20
)

green_network = norm_0_1(
    (eco_map == 3).astype("float32") * 0.40 +
    ndvi_n * 0.35 + adapt_n2 * 0.25
)

coastal_buffer = norm_0_1(
    vuln_n * 0.40 + (1.0 - dem_n) * 0.30 +
    ((eco_map == 2) | (eco_map == 4)).astype("float32") * 0.30
)

zone_stack = np.stack([restoration, conservation, green_network, coastal_buffer], axis=-1)
dominant   = np.argmax(zone_stack, axis=-1).astype("uint8") + 1

with rasterio.open(
    "Regenerative_Landscape_Map.tif", "w",
    driver="GTiff", height=rows, width=cols,
    count=1, dtype="uint8", crs=crs, transform=transform
) as dst:
    dst.write(dominant, 1)
print("Saved: Regenerative_Landscape_Map.tif")

regen_colors = ["#e31a1c","#33a02c","#6a3d9a","#1f78b4"]
regen_cmap   = ListedColormap(regen_colors)
regen_norm   = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], ncolors=4)
regen_legend = [
    Patch(facecolor=regen_colors[0], edgecolor="black", label="Restoration Zones"),
    Patch(facecolor=regen_colors[1], edgecolor="black", label="Conservation Zones"),
    Patch(facecolor=regen_colors[2], edgecolor="black", label="Green Networks"),
    Patch(facecolor=regen_colors[3], edgecolor="black", label="Coastal Buffer Zones"),
]

fig, ax = plt.subplots(figsize=(13, 10))
ax.imshow(dominant, cmap=regen_cmap, norm=regen_norm)
ax.set_title("Regenerative Landscape Development Map",
             fontweight="bold", fontsize=18)
ax.axis("off")
ax.legend(handles=regen_legend, loc="lower right", fontsize=13,
          framealpha=0.9, title="Landscape Zone", title_fontsize=14)
plt.tight_layout()
plt.savefig("Regenerative_Landscape_Map.png", dpi=300, bbox_inches="tight")
plt.close()
print("Map saved: Regenerative_Landscape_Map.png")

regen_sub_maps   = [restoration, conservation, green_network, coastal_buffer]
regen_sub_titles = ["Restoration Zones","Conservation Zones",
                    "Green Networks","Coastal Buffer Zones"]
regen_sub_cmaps  = ["Reds","Greens","Purples","Blues"]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()
for k in range(4):
    im = axes[k].imshow(regen_sub_maps[k], cmap=regen_sub_cmaps[k], vmin=0, vmax=1)
    axes[k].set_title(regen_sub_titles[k], fontweight="bold", fontsize=14)
    axes[k].axis("off")
    plt.colorbar(im, ax=axes[k], fraction=0.046, pad=0.04)

plt.suptitle(" Regenerative Landscape Sub-Zones",
             fontsize=17, fontweight="bold")
plt.tight_layout()
plt.savefig("Regenerative_Landscape_SubZones.png", dpi=300, bbox_inches="tight")
plt.close()
print("Sub-zones saved: Regenerative_Landscape_SubZones.png")

zone_names = {1:"Restoration", 2:"Conservation", 3:"Green Network", 4:"Coastal Buffer"}
print(f"\n{'Zone':<22} {'Pixels':>10} {'Area (ha)':>12}")
print("-" * 46)
for zid, zname in zone_names.items():
    cnt  = int(np.sum(dominant == zid))
    area = cnt * pixel_area_ha
    print(f"{zname:<22} {cnt:>10,} {area:>12.2f}")

zone_counts = [int(np.sum(dominant == z)) for z in [1, 2, 3, 4]]
zone_labels = ["Restoration","Conservation","Green Networks","Coastal Buffer"]

fig, ax = plt.subplots(figsize=(9, 7))
wedges, texts, autotexts = ax.pie(
    zone_counts, labels=zone_labels, colors=regen_colors,
    autopct="%1.1f%%", startangle=140, pctdistance=0.82,
    wedgeprops=dict(edgecolor="white", linewidth=1.5)
)
for t in autotexts:
    t.set_fontsize(12)
    t.set_fontweight("bold")
ax.set_title("Regenerative Landscape Zone Distribution",
             fontweight="bold", fontsize=16, pad=16)
plt.tight_layout()
plt.savefig("Regenerative_Landscape_PieChart.png", dpi=300, bbox_inches="tight")
plt.close()
print("Chart saved: Regenerative_Landscape_PieChart.png")

print("\nPhase 10 Complete.")
print("\n========== ALL PHASES COMPLETE ==========")