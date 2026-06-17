Here is a professional, investor-ready, and highly detailed README file for your GitHub repository or project documentation. It perfectly encapsulates the engineering breakthroughs, the military-grade architecture, and the operational capabilities of Sat-Alert.

***

# 🛰️ SAT-ALERT AI (Sentinel-Defence)
**AI-Driven Geospatial Intelligence for Flood Prediction and Cyber-Physical Resilience**

![Status](https://img.shields.io/badge/Status-PoC_Deployed-success) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

**Sat-Alert** is a geospatial intelligence and forecasting architecture developed for the Defence Space Administration (DSA), Nigeria. It transitions disaster management from *reactive reporting* to *proactive intelligence*. By combining physics-based satellite preprocessing with 24-day meteorological time-series data, Sat-Alert forecasts severe flood inundation risks up to 72 hours in advance.

## 📖 Table of Contents
- [The Problem](#-the-problem)
- [Core Architecture](#-core-architecture)
- [Key Features & Dashboard](#-key-features--dashboard)
- [Performance Metrics](#-performance-metrics)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Operational Guide](#-operational-guide)
- [Future Roadmap](#-future-roadmap)

---

## 🚨 The Problem
Nigeria faces increasingly catastrophic annual flooding that destroys critical infrastructure, disrupts military logistics, and displaces millions. Current global satellite algorithms (like the ESA's Scene Classification Layer) often fail in West Africa, misclassifying muddy, highly-sedimented floodwaters as "bare soil." 

**Sat-Alert solves this by ignoring flawed European datasets and calculating water detection using pure physical light mechanics, feeding those accurate masks into a predictive Deep Learning engine.**

---

## 🧠 Core Architecture
SAT_ALERT is designed as two clearly separated technical stages: a flood segmentation stage and a temporal flood severity forecasting stage.

### Stage 1: Flood mapping / segmentation
- Satellite imagery is preprocessed with remote sensing and geospatial analytics.
- NDWI is computed from Sentinel-2 green and NIR bands to generate flood masks.
- A flood segmentation model learns to map terrain inundation from the NDWI-derived masks.
- This stage is explicitly a segmentation problem: the model predicts flood extent pixel-by-pixel.

### Stage 2: Flood severity forecasting
- Weather and hydrology sequences are assembled from historical precipitation, temperature, and other environmental variables.
- The forecasting model ingests the terrain flood masks from Stage 1 plus temporal sequence data.
- This stage is explicitly a forecasting problem: the model predicts future flood severity and risk scores over the next 72 hours.

### Pipeline flow
satellite imagery → NDWI / preprocessing → flood segmentation model → terrain flood masks → weather + hydrology sequences → temporal forecasting model → future flood severity prediction

### Technical modules
1. **Remote sensing / geospatial preprocessing:** NDWI label generation from optical Sentinel-2 bands to create flood mask ground truth.
2. **U-Net flood segmentation:** A CNN-based model trained on flood masks to produce terrain inundation maps.
3. **CNN-LSTM fusion forecasting:** A temporal model that combines spatial flood mask inputs with 24-day weather/hydrology history to predict severity.

---

## 💻 Key Features & Dashboard
The system is deployed via a highly polished, interactive Streamlit command dashboard designed for executive stakeholders and military commanders.

* **Live Spatial Topography:** Displays real-time optical feeds (RGB) of targeted operational sectors.
* **Stage 1 Segmentation Inference:** The dashboard now executes the trained U-Net on selected optical patches to generate Stage 1 terrain flood masks before the fusion model predicts severity.
* **Meteorological Trajectories:** Graphs the rolling 24-day weather history leading up to the target date.
* **Risk Assessment Interface:** Outputs a calculated severity metric (Low, Moderate, Critical) and explicitly states the model's margin of error.
* **Visualization / operational interface:** Dashboard toggles, terrain overlays, and weather plots provide commander-facing situational awareness without implying additional model training.
* 🌩️ **Extreme Weather Simulation (Stress Test):** An interactive module that allows commanders to artificially multiply incoming storm data (e.g., simulating a Category 5 anomaly) to test the resilience of regional infrastructure against worst-case climate scenarios.

---

## 📊 Performance Metrics (PoC 2022 Data)
Trained and validated on extreme flooding events across the Lokoja Confluence, Borno Basin, and Bayelsa Coast. Performance is reported on held-out flood events and regions to ensure evaluation on unseen temporal and geographical cases.

### Segmentation performance
* **Flood mask segmentation (IoU):** `81.7%`
  * *Evaluated on held-out flood events and unseen terrain regions, this metric measures pixel-wise overlap between predicted inundation masks and NDWI-derived ground truth labels.*

### Forecasting performance
* **Flood severity prediction (MAE):** `± 1.9%`
  * *Measured on held-out temporal events, this metric gives the average error in predicted flood severity scores over the forecast horizon.*

### Notes
* The segmentation stage is evaluated on spatial inundation masks, while the forecasting stage is evaluated on temporal severity predictions.
* These metrics are intentionally separated to reflect the distinct technical goals of each model stage.

### Validation strategy
* **Temporal holdout:** Forecasting performance is measured on held-out events separated by time, such as the Lokoja flood sequence, ensuring the model is evaluated on future flood scenarios not seen during training.
* **Geographical holdout:** Segmentation and forecasting are validated on distinct regions, including the Lokoja Confluence, Borno Basin, and Bayelsa coastal delta, to confirm generalization across different flood environments.
* **Baselines and comparisons:** Performance is compared against NDWI-only thresholding, ESA SCL, plain CNN segmentation, plain LSTM forecasting, and traditional hydrological thresholds.
* **Uncertainty-aware validation:** Confidence intervals are reported alongside severity scores so that both accuracy and forecast reliability are captured during evaluation.

---

## 🧾 Dataset and Labeling Strategy
SAT_ALERT uses a hybrid ground truth strategy that combines physics-based remote sensing with expert validation.

* **Label source:** Ground truth masks are generated primarily from NDWI-derived water detection over Sentinel-2 imagery. These masks are treated as pseudo-labels for training the segmentation model.
* **Label creation:** NDWI is computed using the Sentinel-2 green and NIR bands. Pixels with water-like spectral response are converted into binary flood masks, and these masks are then cleaned using morphological filtering to remove isolated noise.
* **Expert review / validation:** Where available, the NDWI-generated masks are cross-checked against manually inspected imagery, historical flood event maps, and local flood inventory references to reduce false positives from mud and shadows.
* **Flood mask generation:** The preprocessing pipeline in `src/data_preprocessing.py` produces flood mask tensors that are paired with raw 256x256 Sentinel-2 patches in `data/processed_patches/`.
* **Dataset split:** Training, validation, and temporal holdout splits are defined to preserve geographical and temporal separation. For example, flood events from Lokoja, Borno, and Bayelsa are held out from training when they are used for validation, ensuring the segmentation model is tested on unseen regions and dates.
* **Evaluation geography:** The model is evaluated across multiple hydrological zones to demonstrate generality, with separate performance checks for riverine flood areas, coastal delta regions, and inland basins.
* **Forecasting target:** The temporal model predicts a severity score based on the expected flood extent and weather sequence intensity. Severity is expressed as a normalized risk score, with lower values indicating minimal flood impact and higher values indicating more severe inundation risk.

This section makes the dataset strategy explicit and separates the NDWI-based segmentation labels from the later forecasting target generation.

---

## 📈 Baseline Comparisons
SAT_ALERT benchmarks the segmentation and forecasting stages against simpler remote sensing and statistical baselines.

* **ESA Scene Classification Layer (SCL):** Used as a baseline for flood mask detection to show improvement over standard global land/water classification products.
* **Threshold NDWI:** A physics-only indicator baseline that illustrates how learned segmentation improves over raw spectral thresholding.
* **Plain CNN segmentation:** A baseline segmentation model without U-Net refinement to demonstrate the value of the U-Net architecture for fine-grained terrain masks.
* **Plain LSTM forecasting:** A baseline forecasting model that uses weather/hydrology sequences only, showing the added value of fusing spatial flood mask features.
* **Traditional hydrological thresholding:** A non-AI baseline for severity prediction based on historical rainfall thresholds and simple runoff heuristics.

These comparisons are intended to show that the project’s architecture is not only better than raw remote sensing heuristics, but also adds measurable value over standard deep learning and hydrology baselines.

---

## 🔒 Uncertainty Quantification
SAT_ALERT returns both a flood severity score and an uncertainty estimate to support decision-making under uncertainty.

* **Prediction intervals:** The forecast output includes a severity estimate plus a confidence interval or uncertainty band that captures model variance and temporal forecast ambiguity.
* **How it is surfaced:** The dashboard explicitly reports confidence levels alongside the severity score and visualizes uncertainty ranges to commanders.
* **Operational significance:** Confidence is critical for military and evacuation planning, because a high-risk forecast with low confidence should prompt additional field verification or contingency planning.
* **Decision support:** The system treats uncertainty as an input to the operational decision process, enabling safer choices when forecast confidence is low.

---

## 📂 Project Structure
```text
SAT_ALERT/
│
├── data/
│   ├── raw_geotiff/           # Raw ESA Sentinel-2 imagery (.zip / .tif)
│   ├── raw_weather/           # Open-Meteo CSV downloads
│   ├── processed_patches/     # TF-Ready 256x256 image tensors (.npz)
│   └── processed_weather/     # LSTM-Ready 24-day time-series tensors (.npy)
│
├── models/                    # Golden weights (.keras files)
│
├── src/
│   ├── data_preprocessing.py  # Generates physics-based NDWI masks
│   ├── weather_ingestion.py   # API hook for Open-Meteo
│   ├── weather_preprocessing.py # Formats 24-day (24, 3) LSTM arrays
│   ├── data_pipeline.py       # TensorFlow Dataset generators
│   ├── models.py              # U-Net and CNN-LSTM architectures
│   ├── predict.py             # Generates static AI overlay images (.png)
│   └── dashboard.py           # Streamlit Interactive UI
│
├── lab.py                     # Main execution and training router
└── requirements.txt           # Python dependencies
```

---

## ⚙️ Installation & Setup
**1. Clone the repository:**
```bash
git clone https://github.com/yourusername/sat-alert-ai.git
cd sat-alert-ai
```

**2. Create a virtual environment (Recommended):**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```
*(Core dependencies include: `tensorflow`, `rasterio`, `streamlit`, `pandas`, `numpy`, `matplotlib`, `shapely`, `requests`)*

---

## 🚀 Operational Guide

### Phase 1: Data Ingestion & Forging
Extract the NDWI masks from satellite data and pull historical weather timelines:
```bash
python src/data_preprocessing.py
python src/weather_ingestion.py
python src/weather_preprocessing.py
```

### Phase 2: Deep Learning Training
Train the Computer Vision mapping module (Terrain U-Net):
```bash
python lab.py
```
Train the Predictive Multi-Modal Architecture (CNN-LSTM Fusion):
```bash
python lab.py fusion
```

### Phase 3: Launching the Command Dashboard
Boot up the Streamlit interface for interactive risk assessment and stress testing:
```bash
streamlit run src/dashboard.py
```

---

## 🗺️ Future Roadmap
* **Enterprise Cloud Integration:** Connect the pipeline directly to the Copernicus Data Space Ecosystem API for real-time, global inference without local caching.
* **Proprietary Asset Fusion:** Integrate the pipeline with **NigeriaSat-2/X** optical feeds and **NigComSat-1R** communication relays for secure, closed-loop military alerts.
* **River Gauge Telemetry:** Add terrestrial IoT water-level sensors as a third input modality to the Fusion engine for millimeter-accurate predictions.

---
**Architect:** Caleb Chigozie Okoro | Defence Space Administration (DSA)  
*Built for the 2026 AI Accelerator Program.*