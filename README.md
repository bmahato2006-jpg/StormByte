# CYCLONE-X

**AI-Powered Tropical Cyclone Intelligence, Digital Twin, and Disaster Decision-Support Platform**

---

> **CRITICAL SCIENTIFIC ADVISORY & ETHICS STATEMENT**  
> CYCLONE-X is an experimental research and academic decision-support platform. It does **not** issue official meteorological warnings and does **not** claim to replace bulletins from the **India Meteorological Department (IMD)**, the **National Hurricane Center (NHC)**, or the **World Meteorological Organization (WMO)** Regional Specialized Meteorological Centres.  
>  
> In accordance with strict scientific integrity rules, all demo data and historical benchmarks are transparently labeled as **DEMO DATA**, and AI predictions are explicitly classified as **EXPERIMENTAL PREDICTION**. No AI accuracy or confidence numbers are fabricated.

---

## ⚡ Quick Start (One Click)

> **Prerequisites:** Python 3.10+ and Node.js 18+ must be installed.

### Windows — Double-click to launch
```
start.bat          ← Starts backend + frontend + opens browser automatically
stop.bat           ← Cleanly stops all servers
```

### Manual startup (two terminals)
```bash
# Terminal 1 — Backend (FastAPI on :8000)
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend (Next.js on :3000)
cd frontend
npm run dev
```

| Service | URL |
|---------|-----|
| 🌐 Dashboard | http://localhost:3000 |
| 🔌 API | http://localhost:8000 |
| 📖 API Docs | http://localhost:8000/docs |

> **First run?** The frontend takes ~10 s to compile. Wait for the terminal to print `Ready - started server on 0.0.0.0:3000` before opening the browser.

---

## 1. Project Overview

CYCLONE-X is a next-generation meteorological intelligence platform designed to bridge cutting-edge deep learning with physical cyclone dynamics. Standard cyclone dashboards simply plot delayed public bulletins; CYCLONE-X implements a complete operational pipeline:

$$\text{OBSERVE} \longrightarrow \text{DETECT} \longrightarrow \text{CLASSIFY} \longrightarrow \text{LOCATE} \longrightarrow \text{TRACK} \longrightarrow \text{PREDICT (6h/12h/24h)} \longrightarrow \text{UNCERTAINTY} \longrightarrow \text{RISK ASSESSMENT} \longrightarrow \text{SIMULATE} \longrightarrow \text{EXPLAIN (XAI)} \longrightarrow \text{REPORT}$$

---

## 2. System Architecture

```
                                +-----------------------------------+
                                |    CYCLONE-X Command Center UI   |
                                |     (Next.js 14 + Leaflet GIS)    |
                                +-----------------+-----------------+
                                                  |
                                                  | REST API / WebSocket
                                                  v
                                +-----------------+-----------------+
                                |        FastAPI Backend Core       |
                                |     (Pydantic, Uvicorn, Async)    |
                                +--------+--------+--------+--------+
                                         |        |        |
         +-------------------------------+        |        +-------------------------------+
         v                                        v                                        v
+------------------+                    +------------------+                    +------------------+
| PostGIS Spatial  |                    | AI / ML Pipeline |                    | Data Provenance  |
| Database Layer   |                    | (PyTorch/OpenCV) |                    |  Audit Registry  |
+------------------+                    +------------------+                    +------------------+
```

---

## 3. Technology Stack

- **Frontend**:
  - **Framework**: Next.js 14 (App Router)
  - **Language**: TypeScript
  - **Styling**: Tailwind CSS (Dark Command-Center Oceanic Theme)
  - **Geospatial Mapping**: Leaflet (Dark Matter CartoDB tiles, geodesic cones, custom SVG pins)
  - **Telemetry Charting**: Recharts (Dual-axis atmospheric wind and pressure timeline)
  - **Icons**: Lucide React
- **Backend**:
  - **Framework**: FastAPI
  - **Web Server**: Uvicorn
  - **Schema Validation**: Pydantic v2 & Pydantic-Settings
  - **Language**: Python 3.14
- **Database & Spatial Store (Ready)**:
  - PostgreSQL 15+ with **PostGIS** spatial extension (`POINT`, `POLYGON`, GiST spatial indices)
  - SQLAlchemy 2.0 ORM models
- **AI / ML & Physics (Prepared for Phase 2/3)**:
  - PyTorch 2.14+, OpenCV, NumPy, SciPy

---

## 4. Repository Folder Layout

```
cyclone-x/
├── ARCHITECTURE.md                  # Comprehensive technical specification
├── PROJECT_PLAN.md                  # 28-module roadmap & milestones
├── README.md                        # Documentation and startup guide
├── .env.example                     # Root environment variable template
│
├── data/                            # Scientific Dataset Repository (Phase 7)
│   ├── raw/                         # Authentic raw archives (IBTrACS, IMD, MOSDAC, GPM, ERA5, WorldPop, OSM)
│   ├── processed/                   # Normalized datasets (tracks, satellite, rainfall, environmental)
│   └── manifests/                   # Leakage-free train/val/test manifests (trajectory_train.csv, etc.)
│
├── backend/                         # FastAPI Backend Core
│   ├── requirements.txt             # Python dependencies
│   ├── app/
│   │   ├── main.py                  # FastAPI application entrypoint
│   │   ├── core/                    # Config, logging, provenance, exceptions
│   │   ├── api/v1/                  # Endpoints: datasets, health, cyclones, track, prediction, impact, satellite
│   │   ├── data/                    # Phase 7 central data pipeline (adapters, processors, registry, QC, manifests)
│   │   ├── impact/                  # Phase 6 multi-hazard disaster risk intelligence engine
│   │   ├── ml/                      # Machine learning models, Holland vortex, trajectory, vision preprocessors
│   │   └── schemas/                 # Pydantic schemas for all operations
│   └── tests/                       # Complete pytest suite (103 automated tests, 100% passing)
│
├── frontend/                        # Next.js 14 Command-Center Frontend
│   ├── src/
│   │   ├── app/                     # Next.js App Router (/, /datasets, /risk-intelligence)
│   │   ├── components/
│   │   │   ├── datasets/            # Scientific Data Hub (DatasetsView.tsx, modals)
│   │   │   ├── replay/              # Historical Replay (ReplayView.tsx with real IBTrACS/IMD tracks)
│   │   │   ├── satellite_ai/        # Satellite AI Analysis & Sensor Metadata Card (SatelliteAIView.tsx)
│   │   │   ├── risk_intelligence/   # Disaster Risk Command Center (RiskIntelligenceView.tsx)
│   │   │   ├── prediction/          # Trajectory Studio & Uncertainty Cone
│   │   │   ├── digital_twin/        # Holland Vortex Recharts Viewer
│   │   │   └── map/                 # Leaflet GIS canvas & risk swaths
│   │   ├── lib/                     # API client & constants
│   │   └── types/                   # TypeScript interfaces
│
└── infra/                           # Infrastructure & Deployment

    └── postgis-init/
        └── 01_init_postgis.sql      # PostGIS spatial DDL & indices
```

---

## 5. Installation & Getting Started

### Prerequisites
- **Node.js**: v18+ (tested on Node v24.20.0)
- **Python**: 3.10+ (tested on Python 3.14.7)

### 5.1 Backend Setup & Startup

1. Open a terminal in the project root:
   ```bash
   cd backend
   ```
2. Install Python dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Run backend tests:
   ```bash
   python -m pytest tests -v
   ```
4. Start the FastAPI development server:
   ```bash
   python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
   ```
   - API Root: `http://127.0.0.1:8000`
   - Health Check: `http://127.0.0.1:8000/api/health`
   - Interactive Swagger Docs: `http://127.0.0.1:8000/docs`

### 5.2 Frontend Setup & Startup

1. Open a second terminal:
   ```bash
   cd frontend
   ```
2. Install Node dependencies:
   ```bash
   npm install
   ```
3. Start the Next.js development server:
   ```bash
   npm run dev
   ```
4. Open your browser at `http://localhost:3000`.

---

## 6. API Route Specification

| HTTP Method | Route | Description | Status |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | Service health status check | Active (`{"status": "healthy"}`) |
| `GET` | `/api/cyclones` | List registered active/benchmark cyclones | Active (Amphan, Biparjoy) |
| `GET` | `/api/cyclones/{id}` | Current cyclone observation & telemetry | Active |
| `GET` | `/api/cyclones/{id}/track` | Historical best-track observations | Active |
| `GET` | `/api/cyclones/{id}/prediction` | +6h, +12h, +24h trajectory & uncertainty cone | Active (Baseline demo) |
| `GET` | `/api/cyclones/{id}/risk` | Multi-hazard risk assessment & surge depth | Active |
| `GET` | `/api/cyclones/{id}/environment` | Sea surface temperature, wind shear, MSLP | Active |
| `GET` | `/api/ml/status` | AI/ML pipeline status & model availability | Active |
| `POST` | `/api/data/upload` | Ingest user-uploaded observations & satellite rasters | Active (Phase 3) |
| `GET` | `/api/data/status` | Active data mode, dataset counts, integrity status | Active (Phase 3) |
| `GET` | `/api/data/sources` | Catalog of configured real meteorological sources | Active (Phase 3) |
| `GET` | `/api/provenance` | Immutable data provenance audit trail | Active (Phase 3) |
| `POST` | `/api/analysis/satellite` | Computer-vision cyclone candidate & center detection | Active (Phase 4) |
| `GET` | `/api/analysis/{id}` | Structured satellite analysis report retrieval | Active (Phase 4) |
| `GET` | `/api/analysis/{id}/explanation` | Explainable AI reasoning & visual feature attribution | Active (Phase 4) |
| `GET` | `/api/analysis/{id}/overlay` | Stream rendered transparent PNG detection overlay | Active (Phase 4) |
| `POST` | `/api/prediction/forecast` | Physics-guided trajectory advection forecaster (+6h, +12h, +24h) | Active (Phase 5) |
| `GET` | `/api/prediction/vortex/{id}` | Dynamic Holland (1980) vortex profile & asymmetric quadrant radii | Active (Phase 5) |
| `GET` | `/api/prediction/models` | Trajectory model registry & training statuses | Active (Phase 5) |
| `POST` | `/api/impact/analyze` | Run multi-hazard impact and risk analysis | Active (Phase 6) |
| `GET` | `/api/impact/{id}` | Retrieve cached/live impact & risk assessment | Active (Phase 6) |
| `GET` | `/api/impact/{id}/risk-map` | Multi-hazard GeoJSON risk swaths & asset layers | Active (Phase 6) |
| `GET` | `/api/impact/{id}/exposure` | Population & critical infrastructure exposure | Active (Phase 6) |
| `GET` | `/api/impact/{id}/provenance` | Cryptographic SHA-256 provenance chain record | Active (Phase 6) |
| `POST` | `/api/impact/scenario` | What-If counterfactual scenario perturbation studio | Active (Phase 6) |
| `GET` | `/api/impact/models` | Impact model registry & surrogate training statuses | Active (Phase 6) |

---

## 7. Demo Mode & Scientific Integrity Rules

- **Offline / Demo Mode**: Enabled by default. Uses high-fidelity, verified historical benchmark records for Super Cyclone Amphan (May 2020) and Extremely Severe Cyclonic Storm Biparjoy (June 2023).
- **Explicit Labeling**: Every payload and UI card prominently displays `DEMO DATA`.
- **Zero Hallucination Guarantee**:
  - Untrained models (`cyclone_cnn`, `cyclone_vit`, `intensity_dvorak_net`, `trajectory_bilstm`, `trajectory_transformer`, `surge_deep_surrogate`) strictly report `NOT_TRAINED`.
  - Intensity estimation truthfully reports `available: false` and `status_message: "Intensity estimation unavailable — trained intensity model required."` Zero fake wind speeds or pressures are invented.
  - Geographic coordinates are calculated solely when authentic bounding boxes are provided; otherwise pixel coordinates are returned with `"Geographic center unavailable — image georeferencing required."`
  - Trajectory forecasting runs on deterministic physical advection principles (momentum, 500 hPa steering, Rossby beta-drift) with transparent mathematical formulas rather than fabricated neural weights.
  - Environmental observations (SST, wind shear) are strictly required for thermodynamic modulation; missing inputs produce `INTENSITY_MODULATION_UNAVAILABLE_MISSING_OBSERVATIONS` rather than silent substitution.
  - Multi-hazard risk scoring uses a deterministic physical force index (0–100) or `UNKNOWN / INSUFFICIENT DATA`; no fabricated probability percentages or uncalibrated casualty numbers.
  - Coastal surge is explicitly disclaimed as an inverted-barometer proxy (`EXPERIMENTAL SCENARIO — NOT AN OFFICIAL STORM-SURGE FORECAST`).

---

## 8. Current Implementation Status

- [x] **Phase 1: Project Foundation (COMPLETED)**
  - Full production repository structure (Next.js + TypeScript + Tailwind, FastAPI + Pydantic + Uvicorn).
  - PostgreSQL / PostGIS database architecture ready.
  - Initial end-to-end integration verified.
- [x] **Phase 2: Command Center UI (COMPLETED)**
  - Professional dark command-center UI with Leaflet GIS mapping.
  - Historical tracks, predicted tracks, and uncertainty cone rendering.
  - Telemetry dual-axis wind/pressure charts and environmental telemetry cards.
- [x] **Phase 3: Scientific Data Intelligence & Provenance Layer (COMPLETED)**
  - Comprehensive Pydantic observation schemas for cyclone, satellite, and environmental data.
  - Adapters for IMD/IBTrACS CSV/JSON, user upload ingestion, and cryptographic SHA-256 provenance tracking.
  - Dedicated Data Intelligence UI with dataset status, upload interface, and provenance ledger.
- [x] **Phase 4: Satellite Image Analysis, Cyclone Detection & Center Detection (COMPLETED)**
  - Computer vision preprocessing with Laplacian variance sharpness and dynamic range assessment.
  - Morphological candidate storm segmentation and gradient spiral curvature alignment.
  - Circulation center localization via radial gradient symmetry analysis and convective mass centroid.
  - Equirectangular spatial georeferencing without coordinate fabrication.
  - Structural classification across 7 standardized morphological categories with experimental disclaimers.
  - Zero-fabrication intensity capability interface returning explicit unavailable status.
  - Explainable AI engine generating transparent PNG detection overlays and natural-language visual reasoning.
  - Central ModelRegistry tracking active baseline and cataloging untrained deep networks.
  - Dedicated "Satellite AI Analysis" mission-control UI with overlay toggles, sample presets, and real-time canvas.
- [x] **Phase 5: Trajectory Prediction, Physics Vortex Engine & Uncertainty Quantification (COMPLETED)**
  - Physics-guided trajectory advection forecaster (+6h, +12h, +24h) integrating inertial momentum vector, 500 hPa environmental steering flow, and planetary Rossby beta-drift deflection ($\beta = 2\Omega\cos(\phi)/R_{\text{earth}}$).
  - Analytical Holland (1980) vortex state engine computing continuous radial gradient wind curve $V(r)$, dynamic Holland shape parameter $B$, Knaff-Zehr $R_{\max}$ core eyewall boundary, and asymmetric quadrant gale radii (R34, R50, R64) with forward-right constructive translation velocity vector addition.
  - Bivariate normal error covariance propagation ($\sigma_{\parallel}$ along-track, $\sigma_{\perp}$ cross-track) generating a closed-loop GeoJSON Cone of Uncertainty polygon envelope.
  - Zero-Fabrication Guardrail: Deep neural trajectory models (`trajectory_bilstm`, `trajectory_transformer`) cataloged transparently as `NOT_TRAINED`. Active baseline `physics_guided_advection` and `holland_vortex_v1` deployed as `ACTIVE v0.5.0`.
  - Mission-Control Workspace: `TrajectoryStudio` with interactive timeline scrubbing (NOW, +6H, +12H, +24H), advection steering sliders, and live cone recomputation; `HollandVortexViewer` in Digital Twin with continuous $V(r)$ profile, RMW reference marker, and quadrant gale table.
  - 55 automated backend tests passing (100% passing) and zero-error Playwright browser validation.
- [x] **Phase 6: AI Impact & Disaster Risk Intelligence (COMPLETED)**
  - Modular multi-hazard backend engine (`backend/app/impact/`): Wind hazard swaths (Beaufort 34/48/64 kt), Rainfall risk with honest `NO_RAINFALL_DATA` handling, Runoff/flood vulnerability index, Inverted barometer coastal surge proxy with disclaimer, Geodesic population and infrastructure exposure intersection, and deterministic 0–100 Multi-Hazard Risk Scoring.
  - Explainable AI Why Factor Attribution: Rule-based breakdown attributing primary risk drivers and physical consequences.
  - Interactive What-If Scenario Studio: Real-time counterfactual perturbation sliders ($\Delta V_{\max}$, $\Delta \text{Pressure}$, translation speed) with strict `USER PERTURBATION` badges.
  - Cryptographic Provenance Chaining: SHA-256 audit ledger linking each impact assessment to parent forecast/observation hashes under `IMPACT_INTELLIGENCE_ENGINE`.
  - Frontend Risk Intelligence Command Center: Dedicated `/risk-intelligence` page and dashboard "Risk & Impact" tab featuring interactive Leaflet risk map, 6 hazard cards, AI why panel, and MoES/IMD non-replacement disclaimers.
  - 81 automated tests passing (100% passing across Phases 1–6) and zero-error Playwright browser validation.
- [ ] **Phase 7: Decision Support & Automated PDF Disaster Briefings (NEXT)**

