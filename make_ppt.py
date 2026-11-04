from pptx import Presentation
from pptx.util import Inches, Pt

# Create presentation
prs = Presentation()

# Slide 1: Title Slide
title_slide_layout = prs.slide_layouts[0]
slide = prs.slides.add_slide(title_slide_layout)
title = slide.shapes.title
subtitle = slide.placeholders[1]

title.text = "CYCLONE-X: AI-Driven Tropical Cyclone Identification, Classification & Prediction System[cite: 4]"
subtitle.text = "Department of Science & Technology and Biotechnology • Government of West Bengal[cite: 4]\n\nTeam: StormByte | Mentor: Avishek Dey"

# Helper function to add content slides
def add_slide(prs, title_text, bullet_points):
    bullet_slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(bullet_slide_layout)
    shapes = slide.shapes
    
    title_shape = shapes.title
    title_shape.text = title_text
    
    body_shape = shapes.placeholders[1]
    tf = body_shape.text_frame
    tf.text = bullet_points[0]
    
    for point in bullet_points[1:]:
        p = tf.add_paragraph()
        p.text = point
        p.level = 0

# Slide Data
slides_data = [
    ("01 Problem & Location", [
        "The Core Problem: Cyclones intensify and change direction rapidly, and satellite imagery arrives as massive, continuous data streams that are difficult to inspect manually[cite: 4].",
        "Target Location: Coastal West Bengal and the Bay of Bengal[cite: 4].",
        "Primary Focus: District-level monitoring and disaster-management workflows for exposed regions like South 24 Parganas and East Midnapore[cite: 4]."
    ]),
    ("02 Proposed Solution", [
        "Cyclone-X Platform: An AI-driven web platform combining multi-source satellite imagery (geostationary and polar-orbiting)[cite: 4].",
        "Core Capabilities: Automated cyclone/genesis detection, latitude/longitude storm-center tracking, and 24–48 hour trajectory prediction[cite: 4].",
        "User Experience: A single command dashboard for state disaster-management teams and meteorological analysts[cite: 4]."
    ]),
    ("03 Technology or Method", [
        "Data Layer: Ingests satellite feeds alongside historical IBTrACS best-track data and environmental parameters (SST, wind/shear)[cite: 4].",
        "AI/ML Layer: A dual-stream CNN/Vision Transformer for detection, and a ConvLSTM/Transformer for temporal trajectory modeling and uncertainty estimation[cite: 4].",
        "Geo Layer: Built on PostgreSQL/PostGIS for tracking coordinates and forecast paths, visualized through an interactive Leaflet map[cite: 4].",
        "Suitability: Handles time-series imagery, supports multiple data sources, and keeps model/data provenance traceable[cite: 4]."
    ]),
    ("04 Implementation Plan", [
        "1. DATA (ML/Data Team): Collect satellite + IBTrACS data[cite: 4].",
        "2. BACKEND (Backend Team): Build APIs, processing, and database[cite: 4].",
        "3. AI PIPELINE (ML Team): Train detection, intensity, and track models[cite: 4].",
        "4. FRONTEND (Frontend Team): Build map, time-lapse, and dashboard[cite: 4].",
        "5. INTEGRATION (Full Team): Connect real-time inference + UI[cite: 4].",
        "6. VALIDATION (Full Team): Historical case testing + metrics[cite: 4]."
    ]),
    ("05 Cost & Funding", [
        "Software & Development: ₹1.0–1.5 lakh (Development, testing, and integration)[cite: 4].",
        "Cloud / Compute: ₹0.6–1.0 lakh (Training/inference, storage, and deployment)[cite: 4].",
        "Data & Processing: ₹0.2–0.5 lakh (Processing, bandwidth, and supporting services)[cite: 4].",
        "Hardware / Field Setup: ₹0.3–0.7 lakh (Optional workstation/GPU and demonstration setup)[cite: 4].",
        "Validation & Pilot: ₹0.2–0.4 lakh (Testing, documentation, and pilot support)[cite: 4].",
        "Total Budget: Approximately ₹2.3–4.1 lakh, depending on cloud/GPU usage and deployment scale[cite: 4]."
    ]),
    ("06 Beneficiaries", [
        "Direct Users: State/district disaster-management teams, government monitoring units, and emergency-response planners[cite: 4].",
        "Communities: Millions of residents in exposed coastal districts, specifically South 24 Parganas, North 24 Parganas, East Midnapore, and West Midnapore[cite: 4].",
        "Scale of Benefit: The platform supports decision-making; it does not replace official alerts[cite: 4]."
    ]),
    ("07 Timeline", [
        "Weeks 1–2 (Foundation): Architecture, database, APIs[cite: 4].",
        "Weeks 3–5 (Data): Satellite ingestion + IBTrACS pipeline[cite: 4].",
        "Weeks 6–8 (AI): Detection + classification baseline[cite: 4].",
        "Weeks 9–10 (Forecast): Trajectory + uncertainty[cite: 4].",
        "Weeks 11–12 (Dashboard): Map + time-lapse + command center[cite: 4].",
        "Weeks 13–16 (Validation & Pilot): Historical cases testing and cloud deployment for a controlled pilot[cite: 4]."
    ]),
    ("08 Expected Impact", [
        "Social: Faster access to storm information, better visualization for response planning, and improved communication of location and uncertainty[cite: 4].",
        "Economic: Reduces manual monitoring workload, supports earlier operational preparation, and helps prioritize monitoring resources[cite: 4].",
        "Environmental: Continuous observation of cyclone evolution and a reusable platform for other climate-risk monitoring[cite: 4]."
    ]),
    ("09 Scalability", [
        "District to State: Can monitor additional districts by changing geographic bounds, local data feeds, and exposure layers[cite: 4].",
        "State to National: Can extend to other coastal basins by adding region-specific satellite coverage and local historical datasets[cite: 4].",
        "What Stays the Same: FastAPI services, PostGIS geospatial layer, ML pipeline, and web dashboard remain completely reusable[cite: 4]."
    ]),
    ("10 Prototype or Pilot Evidence", [
        "Current Prototype: Implemented FastAPI backend, Next.js dashboard, PostgreSQL/PostGIS schema, and interactive Leaflet map[cite: 4].",
        "Functionality: Multi-frame processing, coordinate extraction, and track visualization are complete[cite: 4].",
        "Next Validation Steps: Collect evidence on detection precision/recall, track error at 6/12/24/48h, and processing latency before operational use[cite: 4]."
    ])
]

# Generate slides
for title_text, points in slides_data:
    add_slide(prs, title_text, points)

# Save file
file_name = "Cyclone_X_WB_DSTBT_Submission_Presentation.pptx"
prs.save(file_name)
print(f"Presentation '{file_name}' generated successfully!")