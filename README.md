PatchRift
Multi-modal, Sun-angle and Scale-invariant Image Correspondence using Chandrayaan-2 Optical Images

PatchRift is a scientific image-correspondence pipeline designed to establish correspondence between lunar surface images acquired under different imaging conditions.

The system is designed around Chandrayaan-2 optical imagery from OHRC, TMC, and IIRS, and addresses variations in solar illumination, spatial scale, viewing geometry, radiometric characteristics, and shadow geometry.

Overview

Conventional image matching can become unreliable when the same lunar surface is observed under substantially different illumination conditions, spatial scales, or viewing geometries.

PatchRift addresses these differences through a staged scientific processing pipeline:

Input Imagery
     │
     ▼
Stage I
Radiometric Preparation
     │
     ▼
Stage II
Solar Geometry & Shadow Analysis
     │
     ▼
Stage III
Geometric & Photometric Conditioning
     │
     ▼
Stage IV
Feature Detection & Description
     │
     ▼
Stage V
Feature Matching & Transformation
     │
     ▼
Image Correspondence
Pipeline Stages
Stage I — Radiometric Preparation

Prepares imagery for downstream processing through:

Radiometric preparation
Destriping
Validity-mask handling
IIRS spectral reduction to a scalar representation where required
Geometric planarity assessment
Stage II — Solar Geometry & Shadow Analysis

Processes solar and shadow information required for illumination-aware correspondence, including:

Solar geometry
Solar azimuth and elevation
Solar-geometry uncertainty propagation
Shadow segmentation
Shadow morphology and measurements
Geometric diagnostics
Stage III — Geometric & Photometric Conditioning

Conditions imagery to account for differences in spatial scale and illumination.

This stage includes:

Scale conditioning
Mask transport
Solar-aware photometric conditioning
Shading/trend removal
Validity propagation
Stage IV — Feature Extraction

Extracts correspondence features from conditioned image patches using the implemented frequency-domain feature-processing architecture.

The stage includes:

Keypoint detection
Log-Gabor filter-bank processing
Phase congruency
Moment analysis
Feature description
Stage V — Feature Matching

Matches feature sets between images and estimates the corresponding geometric transformation.

The stage includes:

Feature-distance computation
Mutual nearest-neighbour matching
Ratio-test filtering
Correspondence selection
Transformation estimation
Matching diagnostics
Mission Context

PatchRift is designed for the Chandrayaan-2 optical imaging ecosystem, with particular consideration for:

OHRC — Optical High Resolution Camera
TMC — Terrain Mapping Camera
IIRS — Imaging Infrared Spectrometer

The architecture separates instrument-specific preparation from the common image-correspondence pipeline.

Prototype Interface

A Streamlit-based prototype is included for demonstrating the PatchRift workflow.

Launch the interface from the repository root:

streamlit run src/patchrift/ui/app.py

The prototype provides a visual interface for uploading supported imagery and interacting with the processing pipeline.

Repository Structure
PatchRift/
│
├── src/
│   └── patchrift/
│       ├── cascade/
│       ├── features/
│       ├── geometry/
│       ├── io/
│       ├── matching/
│       ├── pipeline/
│       ├── preprocessing/
│       ├── shadows/
│       ├── ui/
│       └── utils/
│
├── tests/
│   ├── validation/
│   └── ...
│
├── docs/
├── .streamlit/
├── pyproject.toml
└── README.md
Validation

The current implementation is covered by an automated test suite.

Current status: 506 tests passing

506 passed

The test suite covers the implemented:

I/O and product handling
Radiometric preprocessing
IIRS processing
Geometry and solar calculations
Shadow processing
Image tiling
Pipeline stages
Feature extraction
Feature matching
Stage integration
UI input/output handling
Validation cases

Automated tests provide implementation-level validation using controlled test cases. Validation against real Chandrayaan-2 imagery is treated separately as end-to-end mission-data validation.

Installation

Clone the repository:

git clone https://github.com/stellarvector216/PatchRift.git
cd PatchRift

Install the project and its dependencies according to pyproject.toml.

Run the complete test suite:

pytest -q

Launch the prototype:

streamlit run src/patchrift/ui/app.py
Real-Data Validation

The automated test suite validates the implemented components and their interactions using controlled and synthetic cases.

The next stage of validation is execution against real Chandrayaan-2 imagery to evaluate the complete correspondence workflow on mission data.

Project

Developed for Smart India Hackathon (SIH) 2026.

Problem Statement

Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)

License

This project is distributed under the license specified in the LICENSE file.
