# PatchRift

### Multi-modal, Sun-angle and scale-invariant image correspondence using Chandrayaan-2 optical imagery

PatchRift is a scientific image-correspondence pipeline designed for matching lunar surface imagery acquired under different imaging conditions.

The system processes optical imagery through radiometric preparation, solar-geometry analysis, shadow-based conditioning, feature extraction, and feature matching to establish correspondence between images while accounting for differences in scale, illumination, and viewing geometry.

---

## 🚀 Overview

Lunar images acquired by different instruments or under different observation conditions can exhibit substantial differences in:

- illumination and solar incidence angle
- spatial scale / ground sampling distance
- viewing geometry
- radiometric characteristics
- shadow geometry

PatchRift addresses these differences through a staged processing architecture rather than relying on direct pixel-level comparison.

### Pipeline

```text
Input Imagery
     │
     ▼
┌─────────────────────────┐
│ Stage I                 │
│ Radiometric Preparation │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ Stage II                │
│ Solar Geometry &        │
│ Shadow Analysis         │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ Stage III               │
│ Geometric & Photometric │
│ Conditioning            │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ Stage IV                │
│ Feature Detection &     │
│ Description             │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ Stage V                 │
│ Feature Matching &      │
│ Transformation           │
└────────────┬────────────┘
             ▼
      Image Correspondence
