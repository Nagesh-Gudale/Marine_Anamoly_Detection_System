# Marine Anomaly Detection

### AI-Powered Underwater Marine Debris & Anomaly Detection Using Side-Scan Sonar Imagery

An intelligent computer vision system for detecting, classifying, and geospatially mapping underwater marine debris and anomalies from Side-Scan Sonar (SSS) imagery.

---

## Overview

Underwater marine debris such as ghost nets, fishing gear, metal objects, and other anthropogenic materials poses a significant threat to marine ecosystems and navigation safety. Identifying these objects currently relies heavily on manual inspection of Side-Scan Sonar surveys, making the process time-consuming, subjective, and difficult to scale.

**Marine Anomaly Detection** aims to automate this process using deep learning and sonar-image analysis. The system processes Side-Scan Sonar imagery, detects underwater objects, classifies known debris categories, identifies unfamiliar anomalies, and converts detections into geospatially actionable information.

---

## Objectives

* Automate the detection of underwater marine debris and anomalies.
* Classify detected objects into predefined categories.
* Identify objects outside the known training classes as **unknown anomalies**.
* Apply sonar-specific preprocessing while preserving important debris signatures.
* Use acoustic-shadow characteristics as an additional confidence signal.
* Convert detected sonar objects into geospatial coordinates.
* Provide an intuitive dashboard for reviewing detections and prioritizing cleanup or inspection.

---

## Features

### 1. Sonar Image Preprocessing

A preprocessing pipeline designed specifically for Side-Scan Sonar imagery.

* Noise reduction and despeckling
* Contrast enhancement
* Image normalization
* Optional upscaling
* Comparison of filtering techniques such as Lee and Frost filtering
* Preservation of object highlights and acoustic shadows

### 2. AI-Based Object Detection

Deep learning models detect and localize underwater objects in sonar images.

* Bounding-box-based object detection
* Multi-class debris recognition
* Confidence score generation
* Support for custom-trained detection models

### 3. Marine Debris Classification

The system can be trained to recognize categories such as:

* Ghost nets / fishing nets
* Ropes and cables
* Metal debris
* Tires
* Containers
* Other anthropogenic objects
* Unknown / unclassified anomalies

> The final class list depends on the datasets and annotations used during development.

### 4. Acoustic-Shadow Reasoning

Side-Scan Sonar objects often produce acoustic shadows based on their height, shape, and position relative to the sonar sensor.

Marine Anomaly Detection explores acoustic-shadow characteristics as an additional reasoning signal to:

* Support object-confidence estimation
* Reduce visually ambiguous detections
* Distinguish potential objects from image artifacts
* Improve interpretability of model predictions

### 5. Unknown Anomaly Detection

Real-world marine environments contain objects that may not be represented in the training dataset.

The system is designed to flag detections that:

* Have low classification confidence
* Differ significantly from known object categories
* Require expert review
* May represent previously unseen debris or anomalies

### 6. Geospatial Mapping

Detections can be associated with geographic coordinates using sonar survey metadata.

* Convert image coordinates into geographic coordinates
* Display detected objects on an interactive map
* Record object category, confidence, and location
* Support prioritization of areas for inspection or cleanup

### 7. Interactive Dashboard

A web-based interface for visualizing and reviewing sonar analysis results.

* Upload sonar imagery
* View preprocessing results
* Inspect detected objects
* Review confidence scores
* Explore geospatial locations
* View detection summaries and reports

---

## System Architecture

```text
Side-Scan Sonar Imagery
          │
          ▼
┌──────────────────────────────┐
│ Data Ingestion & Validation  │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Sonar Preprocessing          │
│ Denoising • Enhancement      │
│ Normalization • Upscaling    │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ AI Object Detection          │
│ Detection + Localization     │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Classification & Confidence  │
│ Known Classes + Unknown      │
└──────────────┬───────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌───────────────┐ ┌──────────────────────┐
│ Acoustic      │ │ Geospatial           │
│ Shadow        │ │ Coordinate Mapping   │
│ Reasoning     │ │                      │
└───────┬───────┘ └──────────┬───────────┘
        │                    │
        └─────────┬──────────┘
                  ▼
┌──────────────────────────────┐
│ Results Fusion & Prioritizing│
│ Confidence • Object Priority │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Web Dashboard & Visualization│
│ Images • Reports • Map       │
└──────────────────────────────┘
```

---

## Technology Stack

| Component                | Technology                                          |
| ------------------------ | --------------------------------------------------- |
| Frontend                 | React.js / TypeScript                               |
| Styling                  | Tailwind CSS                                        |
| Build Tool               | Vite                                                |
| Backend                  | Python / FastAPI                                    |
| Deep Learning            | PyTorch                                             |
| Object Detection         | YOLO or other suitable detection architecture       |
| Image Processing         | OpenCV, NumPy                                       |
| Geospatial Visualization | Leaflet / Map-based visualization                   |
| Data Storage             | PostgreSQL / SQLite / JSON, depending on deployment |
| Model Training           | GPU-enabled cloud or local environment              |

> The final model architecture and database choices may evolve during experimentation.

---

## Project Structure

```text
Marine_anamoly_detection/
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── App.tsx
│   ├── package.json
│   └── vite.config.ts
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   ├── models/
│   │   ├── services/
│   │   └── utils/
│   ├── requirements.txt
│   └── README.md
│
├── models/
│   ├── weights/
│   └── configs/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── annotations/
│
├── notebooks/
│   ├── preprocessing/
│   ├── exploratory_analysis/
│   └── model_training/
│
├── docs/
│   ├── architecture/
│   └── research/
│
├── .gitignore
└── README.md
```

---

## Dataset Strategy

The project uses publicly available Side-Scan Sonar datasets and relevant marine-debris imagery where licensing and access permit.

### Planned Data Workflow

1. Collect publicly available SSS datasets.
2. Identify relevant debris and underwater-object categories.
3. Standardize image formats and metadata.
4. Annotate or verify object labels.
5. Apply preprocessing and augmentation.
6. Address class imbalance.
7. Split data into training, validation, and test sets.
8. Evaluate model performance on unseen samples.

### Potential Dataset Sources

* [GhostNetZero](https://www.ghostnetzero.com/)
* [SeaBedObjects Dataset](https://www.kaggle.com/)
* Public Side-Scan Sonar datasets
* Research datasets released with relevant publications

> Verify the license, availability, and suitability of each dataset before using it in training or redistribution.

---

## Research Foundation

The project is informed by research in underwater object detection, Side-Scan Sonar processing, marine anomaly detection, and geospatial mapping.

| Research Area                      | Reference                                                                                                                                        |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Underwater object detection        | [Detection of Underwater Objects Based on Machine Learning](https://www.jstage.jst.go.jp/article/jjasnaoe/18/0/18_115/_article/-char/en)         |
| Side-Scan Sonar mapping            | [High-Resolution Underwater Mapping Using Side-Scan Sonar](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0146396)            |
| Automated sonar object recognition | [A Machine Vision Meta-Algorithm for Automated Recognition of Underwater Objects Using Sidescan Sonar Imagery](https://arxiv.org/abs/1909.07763) |
| Physics-informed sonar perception  | [Physics-Informed Side-Scan Sonar Perception](https://www.mdpi.com/1424-8220/26/6/1938)                                                          |

---

## Installation

### Prerequisites

* Python 3.10+
* Node.js 18+
* npm
* Git
* GPU recommended for model training

### Clone the Repository

```bash
git clone https://github.com/<your-username>/Marine_anamoly_detection.git
cd Marine_anamoly_detection
```

### Backend Setup

```bash
cd backend

python -m venv venv
```

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the backend:

```bash
uvicorn app.main:app --reload
```

### Frontend Setup

Open a new terminal:

```bash
cd frontend
npm install
npm run dev
```

---

## Usage

1. Launch the frontend and backend.
2. Upload a supported Side-Scan Sonar image.
3. Run the preprocessing pipeline.
4. Submit the image for AI inference.
5. Review detected objects and confidence scores.
6. Inspect acoustic-shadow reasoning results, where available.
7. View geospatial information if survey coordinates are provided.
8. Export or review the detection results.

---

## Evaluation Metrics

### Object Detection

* Precision
* Recall
* F1-score
* mAP@50
* mAP@50:95
* Inference time

### Classification & Unknown Detection

* Per-class precision and recall
* Confusion matrix
* Unknown-detection recall
* False-positive rate
* Confidence calibration

### System-Level Evaluation

* Geospatial localization accuracy
* Processing time per image
* Robustness to sonar noise
* Performance across different seabed conditions

---

## Development Roadmap

* [x] Define the problem and system architecture
* [x] Review relevant research papers
* [ ] Collect and validate SSS datasets
* [ ] Build the preprocessing pipeline
* [ ] Prepare annotations and class definitions
* [ ] Train and evaluate baseline detection models
* [ ] Implement unknown-anomaly handling
* [ ] Develop acoustic-shadow reasoning module
* [ ] Implement geospatial coordinate mapping
* [ ] Integrate backend inference API
* [ ] Build the interactive dashboard
* [ ] Conduct end-to-end testing
* [ ] Deploy a working prototype

---

## Potential Applications

* Marine debris monitoring
* Ghost-net detection and recovery
* Underwater environmental surveys
* Seabed inspection
* Marine conservation
* Port and coastal infrastructure inspection
* Underwater archaeology
* Research and oceanographic exploration

---

## Limitations

* Performance depends on the quality and diversity of training data.
* Sonar appearance varies with sensor type, frequency, altitude, and seabed conditions.
* Unknown-anomaly detection requires careful validation to avoid false alarms.
* Geospatial accuracy depends on the availability and quality of survey metadata.
* Model predictions should be reviewed by domain experts before operational decisions.

---

## Team

**Project:** Marine Anomaly Detection
**Repository:** `Marine_anamoly_detection`
**Event:** Smart India Hackathon 2026
**Problem Statement:** SIH26057


We acknowledge the researchers, open-source contributors, and organizations providing Side-Scan Sonar datasets and tools that support this project.
