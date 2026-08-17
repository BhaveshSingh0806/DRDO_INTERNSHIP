# Automatic Number Plate Recognition (ANPR) & Smart Gate Logistics Enforcement Engine

> **DRDO Summer Internship Project | Defence Laboratory, Jodhpur | Session 2025–26**

## 📌 Project Overview

The **Automatic Number Plate Recognition (ANPR) & Smart Gate Logistics Enforcement Engine** is a computer-vision-based system developed during a Summer Internship at the **Defence Laboratory, Jodhpur, Defence Research & Development Organisation (DRDO)**.

The system processes vehicle video streams to detect and track vehicles/license plates, recognize Indian vehicle registration numbers, stabilize OCR results across frames, and generate structured gate-logistics information.

The project addresses two major practical problems in video-based ANPR:

1. **Tracking fragmentation** — vehicle tracking IDs may change when a vehicle slows down or stops at a gate.
2. **OCR fluctuation** — license-plate characters may vary between consecutive frames because of blur, illumination, perspective, or low resolution.

The developed pipeline combines **YOLOv8, ByteTrack, EasyOCR, OpenCV-based image preprocessing, and a custom temporal memory/stabilization algorithm** to produce stable vehicle identification and automated parking/gate logs.

The report describes the system as a persistent database-sentinel layer capable of converting video frames into synchronized vehicle identification, arrival time, waiting duration, and departure information.

---

## 🎯 Objectives

- Detect vehicle/license-plate regions from video.
- Track detected vehicles continuously across frames.
- Recognize Indian vehicle registration numbers using OCR.
- Improve OCR reliability using multi-stage image preprocessing.
- Reduce character flickering across consecutive frames.
- Maintain vehicle identity despite tracking-ID changes.
- Detect when a vehicle stops/waits at the gate.
- Convert frame-level events into synchronized `HH:MM:SS` timestamps.
- Export structured vehicle-transaction data to CSV.
- Produce an annotated output video for visual verification.

---

## 🧠 System Architecture

```text
                    INPUT VIDEO
                         │
                         ▼
                ┌─────────────────┐
                │     YOLOv8      │
                │ Detection Model │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │    ByteTrack    │
                │ Multi-Object    │
                │    Tracking     │
                └────────┬────────┘
                         │
                         ▼
                  License Plate
                      Crop
                         │
                         ▼
              ┌─────────────────────┐
              │ Image Preprocessing │
              ├─────────────────────┤
              │ • 3× Upscaling      │
              │ • Grayscale         │
              │ • CLAHE             │
              │ • Gaussian Blur     │
              │ • Adaptive Threshold│
              │ • Otsu Binarization │
              │ • Sharpening        │
              └──────────┬──────────┘
                         │
                         ▼
                    EasyOCR
                         │
                         ▼
              Text Cleaning & Regex
                    Validation
                         │
                         ▼
          ┌─────────────────────────────┐
          │ IndianPlateMemoryManager    │
          ├─────────────────────────────┤
          │ • Spatial ID Recovery       │
          │ • Sliding History           │
          │ • Character Voting           │
          │ • Plate Locking              │
          │ • Length Filtering            │
          └──────────────┬──────────────┘
                         │
                         ▼
             Gate Logistics Analytics
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      Arrival Time   Waiting Time   Departure Time
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                  ┌─────────────┐
                  │ CSV Traffic │
                  │    Logs     │
                  └─────────────┘
                         │
                         ▼
                 Annotated Output
                      Video
```

---

## 🔧 Technology Stack

| Technology / Library | Role in the Project |
|---|---|
| **Python** | Main development language |
| **Ultralytics YOLOv8** | Vehicle/license-plate detection and tracking integration |
| **ByteTrack** | Multi-object tracking and tracking-ID continuity |
| **EasyOCR** | License-plate text recognition |
| **OpenCV** | Video processing, image transformation and output rendering |
| **NumPy** | Numerical and array operations |
| **Pandas** | Data processing and CSV generation |
| **SciPy** | Scientific/numerical processing support |
| **FilterPy** | Filtering/tracking support |
| **lap** | Assignment/association support |
| **Roboflow** | Dataset management and YOLO training workflow |
| **Google Colab** | Development and execution environment |

### Python Dependencies

The project report specifies the following installation packages:

```bash
pip install ultralytics easyocr filterpy lap scipy pandas
pip install roboflow
```

The implementation also directly imports/uses packages including OpenCV (`cv2`) and NumPy (`numpy`). The final `requirements.txt` should reflect the packages actually used by the submitted notebook.

---

## 🚀 Project Workflow

### Phase 1 — Detection Model

A YOLOv8 model is trained using a custom license-plate dataset.

Reported training configuration:

```text
Model: YOLOv8
Epochs: 25
Image Size: 640
```

The trained model is then used for inference and tracking.

### Phase 2 — Stabilized ANPR & Gate Logistics

The second phase performs:

1. Video frame acquisition.
2. YOLOv8 detection.
3. ByteTrack tracking.
4. License-plate cropping.
5. Image enhancement.
6. EasyOCR recognition.
7. OCR history accumulation.
8. Character-level voting.
9. Vehicle-ID recovery.
10. Plate locking.
11. Arrival/departure detection.
12. Waiting-time calculation.
13. CSV export.
14. Annotated video generation.

---

## 🖼️ Image Preprocessing Pipeline

Before OCR, each license-plate crop is processed through multiple transformations.

### 1. Upscaling

The plate is enlarged by a factor of:

```text
UPSCALE_FACTOR = 3
```

### 2. Grayscale Conversion

The color image is converted into a luminance/grayscale representation.

### 3. CLAHE

Contrast Limited Adaptive Histogram Equalization is applied using:

```text
clipLimit = 2.0
tileGridSize = (8, 8)
```

### 4. Gaussian Blur

A `(3, 3)` Gaussian kernel is used for noise reduction.

### 5. Adaptive Thresholding

Adaptive Gaussian thresholding is used to improve character/background separation.

### 6. Otsu Binarization

Otsu's thresholding method provides an additional binary representation.

### 7. Sharpening

A sharpening kernel is applied to emphasize character edges.

---

## 🔤 OCR Configuration

EasyOCR is initialized for English characters:

```python
reader = easyocr.Reader(['en'], gpu=True)
```

The OCR allowlist is restricted to:

```text
ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-
```

This reduces unwanted symbols and background-related OCR noise.

---

## 🧩 Custom ANPR Stabilization

The project introduces a custom `IndianPlateMemoryManager` to stabilize recognition over time.

### Dynamic Spatial ID Recovery

When a vehicle stops or its bounding box shifts, the system compares spatial centroids and maps a new tracking ID back to the previously associated vehicle.

The reported spatial recovery threshold is:

```text
60.0 pixels
```

### Sliding Recognition History

OCR results are stored in a bounded history buffer.

Reported configuration:

```text
max_history = 25
lock_threshold = 4
```

### Character Voting

The system determines the dominant text length and performs confidence-weighted character voting to construct a stable plate string.

### Plate Locking

Once the stabilized plate satisfies the required length and repeatedly appears in the history, the system locks the recognized plate.

### Non-Decreasing Length Filter

If a strong plate string has already been obtained, shorter degraded readings are discarded when the existing recognized plate has at least 9 characters.

This prevents low-quality later frames from corrupting a stable recognition.

---

## ⏱️ Smart Gate Logistics

The logistics sub-engine converts frame-level information into clock-based events.

The report uses the following baseline:

```text
Base Time: 03:00:00 PM
```

The system calculates:

```text
Arrival Time
Departure Time
Waiting Time
```

The report defines these metrics as:

```text
Arrival Time   = Clock format of first-seen frame
Departure Time = Clock format of last-seen frame
Waiting Time   = Duration corresponding to stopped frames
```

A vehicle is considered stopped when its centroid movement falls below the configured motion threshold.

Reported motion threshold:

```text
motion_delta < 1.5
```

---

# 📊 Exact Output Statistics Reported

The project report contains two output-statistics presentations from the execution/verification sections. They are reproduced here **exactly as reported**, without attempting to reconcile differences between the two recorded examples.

## Output Record — Verification Section

The report's database-table evaluation records:

| Track ID | Vehicle Number | Arrival Time | Departure Time | Waiting Time |
|---:|---|---|---|---|
| **23** | **RJ19CA9632** | **03:00:00** | **03:00:09** | **00:00:05** |

The report explains that this represents:

- Arrival at `03:00:00`
- Departure at `03:00:09`
- A waiting duration of `00:00:05`

## Output Record — Console Execution

The report's console output records:

| Track ID | Vehicle Number | Arrival Time | Departure Time | Waiting Time |
|---:|---|---|---|---|
| **1** | **RJ19CA9632** | **03:00:00** | **03:00:08** | **00:00:04** |

The corresponding execution output identifies the synchronized parking-management database as:

```text
Track_ID    Vehicle_Number    Arrival_Time    Departure_Time    Waiting_Time
1           RJ19CA9632        03:00:00        03:00:08          00:00:04
```

### Important note about the reported statistics

The report contains both records above in different sections. Therefore, this README preserves both values rather than silently selecting one as the "correct" result.

The visual verification section also reports a tracking ID of `23` with the recognized plate `RJ19CA9632`, including a terminal tracking snapshot at approximately `03:00:08 PM`.

---

## 📈 Before vs. After Optimization

| Feature | Before Optimization | After Optimization |
|---|---|---|
| **OCR Stability** | High flickering and recognition noise | Stable voting-based character lock |
| **ID Continuity** | Frequent ID switching | Persistent ByteTrack tracking |
| **Data Integrity** | Background artifacts and stray symbols | Regex and length-based validation |
| **Result Export** | No structured logging | Automated CSV traffic logs |

### Main Improvements

**OCR Stability:**  
A voting-based stabilization mechanism reduces frame-to-frame text flickering.

**ID Continuity:**  
ByteTrack combined with spatial ID recovery improves tracking persistence during vehicle deceleration and stopping.

**Data Integrity:**  
Regex cleaning and length-based validation remove unwanted symbols and degraded OCR strings.

**Result Export:**  
The system automatically produces synchronized CSV traffic logs.

---

## 📁 Output Data Format

The generated CSV contains:

```text
Track_ID
Vehicle_Number
Arrival_Time
Departure_Time
Waiting_Time
```

Example structure:

```csv
Track_ID,Vehicle_Number,Arrival_Time,Departure_Time,Waiting_Time
1,RJ19CA9632,03:00:00,03:00:08,00:00:04
```

> The example above corresponds to the console-output record reported in the internship report.

---

## 📂 Recommended Repository Structure

```text
DRDO-INTERNSHIP/
│
├── Dataset/
│   └── EXP VIDEO.mp4
│
├── Notebook/
│   └── DRDO_PROJECT.ipynb
│
├── Output/
│   ├── CSV TABLE.csv
│   ├── DRDO REPORT.pdf
│   └── EXP OUTPUT.mp4
│
├── requirements.txt
└── README.md
```

For GitHub, avoid committing very large video files or datasets if they exceed GitHub's repository/file-size limits. Keep large experimental assets separately when necessary.

---

## 💻 Running the Project

### 1. Clone or download the repository

```bash
git clone <your-repository-url>
cd DRDO-INTERNSHIP
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

Open:

```text
Notebook/DRDO_PROJECT.ipynb
```

The original project workflow was developed in Google Colab, so Colab-specific commands and paths may need to be adapted when running locally in VS Code.

### 4. Provide the required model and video files

The original notebook uses paths such as:

```text
/content/best.pt
/content/EXP VIDEO.mp4
/content/output/VIDEO.mp4
/content/output/VIDEO.csv
```

For local execution, replace these with paths appropriate to your repository.

---

## 📌 Key Configuration Parameters

The report specifies the following main inference parameters:

```python
YOLO_CONF = 0.35
OCR_INTERVAL = 1
LOCK_THRESHOLD = 4
OCR_CONFIDENCE = 0.20
MAX_HISTORY = 20
MIN_PLATE_WIDTH = 45
MIN_PLATE_HEIGHT = 15
UPSCALE_FACTOR = 3
```

The custom memory-manager implementation also uses:

```python
max_history = 25
lock_threshold = 4
```

These values are retained as reported in the project documentation.

---

## 🔒 Security & Repository Safety

**Do not commit API keys or credentials to GitHub.**

The project report's appendix contains a Roboflow API-key usage example. That credential should **not** be copied into the public repository or README.

Use an environment variable instead, for example:

```python
import os

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
```

Then store the actual key outside the repository.

Add sensitive/local configuration files to `.gitignore` where appropriate.

---

## 🔮 Future Scope

### 1. Edge Deployment

The report proposes quantizing/exporting the inference pipeline for edge hardware such as:

- NVIDIA Jetson
- Raspberry Pi

TensorRT can be considered for optimized deployment.

### 2. Automated Boom-Barrier Control

The system can be connected to GPIO relay hardware. After recognizing and authorizing a plate, the software could issue a control signal to operate a physical boom barrier.

### 3. Database & Cloud Integration

The current CSV-based logging approach can be extended to:

- PostgreSQL
- MySQL
- Distributed multi-gate databases
- Centralized command-center monitoring

This can support synchronized multi-gate vehicle logistics, anomaly monitoring, and authorized/blocked vehicle management.

---

## 🏆 Project Outcomes

The project successfully demonstrates an end-to-end ANPR and gate-logistics workflow capable of:

- Detecting vehicle/license-plate regions.
- Tracking vehicles across frames.
- Recognizing an Indian vehicle registration number.
- Stabilizing OCR results over time.
- Recovering vehicle identity when tracking IDs shift.
- Detecting gate waiting behavior.
- Generating synchronized time-based logistics data.
- Exporting structured CSV traffic logs.
- Producing an annotated output video for verification.

The reported visual execution demonstrates stable recognition of the vehicle number:

```text
RJ19CA9632
```

with persistent tracking and synchronized gate-time information.

---

## 📚 References

The internship report references:

- **Ultralytics YOLOv8** — real-time object detection/tracking resources.
- **EasyOCR** — deep-learning-based text extraction resources.
- **ByteTrack** — multi-object tracking and frame-to-frame association resources.
- **DRDO Technology Division Internship Proforma** — technical internship evaluation framework.

---

## 👨‍💻 Internship Information

| Field | Details |
|---|---|
| **Project** | Automatic Number Plate Recognition (ANPR) & Smart Gate Logistics Enforcement Engine |
| **Organization** | Defence Research & Development Organisation (DRDO) |
| **Centre** | Defence Laboratory, Jodhpur |
| **Internship Type** | Summer Internship Training |
| **Session** | 2025–26 |
| **Duration** | 01 June 2026 – 20 July 2026 |
| **Academic Program** | B.E. Artificial Intelligence & Data Science Engineering |
| **Year** | 3rd Year |
| **University** | MBM University |
| **Project Guide** | Sh. Umesh Chaturvedi, Scientist-E |

---

## 📜 Disclaimer

This repository is intended to document the software implementation and internship work described in the training report.

The project report states that it does not contain confidential information pertaining to DRDO. Nevertheless, repository contents should be reviewed before publication to ensure that no credentials, private datasets, internal files, or confidential information are committed.

---

**Developed as part of the DRDO Summer Internship Training — Defence Laboratory, Jodhpur.**
