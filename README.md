# Driver Distraction & Fatigue Detection System

A real-time facial detection and tracking prototype that identifies driver distraction and fatigue using computer vision and deep learning, deployed on a Raspberry Pi edge device. Built as a Technology Capstone Research Project at the University of Canberra.

## Overview

Driver distraction and fatigue are major contributors to road accidents. This project explores whether a lightweight, real-time facial detection and tracking system can run entirely on low-cost edge hardware — without relying on cloud infrastructure — to monitor a driver's face and flag signs of distraction.

The system captures live video from a camera connected to a Raspberry Pi, detects the driver's face, identifies key facial landmarks (eyes, mouth, eyebrows), and tracks these features across frames to support real-time distraction/fatigue analysis.

## Problem Statement

Many existing facial detection frameworks require high-performance or cloud-based computing environments, making them difficult to deploy on lightweight edge devices. This project addresses that gap by designing a system that can detect facial landmarks and maintain stable face tracking in real time, entirely on affordable, local hardware.

## Objectives

1. Detect a human face from live camera input in real time.
2. Detect facial landmarks — eyes, mouth, and eyebrows.
3. Track facial movement across consecutive video frames.
4. Ensure the detection/tracking pipeline runs in real time on constrained hardware.
5. Deploy and evaluate the system on a Raspberry Pi.

## Scope

**In scope:**
- Real-time facial detection and landmark tracking (eyes, mouth, eyebrows)
- Processing of live camera input
- Deployment and testing on a Raspberry Pi edge device
- Performance evaluation of the detection/tracking pipeline

**Out of scope:**
- Integration with commercial vehicle software
- Individual person identification/recognition
- Emotion or behavioural analysis
- Large-scale/commercial vehicle deployment
- Cloud or external database integration (all processing is local)

## Approach

- **Face & landmark detection:** A lightweight deep learning model is used instead of traditional methods (e.g. Viola–Jones) for better robustness to lighting, angle, and partial occlusion, while remaining efficient enough for the Raspberry Pi.
- **Model architecture:** Baseline CNN, improved with a CNN + LSTM architecture to capture temporal patterns across frames, with model comparison to select the best performer.
- **Explainability:** Grad-CAM is used to visualize what the model attends to when making predictions.
- **Alerting:** Custom alert logic flags detected distraction/fatigue in real time.
- **Why this approach:** Cloud-based processing and commercial APIs (e.g. Amazon Rekognition) were considered but rejected due to latency, network dependency, cost, and privacy concerns around transmitting live facial video — all of which conflict with the project's edge-first, privacy-preserving goals.

## Tech Stack

- **Language:** Python
- **Computer vision:** OpenCV
- **Machine learning:** PyTorch (CNN, CNN+LSTM)
- **Explainability:** Grad-CAM
- **Hardware:** Raspberry Pi (Single Board Computer) + camera module
- **Dev tools:** VS Code, PyCharm
- **Training/compute:** Google Colab / Kaggle Kernels (GPU access for model training)
- **Version control & collaboration:** GitHub, Google Drive

## Dataset

Trained/validated using a publicly available Kaggle dataset for driver behaviour detection (Driver Monitoring Dataset), e.g. [Driver behavior detection by CNN](https://www.kaggle.com/code/minanabil11111212/driver-behavior-detection-by-cnn/output).

## System Requirements

| Component | Notes |
|---|---|
| Raspberry Pi | Edge inference device (~$40–$80) |
| Camera module | Captures live video for analysis (~$20–$50) |
| Python 3.x | Core runtime |
| OpenCV, PyTorch | Computer vision & ML libraries |

## Project Pipeline

1. **Dataset acquisition & EDA** — source and explore the driver behaviour dataset
2. **Data preprocessing** — cleaning, splitting, and augmentation
3. **Model development** — baseline CNN → improved CNN+LSTM model → model comparison
4. **Explainable AI** — Grad-CAM visualizations for model interpretability
5. **Alert logic** — real-time flagging of detected distraction/fatigue
6. **Hardware integration** — deployment onto Raspberry Pi + camera
7. **Testing & benchmarking** — accuracy, precision/recall, and real-time performance evaluation
8. **Reporting & presentation** — final report, poster, and demo

## Success Criteria

- Model achieves **≥85%** classification accuracy on the held-out test set
- Precision and recall **≥80%** for major distraction classes
- Minimal false negatives for high-risk behaviours (e.g. texting)
- Stable real-time performance across varying lighting conditions and head positions

## Project Structure

```
.
├── data/
│   └── driver_dataset/            # Raw/processed dataset (Kaggle DMD or similar)
├── notebooks/
│   ├── eda.ipynb                  # Exploratory data analysis
│   └── model_training.ipynb       # CNN / CNN+LSTM training & comparison
├── src/
│   ├── preprocessing.py           # Data cleaning, splitting, augmentation
│   ├── model.py                   # Model architecture(s)
│   ├── gradcam.py                 # Grad-CAM explainability
│   ├── alert.py                   # Alert logic
│   └── main.py                    # Real-time capture + inference pipeline (Raspberry Pi)
├── models/
│   └── best_model.pt              # Saved trained model
├── requirements.txt
└── README.md
```

*(Adjust file/folder names to match your actual implementation.)*

## Usage

1. Clone the repository:

```bash
git clone <your-repo-url>
cd driver-distraction-fatigue-tracker
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train or evaluate the model (via notebooks or scripts):

```bash
python src/model.py
```

4. Run the real-time detection pipeline (on Raspberry Pi with camera attached):

```bash
python src/main.py
```

## Privacy & Ethics

- No facial recognition or individual identification is performed.
- No video or facial data is transmitted to external servers or cloud services — all processing happens locally on-device.
- The project focuses solely on detecting distraction/fatigue indicators, not identity.

## Team

| Name | Role |
|---|---|
| Michael Marin | Project Lead / Scrum Lead |
| George Djakovic | ML Engineer |
| Kasra Zahedpur | Data Engineer |
| Alex Clare | IoT Engineer |

## Risks & Limitations

- Facial detection accuracy may degrade under specific conditions (e.g. glasses, low light) — documented as a known limitation.
- Real-time performance is constrained by the Raspberry Pi's limited compute power, requiring lightweight model design.
- Model accuracy is dependent on dataset quality and class balance.

## References

- Viola, P., & Jones, M. (2001). *Rapid object detection using a boosted cascade of simple features.* IEEE CVPR.
- Zhang, H., Liu, J., & Wang, T. (2024). *Real-time driver monitoring using deep learning and computer vision.*
- Li, X., Zhang, Y., & Chen, L. (2025). *Facial landmark detection and real-time tracking using lightweight deep learning models.*
- Sinha, N. (2023). *Drowsiness Detection in Drivers Using Deep Learning.* GitHub Repository.
- Minanabil11111212 (n.d.). *Driver behavior detection by CNN.* Kaggle.

## Academic Context

This project is being developed as a Technology Capstone Research Project at the University of Canberra, supervised by an academic mentor and industry sponsor, following an Agile/Scrum-style team structure over a 12-week period.

## License

This project was created for academic purposes as part of a university capstone project.
