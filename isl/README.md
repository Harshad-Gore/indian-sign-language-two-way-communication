# Sign Language Interpreter

**Real-time Indian Sign Language (ISL) recognition from a standard webcam.**
Translates 71 ISL signs into text and speech using a trained deep learning model. No GPU required, no cloud dependency.

---

## What It Does

Point your webcam at your hands and the system will:

- Recognize **71 Indian Sign Language signs** across 5 categories:
  - **Greetings**: Hello, How are you, Good Morning/Afternoon/Evening/Night, Thank you, Alright, Pleased
  - **Days & Time**: Sunday-Saturday, Today, Tomorrow, Yesterday, Week, Month, Year, Hour, Minute, Second, Morning, Afternoon, Evening, Night, Time
  - **Home**: Table, Chair, Bed, Window, Door, Kitchen, Bathroom, Bedroom, Book, Pen, Pencil, Key, Lock, Telephone, Bag, Box, Gift, and more
  - **Pronouns**: I, you, he, she, it, we, you (plural), they
  - **Seasons**: Summer, Spring, Winter, Fall, Season, Monsoon
- Build a **running sentence** from detected signs
- **Speak signs aloud** in real-time via text-to-speech
- Display a **live visualization** with hand skeleton overlay, confidence bars, and hold-time progress

---

## Quick Start

```bash
# Clone and set up
git clone https://github.com/yourusername/ISL.git
cd ISL
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Run the interpreter
python main.py
```

A window will open showing your camera feed with the interpreter overlay.

---

## Training from Dataset

If you have the ISL video dataset, you can retrain the model:

```bash
# Step 1: Extract landmarks from videos (one-time, ~40 minutes)
python main.py extract --data-dir data --output-dir extracted_data

# Step 2: Benchmark several model families and pick the best real checkpoint
python main.py benchmark --data-dir extracted_data --output-dir models/benchmark --device auto --epochs 180 --augment-repeats 3

# Step 3: Train the selected model more strongly if needed
python main.py train --data-dir extracted_data --output-dir models --model-type hybrid --device auto --epochs 220 --augment-repeats 3 --patience 45

# Step 4: Run the interpreter (uses trained model automatically)
python main.py
```

The dataset should be organized as:
```
data/
  Category/
    XX. SignName/
      video1.MOV
      video2.MP4
      ...
```

---

## Controls

| Key         | Action              |
|-------------|---------------------|
| `Q` / `ESC` | Quit               |
| `C`         | Clear sentence       |
| `R`         | Full reset           |
| `Backspace` | Undo last sign       |
| `E`         | Edit sentence        |
| `V`         | Voice settings       |

---

## How It Works

The system processes each camera frame through a trained ML pipeline:

```
Camera Frame
    |
    v
[Hand Tracker] ---- MediaPipe Hands (21 landmarks per hand)
    |                MediaPipe Pose  (body reference frame)
    v
[Feature Extractor] - Normalize landmarks to body center
    |                  Scale by shoulder width
    |                  158 features per frame:
    |                    Right hand (63) + Left hand (63) + Pose (30) + Flags (2)
    v
[ML Model] --------- Hybrid landmark model
    |                 Part-aware hand/pose streams
    |                 Temporal conv + BiGRU + attention pooling
    |                 30-frame sliding window with padding masks
    |                 Trained on 1120 ISL video samples
    v
[Sign Engine] ------ Temporal smoothing (weighted voting)
    |                 Confidence + margin + entropy gating
    |                 Hold-time commit after stable agreement
    |                 Cooldown between repeated signs
    v
[TTS Output] ------- Windows SAPI voice synthesis
                      Non-blocking subprocess isolation
```

### Model Architecture

- **Input**: 30-frame sequences of 158-dimensional normalized landmark features
- **Encoder**: Hybrid hand/pose streams + temporal conv + bidirectional GRU
- **Pooling**: Temporal attention plus masked mean pooling
- **Classifier**: 3-layer FC head with BatchNorm, ReLU, Dropout
- **Temporal features**: raw landmarks plus frame-to-frame velocity cues
- **Training**: AdamW optimizer, cosine annealing with warmup, label smoothing, mixup augmentation, padding-aware masking
- **Current saved-checkpoint validation accuracy**: 19.23% after 5 recorded epochs in `models/training_history.json`
- **Accuracy target**: 90%+ is a training goal; only claim it after `metrics.json` or `benchmark_results.csv` shows it.

### Data Augmentation

Training uses configurable augmentation to improve generalization:
- Time stretching (default 0.85x-1.18x speed)
- Random frame dropping
- Gaussian noise on landmarks
- Spatial jitter (small random translation)
- Optional temporal crop
- Optional left-right hand mirroring (`--mirror-augment`; off by default because mirroring can change sign meaning)

---

## Project Structure

```
ISL/
  main.py                          Entry point (interpret, extract, train, test)
  sign_to_text.py                  Standalone interpreter script
  requirements.txt                 Dependencies
  data/                            ISL video dataset (not included)
  extracted_data/                  Extracted landmark sequences
  models/                          Trained model checkpoints
  src/
    core/
      hand_tracker.py              MediaPipe Hands + Pose tracking
      hand_analyzer.py             Geometric analysis (finger states, palm, regions)
      speaker.py                   Text-to-speech via SAPI subprocess
    ml/
      extract_landmarks.py         Video-to-landmark extraction pipeline
      dataset.py                   PyTorch Dataset with augmentation
      model.py                     GRU + Attention model architecture
      train.py                     Training loop with scheduling and early stopping
      recognizer.py                Real-time ML inference with sliding window
    recognition/
      ml_engine.py                 ML-based sign engine (temporal smoothing + TTS)
      sign_engine.py               Rule-based engine (fallback)
      finger_spelling.py           ASL alphabet rules (fallback)
      gesture_recognizer.py        Dynamic gesture detection (fallback)
    pipeline/
      fast_pipeline.py             Camera loop, visualization, controls
  tests/
    test_ml_system.py              22 ML system tests
    test_new_system.py             21 rule-based system tests
```

---

## Command-Line Options

### Interpreter (default)

```bash
python main.py                               # Start interpreter
python main.py interpret --camera 1          # Use second camera
python main.py interpret --width 640 --height 480   # Lower resolution
python main.py interpret --verbose           # Print detections to terminal
```

### Extract landmarks

```bash
python main.py extract                       # Extract from data/ to extracted_data/
python main.py extract --data-dir path/to/videos --fps 15
```

### Train model

```bash
python main.py train --device auto                         # Train recommended hybrid model
python main.py train --epochs 220 --patience 45            # Longer high-accuracy run
python main.py train --model-type transformer              # Transformer candidate
python main.py train --model-type tcn                      # Fast TCN candidate
python main.py train --model-type lite                     # Smaller low-latency model
python main.py train --device cpu                          # Force CPU training
python main.py train --device cuda --batch-size 64         # Force NVIDIA GPU training
python main.py train --mirror-augment                      # Optional; use carefully
```

### Benchmark models

```bash
python main.py benchmark --device auto --epochs 180 --augment-repeats 3
python main.py benchmark --models hybrid transformer tcn --device cuda --epochs 220
```

### Run tests

```bash
python main.py test
python main.py test --coverage
```

---

## Requirements

| Dependency     | Version   | Purpose                        |
|----------------|-----------|--------------------------------|
| numpy          | >= 1.21   | Array operations               |
| opencv-python  | >= 4.6    | Camera capture, visualization  |
| mediapipe      | >= 0.10   | Hand and pose landmark tracking|
| torch          | >= 2.0    | ML model training and inference|
| pywin32        | any       | Text-to-speech (Windows SAPI)  |
| pytest         | >= 7.0    | Testing (dev only)             |

**Hardware**: any machine with a webcam. No GPU required (CPU inference is fast enough for real-time).
Tested on Python 3.11, Windows 10/11.

For training experiments from the repository root:

```bash
pip install -r training-rq.txt
```

Training automatically uses CUDA when `--device auto` detects a compatible GPU.

---

## Recognized Signs (71 classes)

| Category | Signs |
|----------|-------|
| **Greetings** (9) | Hello, How are you, Alright, Good Morning, Good afternoon, Good evening, Good night, Thank you, Pleased |
| **Days & Time** (21) | Sunday, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Today, Tomorrow, Yesterday, Week, Month, Year, Hour, Minute, Second, Morning, Afternoon, Evening, Night, Time |
| **Home** (27) | Table, Chair, Bed, Dream, Window, Door, Bedroom, Kitchen, Bathroom, Pencil, Pen, Photograph, Soap, Book, Page, Key, Paint, Letter, Paper, Lock, Telephone, Bag, Box, Gift, Card, Ring, Tool |
| **Pronouns** (8) | I, you, he, she, it, we, you (plural), they |
| **Seasons** (6) | Summer, Spring, Winter, Fall, Season, Monsoon |

---

## License

MIT

---

Built for the Deaf community and sign language researchers.
