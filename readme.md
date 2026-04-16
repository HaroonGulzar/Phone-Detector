# AI Phone Detection & Alert System

An advanced, Python-based desktop application that leverages Computer Vision (YOLOv8) to detect mobile phones in a live webcam feed. Designed to foster productive environments, the system triggers audio alerts and logs detection events upon spotting a mobile device.

It is particularly useful for **Study Sessions**, **No-Phone Zones**, and **Focus Time** settings where digital distractions must be minimized.

## 🚀 Features

- **Real-Time Object Detection**: Employs the YOLOv8 Nano model for lightning-fast and highly accurate mobile phone identification.
- **Multithreaded Video Streaming**: Utilizes a dedicated background thread for video capture, significantly optimizing Frames Per Second (FPS) and overall performance on standard hardware.
- **Interactive GUI**: Features a sleek, PyQt5-based graphical user interface allowing users to dynamically adjust algorithm confidence levels, cooldown periods, and customize the alert sound.
- **Intelligent Audio Alerts**: Incorporates an audio debounce mechanism to prevent stuttering when detection flickers, providing a seamless user experience.
- **Comprehensive Logging**: 
  - Captures and saves image snapshots locally upon trigger.
  - Logs all detection events, including timestamps and confidence metrics, into a **CSV file**.
  - Maintains a permanent, structured record in a local **SQLite database**.
- **Performance Optimized**: Built-in frame skipping efficiently reduces processing load without sacrificing capabilities.

## 🛠️ Prerequisites

To run the application, ensure the following are installed:
- Python 3.8 or higher
- A compatible webcam
- (Optional but recommended) CUDA-enabled GPU for hardware-accelerated inference.

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone <YOUR-REPOSITORY-URL>
   cd Phone-detector
   ```

2. **Set up a virtual environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   # On Windows use: venv\Scripts\activate
   # On MacOS/Linux use: source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Usage

Execute the application via the command line:

```bash
python app.py
```

Upon launching, you will be presented with the **Settings GUI**:
1. Select an audio file (e.g., `funny.mp3`) for the alert.
2. Adjust the detection **Confidence Level** slider according to your environment's lighting.
3. Set the **Cooldown Period** to specify the delay between subsequent detection logs.
4. Click **Start Detection** to initialize the webcam feed and the AI model.

Press `q` within the video window to safely exit the application and safely close database connections.

## 📁 System Logging & Data

The application automatically creates a `logs/` directory in the root folder containing:
- `images/`: Stores snapshot frames highlighting the detected devices.
- `detection_history.csv`: A straightforward, readable spreadsheet of past detections.
- `detections.db`: A SQLite database schema recording standard detection metrics (`timestamp`, `confidence`, `image_path`).
