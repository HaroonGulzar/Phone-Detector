import cv2
from ultralytics import YOLO
import pygame
import time
import os
import threading
import torch
import csv
import sqlite3
from datetime import datetime
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QFileDialog, QLineEdit
from PyQt5.QtCore import Qt

# --- CONFIGURATION ---
AUDIO_FILE = "audio.mp3"  
CONFIDENCE_LEVEL = 0.5    
COOLDOWN_SECONDS = 11     
FRAME_SKIP = 3
# ---------------------

class VideoStream:
    """Threaded Video Stream to improve FPS"""
    def __init__(self, src=0, width=1024, height=768):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        
    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self
        
    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()
                
    def read(self):
        if not self.grabbed:
            return None
        return self.frame.copy()
        
    def stop(self):
        self.stopped = True
        self.stream.release()

class PhoneDetector:
    def __init__(self, audio_file=AUDIO_FILE, conf_level=CONFIDENCE_LEVEL, cooldown=COOLDOWN_SECONDS, skip=FRAME_SKIP):
        self.audio_file = audio_file
        self.conf_level = conf_level
        self.cooldown = cooldown
        self.frame_skip = skip
        
        self._init_audio()
        self._init_logging()
        
        print("Loading Fast AI Model...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device.upper()}")
        self.model = YOLO('yolov8n.pt')
        
        self.last_alert_time = 0
        self.last_phone_time = 0
        self.frame_count = 0
        self.prev_time = time.time()
        self.fps = 0.0
        self.phone_detected = False
        self.boxes_to_draw = []
        
    def _init_audio(self):
        if not os.path.exists(self.audio_file):
            print(f"ERROR: '{self.audio_file}' file nahi mili!")
            self.has_audio = False
            return
            
        pygame.mixer.init()
        try:
            pygame.mixer.music.load(self.audio_file)
            self.has_audio = True
        except Exception as e:
            print(f"Audio Error: {e}")
            self.has_audio = False

    def _init_logging(self):
        self.logs_dir = "logs"
        self.images_dir = os.path.join(self.logs_dir, "images")
        self.csv_file = os.path.join(self.logs_dir, "detection_history.csv")
        self.db_file = os.path.join(self.logs_dir, "detections.db")
        
        os.makedirs(self.images_dir, exist_ok=True)
        
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Confidence", "Image Path"])
                
        # Initialize SQLite Database
        self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                confidence REAL NOT NULL,
                image_path TEXT NOT NULL
            )
        ''')
        self.conn.commit()

    def log_detection(self, frame, conf):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        readable_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        img_name = f"phone_{timestamp}.jpg"
        img_path = os.path.join(self.images_dir, img_name)
        
        cv2.imwrite(img_path, frame)
        
        # Log to CSV
        try:
            with open(self.csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([readable_time, f"{conf:.2f}", img_path])
        except Exception as e:
            print(f"Failed to log CSV: {e}")
            
        # Log to SQLite DB
        try:
            self.cursor.execute('''
                INSERT INTO detections (timestamp, confidence, image_path)
                VALUES (?, ?, ?)
            ''', (readable_time, float(conf), img_path))
            self.conn.commit()
        except Exception as e:
            print(f"Failed to log to DB: {e}")

    def trigger_alert(self, frame, conf):
        current_time = time.time()
        if current_time - self.last_alert_time > self.cooldown:
            print(f">>> ALERT: Phone Detected! ({conf:.2f})")
            
            self.log_detection(frame, conf)
            self.last_alert_time = current_time

    def process_frame(self, frame):
        self.frame_count += 1
        
        if self.frame_count % self.frame_skip == 0:
            results = self.model(frame, stream=True, verbose=False, conf=self.conf_level, imgsz=640, device=self.device)
            
            self.phone_detected = False
            self.boxes_to_draw = []
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item() if hasattr(box.cls[0], 'item') else box.cls[0])
                    class_name = self.model.names[cls_id]
                    
                    if class_name == 'cell phone':
                        self.phone_detected = True
                        self.last_phone_time = time.time()
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist() if hasattr(box.xyxy[0], 'tolist') else box.xyxy[0])
                        conf = float(box.conf[0].item() if hasattr(box.conf[0], 'item') else box.conf[0])
                        self.boxes_to_draw.append((x1, y1, x2, y2, conf))
                        
        return frame

    def draw_overlays(self, frame):
        # Frame rate calculation
        current_time = time.time()
        time_diff = current_time - self.prev_time
        if time_diff > 0:
            self.fps = 1 / time_diff
        self.prev_time = current_time
        
        cv2.putText(frame, f"FPS: {int(self.fps)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
        if self.phone_detected:
            try:
                for (x1, y1, x2, y2, conf) in self.boxes_to_draw:
                    color = (0, 0, 255) # Red for alert
                    thickness = 4
                    length = 30
                    
                    # Top-Left
                    cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
                    cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
                    # Top-Right
                    cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
                    cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
                    # Bottom-Left
                    cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
                    cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
                    # Bottom-Right
                    cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
                    cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)

                    # Semi-transparent red overlay
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

                    conf_text = f"PHONE ({int(conf * 100)}%)"
                    cv2.putText(frame, conf_text, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                
                # Find the max confidence for the logs
                if self.boxes_to_draw:
                    max_conf = max([box[4] for box in self.boxes_to_draw])
                    try:
                        self.trigger_alert(frame, max_conf)
                    except Exception as e:
                        import traceback
                        print(f"Error in trigger_alert: {e}")
                        traceback.print_exc()
            except Exception as e:
                import traceback
                print(f"Error in draw overlays: {e}")
                traceback.print_exc()

        # Audio logic with 1-second debounce (patience)
        # This fixes stuttering when detection flickers for a few frames
        if self.has_audio:
            if time.time() - self.last_phone_time < 1.0:
                # Phone was seen recently. Ensure music is playing.
                if not pygame.mixer.music.get_busy():
                    try:
                        pygame.mixer.music.play()
                    except:
                        pass
            else:
                # Phone has been gone for at least 1 full second. Stop music.
                if pygame.mixer.music.get_busy():
                    try:
                        pygame.mixer.music.stop()
                    except:
                        pass

    def run(self):
        print("System Ready. Detecting phones...")
        print("Press 'q' to exit.")
        cap = VideoStream(src=0).start()
        time.sleep(1.0) # wait for camera to warm up
        
        try:
            while True:
                frame = cap.read()
                if frame is None:
                    break
                    
                self.process_frame(frame)
                self.draw_overlays(frame)
                
                cv2.imshow('No Phone Zone', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.stop()
            cv2.destroyAllWindows()
            if hasattr(self, 'conn'):
                try:
                    self.conn.close()
                except:
                    pass


class SettingsApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Phone Detector Settings")
        self.resize(400, 300)
        
        self.audio_file = AUDIO_FILE
        self.conf_level = CONFIDENCE_LEVEL
        self.cooldown = COOLDOWN_SECONDS
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Audio File:"))
        self.audio_input = QLineEdit(self.audio_file)
        self.audio_input.setReadOnly(True)
        layout.addWidget(self.audio_input)
        
        btn_audio = QPushButton("Select Audio")
        btn_audio.clicked.connect(self.select_audio)
        btn_audio.setStyleSheet("padding: 5px; background-color: #eee;")
        layout.addWidget(btn_audio)
        
        self.lbl_conf = QLabel(f"Confidence Level: {self.conf_level}")
        layout.addWidget(self.lbl_conf)
        
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(10, 100)
        self.slider_conf.setValue(int(self.conf_level * 100))
        self.slider_conf.valueChanged.connect(self.update_conf)
        layout.addWidget(self.slider_conf)
        
        self.lbl_cooldown = QLabel(f"Cooldown Seconds: {self.cooldown}")
        layout.addWidget(self.lbl_cooldown)
        
        self.slider_cooldown = QSlider(Qt.Horizontal)
        self.slider_cooldown.setRange(1, 60)
        self.slider_cooldown.setValue(self.cooldown)
        self.slider_cooldown.valueChanged.connect(self.update_cooldown)
        layout.addWidget(self.slider_cooldown)
        
        btn_start = QPushButton("Start Detection")
        btn_start.clicked.connect(self.start_detection)
        btn_start.setStyleSheet("padding: 10px; font-weight: bold; background-color: #4CAF50; color: white;")
        layout.addWidget(btn_start)
        
        self.setLayout(layout)
        
    def select_audio(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.mp3 *.wav)")
        if filename:
            self.audio_file = filename
            self.audio_input.setText(self.audio_file)
            
    def update_conf(self, value):
        self.conf_level = value / 100.0
        self.lbl_conf.setText(f"Confidence Level: {self.conf_level}")
        
    def update_cooldown(self, value):
        self.cooldown = value
        self.lbl_cooldown.setText(f"Cooldown (Secs): {self.cooldown}")
        
    def start_detection(self):
        self.close()
        detector = PhoneDetector(audio_file=self.audio_file, conf_level=self.conf_level, cooldown=self.cooldown)
        detector.run()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = SettingsApp()
    ex.show()
    sys.exit(app.exec_())