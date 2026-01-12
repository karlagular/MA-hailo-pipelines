#!/usr/bin/env python3
"""
Experiment Configuration GUI
Provides a graphical interface for configuring experiment parameters,
launching detection pipeline, and monitoring for specific object detections.
"""

import sys
import json
import subprocess
import signal
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QCheckBox, QPushButton, QMessageBox, QGroupBox,
    QGridLayout
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont


# Configuration options for dropdowns
STREAM_OPTIONS = ["rpi camera cable", "wifi", "ethernet", "usb"]
CAMERA_OPTIONS = ["rpicam v2", "rpicam v3", "usb webcam", "other"]
COMPUTER_OPTIONS = ["rpi5+aihat", "rpi5", "rpi4", "laptop", "desktop"]
Z_AXIS_OPTIONS = ["Druckkopf", "Heizbett", "None"]
KAMERAWINKEL_OPTIONS = ["diagonal", "frontal", "seitlich", "top-down"]


class DetectionMonitorThread(QThread):
    """Thread to monitor detection script output for specific objects."""
    object_detected = pyqtSignal(str)  # Signal emitted when target object detected
    script_finished = pyqtSignal(int)  # Signal emitted when script exits
    
    def __init__(self, process):
        super().__init__()
        self.process = process
        self.target_objects = ["cup"]  # Objects to monitor
        self.running = True
    
    def run(self):
        """Monitor stdout/stderr for detection messages."""
        try:
            while self.running and self.process.poll() is None:
                output = self.process.stdout.readline()
                if output:
                    line = output.strip()
                    # Check if any target object is detected
                    for obj in self.target_objects:
                        if obj.lower() in line.lower() and "detection" in line.lower():
                            self.object_detected.emit(obj)
                            break
            
            # Get exit code
            exit_code = self.process.wait()
            self.script_finished.emit(exit_code)
        except Exception as e:
            print(f"[Monitor Error] {e}")
            self.script_finished.emit(-1)
    
    def stop(self):
        """Stop monitoring."""
        self.running = False


class ExperimentGUI(QMainWindow):
    """Main GUI window for experiment configuration."""
    
    def __init__(self):
        super().__init__()
        self.process = None
        self.monitor_thread = None
        self.config_path = Path(__file__).parent / "experiment_config.json"
        self.script_path = Path(__file__).parent / "basic_pipelines" / "detection_lat.py"
        self.alert_dialog = None  # Track active alert dialog
        self.stopping = False  # Flag to prevent alerts after stop is requested
        
        self.init_ui()
        self.load_current_config()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Experiment Configuration")
        self.setMinimumSize(500, 600)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("Experiment Parameter Configuration")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        main_layout.addWidget(title)
        
        # Parameters group
        params_group = QGroupBox("Experiment Parameters")
        params_layout = QGridLayout()
        
        # Dropdown fields
        row = 0
        
        # Stream
        params_layout.addWidget(QLabel("Stream:"), row, 0)
        self.stream_combo = QComboBox()
        self.stream_combo.addItems(STREAM_OPTIONS)
        params_layout.addWidget(self.stream_combo, row, 1)
        row += 1
        
        # Camera
        params_layout.addWidget(QLabel("Camera:"), row, 0)
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(CAMERA_OPTIONS)
        params_layout.addWidget(self.camera_combo, row, 1)
        row += 1
        
        # Computer
        params_layout.addWidget(QLabel("Computer:"), row, 0)
        self.computer_combo = QComboBox()
        self.computer_combo.addItems(COMPUTER_OPTIONS)
        params_layout.addWidget(self.computer_combo, row, 1)
        row += 1
        
        # Z-Axis
        params_layout.addWidget(QLabel("Z-Axis:"), row, 0)
        self.z_axis_combo = QComboBox()
        self.z_axis_combo.addItems(Z_AXIS_OPTIONS)
        params_layout.addWidget(self.z_axis_combo, row, 1)
        row += 1
        
        # Kamerawinkel
        params_layout.addWidget(QLabel("Kamerawinkel:"), row, 0)
        self.kamerawinkel_combo = QComboBox()
        self.kamerawinkel_combo.addItems(KAMERAWINKEL_OPTIONS)
        params_layout.addWidget(self.kamerawinkel_combo, row, 1)
        row += 1
        
        # Checkbox fields
        params_layout.addWidget(QLabel("Beleuchtung:"), row, 0)
        self.beleuchtung_check = QCheckBox()
        params_layout.addWidget(self.beleuchtung_check, row, 1)
        row += 1
        
        params_layout.addWidget(QLabel("Enclosure:"), row, 0)
        self.enclosure_check = QCheckBox()
        params_layout.addWidget(self.enclosure_check, row, 1)
        row += 1
        
        params_layout.addWidget(QLabel("Vibration:"), row, 0)
        self.vibration_check = QCheckBox()
        params_layout.addWidget(self.vibration_check, row, 1)
        
        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)
        
        # Status label
        self.status_label = QLabel("Ready to configure experiment")
        self.status_label.setStyleSheet("color: blue; font-weight: bold;")
        main_layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_fields)
        button_layout.addWidget(self.reset_button)
        
        self.continue_button = QPushButton("Continue")
        self.continue_button.clicked.connect(self.continue_clicked)
        self.continue_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        button_layout.addWidget(self.continue_button)
        
        self.stop_button = QPushButton("Stop Inference")
        self.stop_button.clicked.connect(self.stop_inference)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("background-color: #f44336; color: white;")
        button_layout.addWidget(self.stop_button)
        
        main_layout.addLayout(button_layout)
        
        # Add stretch to push everything to top
        main_layout.addStretch()
    
    def load_current_config(self):
        """Load current configuration from JSON file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                # Set dropdown values
                if "stream" in config:
                    idx = self.stream_combo.findText(config["stream"])
                    if idx >= 0:
                        self.stream_combo.setCurrentIndex(idx)
                
                if "camera" in config:
                    idx = self.camera_combo.findText(config["camera"])
                    if idx >= 0:
                        self.camera_combo.setCurrentIndex(idx)
                
                if "computer" in config:
                    idx = self.computer_combo.findText(config["computer"])
                    if idx >= 0:
                        self.computer_combo.setCurrentIndex(idx)
                
                if "z-axis" in config:
                    idx = self.z_axis_combo.findText(config["z-axis"])
                    if idx >= 0:
                        self.z_axis_combo.setCurrentIndex(idx)
                
                if "kamerawinkel" in config:
                    idx = self.kamerawinkel_combo.findText(config["kamerawinkel"])
                    if idx >= 0:
                        self.kamerawinkel_combo.setCurrentIndex(idx)
                
                # Set checkbox values
                self.beleuchtung_check.setChecked(bool(config.get("beleuchtung", 0)))
                self.enclosure_check.setChecked(bool(config.get("enclosure", 0)))
                self.vibration_check.setChecked(bool(config.get("vibration", 0)))
                
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Could not load config: {e}")
    
    def reset_fields(self):
        """Reset all input fields to default values."""
        self.stream_combo.setCurrentIndex(0)
        self.camera_combo.setCurrentIndex(0)
        self.computer_combo.setCurrentIndex(0)
        self.z_axis_combo.setCurrentIndex(0)
        self.kamerawinkel_combo.setCurrentIndex(0)
        self.beleuchtung_check.setChecked(False)
        self.enclosure_check.setChecked(False)
        self.vibration_check.setChecked(False)
        self.status_label.setText("Fields reset")
        self.status_label.setStyleSheet("color: blue; font-weight: bold;")
    
    def continue_clicked(self):
        """Save configuration and start inference."""
        # Gather configuration
        config = {
            "stream": self.stream_combo.currentText(),
            "camera": self.camera_combo.currentText(),
            "computer": self.computer_combo.currentText(),
            "beleuchtung": 1 if self.beleuchtung_check.isChecked() else 0,
            "enclosure": 1 if self.enclosure_check.isChecked() else 0,
            "vibration": 1 if self.vibration_check.isChecked() else 0,
            "z-axis": self.z_axis_combo.currentText(),
            "kamerawinkel": self.kamerawinkel_combo.currentText()
        }
        
        # Save to JSON
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.status_label.setText("Configuration saved. Starting inference...")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            
            # Freeze input fields
            self.freeze_inputs(True)
            
            # Start inference after short delay
            QTimer.singleShot(500, self.start_inference)
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save configuration: {e}")
    
    def freeze_inputs(self, frozen):
        """Enable or disable input fields."""
        self.stream_combo.setEnabled(not frozen)
        self.camera_combo.setEnabled(not frozen)
        self.computer_combo.setEnabled(not frozen)
        self.z_axis_combo.setEnabled(not frozen)
        self.kamerawinkel_combo.setEnabled(not frozen)
        self.beleuchtung_check.setEnabled(not frozen)
        self.enclosure_check.setEnabled(not frozen)
        self.vibration_check.setEnabled(not frozen)
        self.reset_button.setEnabled(not frozen)
        self.continue_button.setEnabled(not frozen)
        self.stop_button.setEnabled(frozen)
    
    def start_inference(self):
        """Launch the detection script."""
        try:
            # Reset stopping flag for new inference session
            self.stopping = False
            
            # Start detection script as subprocess
            cmd = [
                sys.executable,
                str(self.script_path),
                "--input", "rpi",
                "--save-logs"
            ]
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Start monitoring thread
            self.monitor_thread = DetectionMonitorThread(self.process)
            self.monitor_thread.object_detected.connect(self.on_object_detected)
            self.monitor_thread.script_finished.connect(self.on_script_finished)
            self.monitor_thread.start()
            
            self.status_label.setText("Inference running... Monitoring for objects")
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")
            
        except Exception as e:
            QMessageBox.critical(self, "Launch Error", f"Could not start inference: {e}")
            self.freeze_inputs(False)
    
    def stop_inference(self):
        """Stop the running inference process."""
        if self.process and self.process.poll() is None:
            reply = QMessageBox.question(
                self,
                "Confirm Stop",
                "Are you sure you want to stop the inference?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                try:
                    # Set flag to prevent new alerts after having stopped the program
                    self.stopping = True
                    
                    # Send SIGINT for graceful shutdown
                    self.process.send_signal(signal.SIGINT)
                    
                    # Wait a bit, then force kill if needed
                    QTimer.singleShot(2000, self.force_kill_if_needed)
                    
                    self.status_label.setText("Stopping inference...")
                    self.status_label.setStyleSheet("color: red; font-weight: bold;")
                    
                except Exception as e:
                    QMessageBox.warning(self, "Stop Error", f"Error stopping process: {e}")
    
    def force_kill_if_needed(self):
        """Force kill process if it hasn't stopped gracefully."""
        if self.process and self.process.poll() is None:
            self.process.kill()
    
    def on_object_detected(self, object_name):
        """Handle object detection alert."""
        # Don't show alerts if stopping or if an alert is already open
        if self.stopping:
            return  # Stop was requested, ignore detections
        
        if self.alert_dialog is not None and self.alert_dialog.isVisible():
            return  # Alert already open, ignore this detection
        
        alert = QMessageBox(self)
        alert.setIcon(QMessageBox.Warning)
        alert.setWindowTitle("Object Detected!")
        alert.setText(f"Alert: {object_name.upper()} detected in frame!")
        alert.setInformativeText("This may indicate an error condition.")
        
        stop_btn = alert.addButton("Stop Program", QMessageBox.ActionRole)
        continue_btn = alert.addButton("Continue", QMessageBox.RejectRole)
        
        self.alert_dialog = alert  # Track the active alert
        alert.exec_()
        self.alert_dialog = None  # Clear when closed
        
        if alert.clickedButton() == stop_btn:
            # Send error signal (could write to file or socket)
            self.send_error_signal(object_name)
            self.stop_inference()
    
    def send_error_signal(self, object_name):
        """Send error signal (e.g., write to file)."""
        error_file = Path(__file__).parent / "error_signal.txt"
        try:
            with open(error_file, 'w') as f:
                f.write(f"ERROR: {object_name} detected\n")
            print(f"[ERROR SIGNAL] {object_name} detected - signal written to {error_file}")
        except Exception as e:
            print(f"[ERROR] Could not write error signal: {e}")
    
    def on_script_finished(self, exit_code):
        """Handle inference script completion."""
        if self.monitor_thread:
            self.monitor_thread.stop()
            self.monitor_thread.wait()
        
        self.freeze_inputs(False)
        
        if exit_code == 0:
            self.status_label.setText("Inference completed successfully")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.status_label.setText(f"Inference stopped (exit code: {exit_code})")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
        
        self.process = None
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.process and self.process.poll() is None:
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Inference is still running. Stop it and exit?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                if self.monitor_thread:
                    self.monitor_thread.stop()
                self.process.terminate()
                self.process.wait(timeout=3)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern looking style
    
    window = ExperimentGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
