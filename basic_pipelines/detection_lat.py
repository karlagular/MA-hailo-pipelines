from pathlib import Path
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import os
import atexit
import numpy as np
import cv2
import hailo
from collections import deque

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp


# -----------------------------------------------------------------------------------------------
# Log buffer with graceful Ctrl+C shutdown
# -----------------------------------------------------------------------------------------------
class LogBuffer:
    """Accumulates log messages and saves on process exit (atexit)."""
    def __init__(self, log_file="latency_log.txt"):
        self.log_file = log_file
        self.logs = []
        atexit.register(self._save)
    
    def log(self, msg: str):
        """Log message to buffer and print to console."""
        self.logs.append(msg)
        print(msg)
    
    def append_summary(self, msg: str):
        """Append summary text without printing to console."""
        self.logs.append(msg)
    
    def _save(self):
        """Save logs to file; create the file even if empty."""
        try:
            with open(self.log_file, 'w') as f:
                f.write('\n'.join(self.logs))
            print(f"\n[EXPORT] Logs saved to {self.log_file}")
        except Exception as e:
            print(f"\n[ERROR] Could not save logs: {e}")


# -----------------------------------------------------------------------------------------------
# Stage timing helpers (per-buffer timestamps keyed by PTS)
# -----------------------------------------------------------------------------------------------
REQUIRED_STAGES = [
    "src_out",          # source_fps_caps:src  (start of video branch)
    "wrapper_in",       # inference_wrapper_input_q:src
    "wrapper_out",      # inference_wrapper_output_q:src
    "tracker_out",      # hailo_tracker:src
    "cb_out",           # identity_callback:src (your callback point)
]

class StageTimer:
    """
    Stores per-frame timestamps at different stages of the pipeline.
    Key is buffer.pts (ns). Values are pipeline running-time (ns).
    """
    def __init__(self, max_inflight=800):
        self.t = {}          # pts_ns -> {stage_name: running_ns}
        self.order = deque()
        self.max_inflight = max_inflight

    def mark(self, pts_ns: int, stage: str, running_ns: int):
        if pts_ns == Gst.CLOCK_TIME_NONE:
            return
        if pts_ns not in self.t:
            self.t[pts_ns] = {}
            self.order.append(pts_ns)
            if len(self.order) > self.max_inflight:
                old = self.order.popleft()
                self.t.pop(old, None)
        self.t[pts_ns][stage] = running_ns

    def missing(self, pts_ns: int):
        rec = self.t.get(pts_ns)
        if not rec:
            return REQUIRED_STAGES
        return [s for s in REQUIRED_STAGES if s not in rec]

    def compute(self, pts_ns: int):
        rec = self.t.get(pts_ns)
        if not rec:
            return None
        if any(s not in rec for s in REQUIRED_STAGES):
            return None

        # All values in ms
        result = {
            "upstream_ms": (rec["src_out"] - pts_ns) / 1e6,
            "source_ms":  (rec["wrapper_in"]  - rec["src_out"])     / 1e6,
            "wrapper_ms": (rec["wrapper_out"] - rec["wrapper_in"])  / 1e6,
            "tracker_ms": (rec["tracker_out"] - rec["wrapper_out"]) / 1e6,
            "to_cb_ms":   (rec["cb_out"]      - rec["tracker_out"]) / 1e6,
            "e2e_ms":     (rec["cb_out"]      - rec["src_out"])     / 1e6,
        }
        # Full latency from PTS to cb_out
        result["pts_to_cb_ms"] = (rec["cb_out"] - pts_ns) / 1e6
        return result

    def cleanup(self, pts_ns: int):
        self.t.pop(pts_ns, None)


def running_time_ns(pipeline: Gst.Pipeline) -> int:
    """Pipeline running-time (ns) = clock_time - base_time."""
    return pipeline.get_clock().get_time() - pipeline.get_base_time()


# -----------------------------------------------------------------------------------------------
# Latency statistics tracker
# -----------------------------------------------------------------------------------------------
class LatencyStats:
    """Tracks min/max/sum of latencies for each stage."""
    def __init__(self):
        self.stages = [
            "upstream_ms", "source_ms", "wrapper_ms", "tracker_ms",
            "to_cb_ms", "e2e_ms", "pts_to_cb_ms"
        ]
        self.data = {stage: [] for stage in self.stages}
    
    def record(self, stage_dict):
        """Record a measurement from computed stage latencies."""
        if stage_dict is None:
            return
        for stage in self.stages:
            if stage in stage_dict:
                self.data[stage].append(stage_dict[stage])
    
    def summary_lines(self):
        """Generate summary lines with min/max/avg for each stage."""
        lines = ["\n" + "="*80]
        lines.append("LATENCY SUMMARY (min / max / average)")
        lines.append("="*80)
        for stage in self.stages:
            values = self.data[stage]
            if not values:
                lines.append(f"{stage:15} : no data")
            else:
                min_val = min(values)
                max_val = max(values)
                avg_val = sum(values) / len(values)
                lines.append(f"{stage:15} : {min_val:8.2f} / {max_val:8.2f} / {avg_val:8.2f} ms")
        lines.append("="*80)
        return lines


def make_stage_probe(user_data, stage_name: str):
    """Pad probe that stamps a running-time mark for this buffer at a given stage."""
    def _probe(pad, info):
        buf = info.get_buffer()
        if buf is None or user_data.pipeline is None:
            return Gst.PadProbeReturn.OK
        pts = buf.pts
        if pts == Gst.CLOCK_TIME_NONE:
            return Gst.PadProbeReturn.OK

        user_data.stage_timer.mark(pts, stage_name, running_time_ns(user_data.pipeline))
        return Gst.PadProbeReturn.OK
    return _probe


def attach_stage_probes(user_data):
    """
    Attach probes to the branch that actually feeds identity_callback.
    This avoids the PTS mismatch you saw with hailonet/hailofilter pads.
    """
    pipeline = user_data.pipeline

    def _pad(elem_name: str, pad_name: str):
        elem = pipeline.get_by_name(elem_name)
        if elem is None:
            print(f"[PROBES ERROR] Element '{elem_name}' not found.")
            return None
        pad = elem.get_static_pad(pad_name)
        if pad is None:
            print(f"[PROBES ERROR] Pad '{pad_name}' not found on '{elem_name}'.")
            return None
        return pad

    attached = 0

    # Start of the video branch after source conditioning
    p = _pad("source_fps_caps", "src")
    if p:
        p.add_probe(Gst.PadProbeType.BUFFER, make_stage_probe(user_data, "src_out"))
        attached += 1
        print("[PROBES OK] source_fps_caps:src -> src_out")

    # Enter wrapper (before crop/agg/inference merge)
    p = _pad("inference_wrapper_input_q", "src")
    if p:
        p.add_probe(Gst.PadProbeType.BUFFER, make_stage_probe(user_data, "wrapper_in"))
        attached += 1
        print("[PROBES OK] inference_wrapper_input_q:src -> wrapper_in")

    # Wrapper merged output (after inference/bypass aggregation)
    p = _pad("inference_wrapper_output_q", "src")
    if p:
        p.add_probe(Gst.PadProbeType.BUFFER, make_stage_probe(user_data, "wrapper_out"))
        attached += 1
        print("[PROBES OK] inference_wrapper_output_q:src -> wrapper_out")

    # Tracker output (optional but recommended; exists in your element list)
    p = _pad("hailo_tracker", "src")
    if p:
        p.add_probe(Gst.PadProbeType.BUFFER, make_stage_probe(user_data, "tracker_out"))
        attached += 1
        print("[PROBES OK] hailo_tracker:src -> tracker_out")
    else:
        # If tracker doesn't have a src pad, fall back to queue after tracker (present in your list)
        p = _pad("hailo_tracker_q", "src")
        if p:
            p.add_probe(Gst.PadProbeType.BUFFER, make_stage_probe(user_data, "tracker_out"))
            attached += 1
            print("[PROBES OK] hailo_tracker_q:src -> tracker_out")

    user_data.probes_attached = True
    print(f"[PROBES SUMMARY] attached={attached}/5\n")


# -----------------------------------------------------------------------------------------------
# User data container
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    """
    Shared state across callbacks.
    """
    def __init__(self, logger=None):
        super().__init__()
        self.pipeline = None
        self.logger = logger  # Reference to LogBuffer
        self.logger_announced = False

        # End-to-end latency stats (based on callback PTS vs running-time)
        self.n = 0
        self.last_ms = 0.0
        self.avg_ms = 0.0

        # Stage timing
        self.stage_timer = StageTimer()
        self.probes_attached = False
        self.missing_logs = 0
        
        # Latency statistics
        self.latency_stats = LatencyStats()

    def find_pipeline_from_pad(self, pad):
        elem = pad.get_parent_element()
        while elem is not None:
            if isinstance(elem, Gst.Pipeline):
                return elem
            elem = elem.get_parent()
        return None


# -----------------------------------------------------------------------------------------------
# Callback
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    buf = info.get_buffer()
    if buf is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()

    # Confirm callback attachment once
    if not hasattr(user_data, "printed_cb_owner"):
        owner = pad.get_parent_element()
        print("[CB] pad owner:", owner.get_name() if owner else None)
        print("[CB] pad name:", pad.get_name())
        user_data.printed_cb_owner = True

    # Announce logger status once for debug
    if not user_data.logger_announced:
        if user_data.logger:
            user_data.logger.log("[DEBUG] Logger is attached and active")
        else:
            print("[DEBUG] Logger is NOT attached; falling back to print")
        user_data.logger_announced = True

    # Resolve pipeline once
    if user_data.pipeline is None:
        user_data.pipeline = user_data.find_pipeline_from_pad(pad)
        if user_data.pipeline is not None:
            print("[INFO] Pipeline resolved:", user_data.pipeline.get_name())
        else:
            print("[WARN] Could not resolve pipeline yet")
            return Gst.PadProbeReturn.OK

    # Attach probes once
    if not user_data.probes_attached:
        attach_stage_probes(user_data)

    # ------------------------
    # Stage latency reporting
    # ------------------------
    pts_ns = buf.pts
    if pts_ns != Gst.CLOCK_TIME_NONE:
        # We are executing on an identity_callback:src, so this is the correct cb_out mark.
        cb_rt_ns = running_time_ns(user_data.pipeline)
        user_data.stage_timer.mark(pts_ns, "cb_out", cb_rt_ns)
        
        stage = user_data.stage_timer.compute(pts_ns)

        # Print stage latencies every 30 frames (based on app_callback frame counter)
        if user_data.get_count() % 30 == 0:
            if stage is not None:
                # Record statistics
                user_data.latency_stats.record(stage)
                
                msg = (
                    "[STAGES] "
                    f"upstream={stage['upstream_ms']:.1f} ms | "
                    f"source={stage['source_ms']:.1f} ms | "
                    f"wrapper={stage['wrapper_ms']:.1f} ms | "
                    f"tracker={stage['tracker_ms']:.1f} ms | "
                    f"to_cb={stage['to_cb_ms']:.1f} ms | "
                    f"e2e={stage['e2e_ms']:.1f} ms | "
                    f"pts_to_cb={stage['pts_to_cb_ms']:.1f} ms"
                )
                if user_data.logger:
                    user_data.logger.log(msg)
                else:
                    print(msg)
                user_data.stage_timer.cleanup(pts_ns)
            else:
                # Limited debug: show what is missing (helps confirm correlation is working)
                if user_data.missing_logs < 5:
                    msg = f"[STAGES] waiting for: {user_data.stage_timer.missing(pts_ns)}"
                    if user_data.logger:
                        user_data.logger.log(msg)
                    else:
                        print(msg)
                    user_data.missing_logs += 1

    # Optional: keep your detection parsing / frame code here

    return Gst.PadProbeReturn.OK

# -----------------------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    os.environ["HAILO_ENV_FILE"] = str(env_file)

    # Initialize logger (saves on Ctrl+C with zero runtime impact)
    logger = LogBuffer("latency_log.txt")
    logger.log("[INFO] Starting latency measurement...")
    
    user_data = user_app_callback_class(logger=logger)
    
    # Register cleanup function to append summary before exit
    def append_summary_on_exit():
        summary_lines = user_data.latency_stats.summary_lines()
        for line in summary_lines:
            logger.append_summary(line)
    
    atexit.register(append_summary_on_exit)
    
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
