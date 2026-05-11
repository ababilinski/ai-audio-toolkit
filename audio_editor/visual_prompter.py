"""Visual prompting for SAM-Audio using SAM2 or SAM3 video segmentation."""
import json
import logging
import os
import subprocess
import shutil
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPoint, QUrl
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QMessageBox, QProgressBar, QSlider, QWidget,
    QStackedWidget, QGroupBox,
)

from .runtime import ffmpeg_command, get_ffmpeg_executable

log = logging.getLogger(__name__)

# ── Availability Detection ──

_checked = False
VISION_BACKEND = None  # "sam2" or "sam3" or None
VISION_IMPORT_ERROR = ""


def check_vision_available() -> str | None:
    """Check which vision segmentation backend is available. Returns 'sam2', 'sam3', or None."""
    global _checked, VISION_BACKEND, VISION_IMPORT_ERROR
    if _checked:
        return VISION_BACKEND
    _checked = True

    # Check SAM3 first (preferred — supports text + visual, no approval via jetjodh/sam3)
    try:
        from sam3.build_sam import build_sam3_video_predictor  # noqa: F401
        VISION_BACKEND = "sam3"
        log.info("Vision backend: SAM3 available (native)")
        return VISION_BACKEND
    except ImportError:
        pass

    # Fallback: SAM3 via transformers (tracker or base model)
    try:
        from transformers import Sam3TrackerVideoModel  # noqa: F401
        VISION_BACKEND = "sam3"
        log.info("Vision backend: SAM3 available (transformers - tracker)")
        return VISION_BACKEND
    except (ImportError, Exception):
        pass

    try:
        from transformers import Sam3VideoModel  # noqa: F401
        VISION_BACKEND = "sam3"
        log.info("Vision backend: SAM3 available (transformers - base, point prompt support will be checked at runtime)")
        return VISION_BACKEND
    except (ImportError, Exception):
        pass

    # Check SAM2
    try:
        from sam2.sam2_video_predictor import SAM2VideoPredictor  # noqa: F401
        VISION_BACKEND = "sam2"
        log.info("Vision backend: SAM2 available")
        return VISION_BACKEND
    except ImportError as e:
        VISION_IMPORT_ERROR = str(e)

    log.info("No vision segmentation backend available (SAM2 or SAM3)")
    return None


# ── Mask Project Persistence ──

MASK_CONFIG_NAME = "mask_config.json"
MASK_VIDEO_NAME = "masked_video.mp4"


def save_mask_project(project_dir: str, video_path: str, click_point: tuple[int, int],
                      frame_index: int, video_scale: int, backend: str,
                      masked_video_path: str,
                      click_points: list[tuple[int, int]] | None = None) -> str:
    """Save mask project: copy masked video + write config to project_dir.

    Returns the path to the copied masked video inside the project folder.
    """
    os.makedirs(project_dir, exist_ok=True)
    dst_video = os.path.join(project_dir, MASK_VIDEO_NAME)
    if os.path.abspath(masked_video_path) != os.path.abspath(dst_video):
        shutil.copy2(masked_video_path, dst_video)
    all_points = click_points or [click_point]
    config = {
        "source_video": video_path,
        "click_point": list(click_point),
        "click_points": [list(pt) for pt in all_points],
        "num_objects": len(all_points),
        "frame_index": frame_index,
        "video_scale": video_scale,
        "backend": backend,
    }
    with open(os.path.join(project_dir, MASK_CONFIG_NAME), "w") as f:
        json.dump(config, f, indent=2)
    log.info("Mask project saved to %s (%d objects)", project_dir, len(all_points))
    return dst_video


def load_mask_project(project_dir: str) -> tuple[str, dict] | None:
    """Load a mask project folder. Returns (masked_video_path, config) or None."""
    config_path = os.path.join(project_dir, MASK_CONFIG_NAME)
    video_path = os.path.join(project_dir, MASK_VIDEO_NAME)
    if not os.path.isfile(config_path) or not os.path.isfile(video_path):
        return None
    try:
        with open(config_path) as f:
            config = json.load(f)
        return video_path, config
    except (json.JSONDecodeError, OSError):
        return None


def validate_mask_folder(folder_path: str) -> str | None:
    """Validate a mask project folder. Returns error message or None if valid."""
    if not os.path.isdir(folder_path):
        return "Not a directory."
    config_path = os.path.join(folder_path, MASK_CONFIG_NAME)
    video_path = os.path.join(folder_path, MASK_VIDEO_NAME)
    if not os.path.isfile(config_path):
        return f"Missing {MASK_CONFIG_NAME} — not a valid mask project folder."
    if not os.path.isfile(video_path):
        return f"Missing {MASK_VIDEO_NAME} — the masked video file is missing."
    try:
        with open(config_path) as f:
            config = json.load(f)
        if "source_video" not in config or "click_point" not in config:
            return f"Invalid {MASK_CONFIG_NAME} — missing required fields."
    except (json.JSONDecodeError, OSError) as e:
        return f"Cannot read {MASK_CONFIG_NAME}: {e}"
    return None


# ── Mask Generation Workers ──

class MaskGenerationWorker(QThread):
    """Generate object segmentation masks from video using SAM2 or SAM3."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)  # path to masked video
    error = pyqtSignal(str)

    def __init__(self, video_path: str, click_point: tuple[int, int],
                 frame_size: tuple[int, int], output_dir: str,
                 backend: str = "sam2", model_id: str | None = None,
                 frame_index: int = 0, video_scale: int = 1,
                 click_points: list[tuple[int, int]] | None = None,
                 parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.click_point = click_point  # (x, y) — first/primary point (legacy compat)
        self.click_points = click_points or [click_point]  # all points, one per object
        self.frame_size = frame_size  # (width, height) of original video
        self.output_dir = output_dir
        self.backend = backend
        self.model_id = model_id
        self.frame_index = frame_index
        self.video_scale = video_scale  # 1=full, 2=half, 3=third, 4=quarter

    def run(self):
        try:
            if self.backend == "sam2":
                self._run_sam2()
            elif self.backend == "sam3":
                self._run_sam3()
            else:
                self.error.emit(f"Unknown backend: {self.backend}")
        except Exception as e:
            import traceback
            log.error(traceback.format_exc())
            # Ensure GPU is freed even on error
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
            except Exception:
                pass
            self.error.emit(str(e))

    def _prepare_scaled_video(self) -> str:
        """If video_scale > 1, create a downscaled copy. Returns path to use."""
        if self.video_scale <= 1:
            return self.video_path
        s = self.video_scale
        # trunc(.../2)*2 ensures even dimensions (required by libx264)
        vf = f"scale=trunc(iw/{s}/2)*2:trunc(ih/{s}/2)*2"
        scaled_path = str(Path(self.output_dir) / f"_scaled_{s}x.mp4")
        log.info("Scaling video by 1/%d for mask generation...", s)
        self.progress.emit(3, f"Scaling video to 1/{s} resolution...")
        subprocess.run(
            ffmpeg_command(
                "-y", "-i", self.video_path,
                "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-an", scaled_path,
            ),
            capture_output=True, check=True,
        )
        log.info("Scaled video created: %s", scaled_path)
        return scaled_path

    def _scaled_click_point(self) -> tuple[int, int]:
        """Return first click point adjusted for video scale."""
        x, y = self.click_point
        return (x // self.video_scale, y // self.video_scale)

    def _scaled_click_points(self) -> list[tuple[int, int]]:
        """Return all click points adjusted for video scale."""
        s = self.video_scale
        return [(x // s, y // s) for x, y in self.click_points]

    def _cleanup_scaled_video(self):
        """Remove the scaled video if we created one."""
        if self.video_scale > 1:
            scaled_path = str(Path(self.output_dir) / f"_scaled_{self.video_scale}x.mp4")
            if os.path.exists(scaled_path):
                os.remove(scaled_path)

    def _run_sam2(self):
        """Generate masks using SAM2 video predictor."""
        import torch
        from sam2.sam2_video_predictor import SAM2VideoPredictor

        model_id = self.model_id or "facebook/sam2.1-hiera-large"
        log.info("SAM2: Loading model %s...", model_id)
        self.progress.emit(5, f"Downloading/loading SAM2 model ({model_id})...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("SAM2: Using device %s", device)
        self.progress.emit(10, f"Loading SAM2 weights to {device}...")
        predictor = SAM2VideoPredictor.from_pretrained(model_id).to(device)
        log.info("SAM2: Model loaded successfully")

        self.progress.emit(30, "Initializing video...")

        # SAM2 needs mp4 or JPEG directory
        video_path = self.video_path
        if not video_path.lower().endswith(".mp4"):
            # Convert to mp4 if needed
            tmp_mp4 = str(Path(self.output_dir) / "_sam2_temp.mp4")
            subprocess.run(
                ffmpeg_command(
                    "-y", "-i", video_path, "-c:v", "libx264",
                    "-preset", "fast", tmp_mp4,
                ),
                capture_output=True, check=True,
            )
            video_path = tmp_mp4

        with torch.inference_mode():
            state = predictor.init_state(video_path=video_path)

            self.progress.emit(40, f"Adding {len(self.click_points)} object marker(s)...")

            for obj_idx, pt in enumerate(self.click_points):
                points = np.array([list(pt)], dtype=np.float32)
                labels = np.array([1], dtype=np.int32)  # 1 = foreground
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=self.frame_index,
                    obj_id=obj_idx + 1,
                    points=points,
                    labels=labels,
                )
            log.info("SAM2: Added %d object markers", len(self.click_points))

            self.progress.emit(50, "Propagating masks across video...")

            video_segments = {}
            for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
                # masks: (num_obj, 1, H, W) logits — OR all objects together
                combined = (masks[:, 0] > 0.0).any(dim=0).cpu().numpy()  # (H, W) bool
                video_segments[frame_idx] = combined

        self.progress.emit(80, "Creating masked video...")

        # Free GPU memory before creating masked video
        del predictor, state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        log.info("SAM2: GPU memory freed")

        masked_path = self._create_masked_video(video_segments)

        # Cleanup temp
        tmp_mp4 = str(Path(self.output_dir) / "_sam2_temp.mp4")
        if os.path.exists(tmp_mp4):
            os.remove(tmp_mp4)

        self.progress.emit(100, "Mask generation complete!")
        self.finished.emit(masked_path)

    def _run_sam3(self):
        """Generate masks using SAM3 (native API or transformers fallback)."""
        import torch

        model_id = self.model_id or "jetjodh/sam3"
        log.info("SAM3: Starting with model %s", model_id)
        self.progress.emit(5, f"Downloading/loading SAM3 model ({model_id})...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("SAM3: Using device %s", device)

        # Try native sam3 package first, fall back to transformers
        try:
            from sam3.build_sam import build_sam3_video_predictor
            log.info("SAM3: Loading via native package...")
            self.progress.emit(10, f"Loading SAM3 weights to {device} (native)...")
            predictor = build_sam3_video_predictor(model_id=model_id, device=device)
            log.info("SAM3: Native model loaded successfully")
            self._run_sam3_native(predictor, device)
            return
        except ImportError:
            log.info("SAM3: Native package not found, trying transformers...")
            pass

        # Transformers fallback
        self._run_sam3_transformers(model_id, device)

    def _run_sam3_native(self, predictor, device):
        """SAM3 using native sam3 package with handle_request() API."""
        import torch

        self.progress.emit(20, "Preparing video for SAM3...")

        video_path = self.video_path
        if not video_path.lower().endswith(".mp4"):
            tmp_mp4 = str(Path(self.output_dir) / "_sam3_temp.mp4")
            subprocess.run(
                ffmpeg_command(
                    "-y", "-i", video_path, "-c:v", "libx264",
                    "-preset", "fast", tmp_mp4,
                ),
                capture_output=True, check=True,
            )
            video_path = tmp_mp4

        self.progress.emit(30, "Starting SAM3 video session...")
        log.info("SAM3: Starting video session for %s", video_path)

        with torch.inference_mode():
            # SAM3 native API uses handle_request() pattern
            response = predictor.handle_request(request=dict(
                type="start_session",
                resource_path=video_path,
            ))
            session_id = response["session_id"]
            log.info("SAM3: Session started (id=%s)", session_id)

            self.progress.emit(40, f"Adding {len(self.click_points)} object marker(s)...")

            for obj_idx, pt in enumerate(self.click_points):
                response = predictor.handle_request(request=dict(
                    type="add_new_points_or_box",
                    session_id=session_id,
                    frame_index=self.frame_index,
                    points=[list(pt)],
                    labels=[1],  # 1 = foreground
                    obj_id=obj_idx + 1,
                ))
            log.info("SAM3: Added %d object markers on frame %d",
                     len(self.click_points), self.frame_index)

            self.progress.emit(50, "Propagating masks across video...")
            log.info("SAM3: Propagating masks...")

            # Propagate masks across all frames
            response = predictor.handle_request(request=dict(
                type="propagate_in_video",
                session_id=session_id,
            ))

            video_segments = {}
            # Response contains masks for all frames — OR all objects together
            if "results" in response:
                for result in response["results"]:
                    frame_idx = result["frame_index"]
                    masks = result["masks"]
                    if hasattr(masks, 'cpu'):
                        # masks: (num_obj, H, W) or (num_obj, 1, H, W)
                        binary_mask = (masks > 0.0).any(dim=0).cpu().numpy()
                    else:
                        binary_mask = (np.array(masks) > 0.0).any(axis=0)
                    video_segments[frame_idx] = np.squeeze(binary_mask)
            else:
                # Some versions return an iterator-style response
                for frame_idx, obj_ids, masks in response.get("segments", []):
                    binary_mask = (masks[:, 0] > 0.0).any(dim=0).cpu().numpy()
                    video_segments[frame_idx] = binary_mask

            log.info("SAM3: Got masks for %d frames", len(video_segments))

        # Free GPU memory before creating masked video
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        log.info("SAM3 native: GPU memory freed")

        self.progress.emit(80, "Creating masked video...")
        masked_path = self._create_masked_video(video_segments)

        tmp_mp4 = str(Path(self.output_dir) / "_sam3_temp.mp4")
        if os.path.exists(tmp_mp4):
            os.remove(tmp_mp4)

        self.progress.emit(100, "Mask generation complete!")
        self.finished.emit(masked_path)

    def _run_sam3_transformers(self, model_id, device):
        """SAM3 using HuggingFace transformers API.

        Uses Sam3TrackerVideoModel + Sam3TrackerVideoProcessor for point prompts.
        The documented public API for adding clicks is:
            processor.add_inputs_to_inference_session(
                inference_session, frame_idx, obj_ids,
                input_points=[[[[x, y]]]], input_labels=[[[1]]]
            )
        Points are 4D: [batch[obj[points[x,y]]]], labels are 3D: [batch[obj[labels]]].
        """
        import torch

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Sam3TrackerVideoModel is required for point/click prompts
        try:
            from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
        except ImportError:
            self.error.emit(
                "Sam3TrackerVideoModel not found in transformers.\n"
                "Update transformers: pip install -U transformers\n"
                "Or install the native sam3 package:\n"
                "  git clone https://github.com/facebookresearch/sam3\n"
                "  cd sam3 && pip install -e ."
            )
            return

        log.info("Loading SAM3 tracker model (%s)...", model_id)
        self.progress.emit(10, f"Loading SAM3 model ({model_id})...")

        model = Sam3TrackerVideoModel.from_pretrained(model_id).to(device, dtype=dtype)
        log.info("SAM3 tracker model loaded on %s (%s)", device, dtype)

        self.progress.emit(20, "Loading SAM3 processor...")
        processor = Sam3TrackerVideoProcessor.from_pretrained(model_id)
        log.info("SAM3 tracker processor loaded")

        self.progress.emit(30, "Loading video frames...")
        from transformers.video_utils import load_video
        # Use scaled video if user selected a lower resolution
        video_for_model = self._prepare_scaled_video()
        video_frames, _ = load_video(video_for_model)
        log.info("Loaded %d video frames (scale=1/%d)", len(video_frames), self.video_scale)

        self.progress.emit(40, "Initializing video session...")
        # Keep video frames on CPU to avoid OOM — only inference runs on GPU
        session = processor.init_video_session(
            video=video_frames,
            inference_device=device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=dtype,
        )
        log.info("SAM3 video session initialized (%dx%d)",
                 session.video_height, session.video_width)

        scaled_pts = self._scaled_click_points()
        self.progress.emit(50, f"Adding {len(scaled_pts)} object marker(s)...")

        for obj_idx, (x, y) in enumerate(scaled_pts):
            # Points must be 4D: [batch[obj[points[x, y]]]]
            # Labels must be 3D: [batch[obj[labels]]]
            processor.add_inputs_to_inference_session(
                inference_session=session,
                frame_idx=self.frame_index,
                obj_ids=obj_idx + 1,
                input_points=[[[[x, y]]]],
                input_labels=[[[1]]],  # 1 = foreground
            )
            log.info("Point prompt added at (%d, %d) obj_id=%d on frame %d",
                     x, y, obj_idx + 1, self.frame_index)

        # Must run inference on the annotated frame before propagating
        self.progress.emit(55, "Running inference on annotated frame...")
        ann_output = model(inference_session=session, frame_idx=self.frame_index)
        log.info("Annotated frame inference complete")

        self.progress.emit(60, "Propagating masks across video...")
        video_segments = {}
        total_frames = len(video_frames)
        for output in model.propagate_in_video_iterator(inference_session=session):
            try:
                masks = processor.post_process_masks(
                    [output.pred_masks],
                    original_sizes=[[session.video_height, session.video_width]],
                    binarize=True,
                )[0]
                # masks: (num_obj, H, W) or (num_obj, 1, H, W) — OR all objects
                combined = (masks > 0).any(dim=0).cpu().numpy()
                mask_np = np.squeeze(combined)
            except Exception:
                # Fallback: use pred_masks directly without post-processing
                mask_np = (output.pred_masks > 0).any(dim=0).cpu().float().numpy()
                mask_np = np.squeeze(mask_np)
            video_segments[output.frame_idx] = mask_np > 0
            # Update progress periodically
            if total_frames > 0 and output.frame_idx % 50 == 0:
                pct = 60 + int(20 * output.frame_idx / total_frames)
                self.progress.emit(min(pct, 79),
                    f"Propagating masks... frame {output.frame_idx}/{total_frames}")

        log.info("SAM3: got masks for %d frames", len(video_segments))
        self.progress.emit(80, "Creating masked video...")

        del model, processor, session
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

        masked_path = self._create_masked_video(video_segments)
        self._cleanup_scaled_video()
        self.progress.emit(100, "Mask generation complete!")
        self.finished.emit(masked_path)

    def _create_masked_video(self, video_segments: dict[int, np.ndarray]) -> str:
        """Create a masked video from binary masks using ffmpeg.

        Extracts frames, applies masks (black out non-object regions),
        re-encodes to video. Uses scaled video if video_scale > 1.
        """
        import cv2

        masked_dir = Path(self.output_dir) / "_masked_frames"
        masked_dir.mkdir(parents=True, exist_ok=True)

        # Extract frames — use scaled video if one was created
        frames_dir = Path(self.output_dir) / "_orig_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Use the scaled video as source so masks align with frame resolution
        source_video = self.video_path
        if self.video_scale > 1:
            scaled_path = str(Path(self.output_dir) / f"_scaled_{self.video_scale}x.mp4")
            if os.path.exists(scaled_path):
                source_video = scaled_path

        subprocess.run(
            ffmpeg_command(
                "-y", "-i", source_video,
                str(frames_dir / "frame_%06d.png"),
            ),
            capture_output=True, check=True,
        )

        frame_files = sorted(frames_dir.glob("frame_*.png"))

        for i, frame_file in enumerate(frame_files):
            frame = cv2.imread(str(frame_file))
            if frame is None:
                continue

            if i in video_segments:
                mask = video_segments[i]
                # Resize mask to frame size if needed
                if mask.shape[:2] != frame.shape[:2]:
                    mask = cv2.resize(
                        mask.astype(np.uint8), (frame.shape[1], frame.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                # Black out regions outside the mask
                frame[~mask] = 0

            cv2.imwrite(str(masked_dir / f"frame_{i:06d}.png"), frame)

        # Re-encode to video
        output_path = str(Path(self.output_dir) / "masked_video.mp4")
        subprocess.run(
            ffmpeg_command(
                "-y", "-framerate", "30",
                "-i", str(masked_dir / "frame_%06d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                output_path,
            ),
            capture_output=True, check=True,
        )

        # Cleanup temp directories
        shutil.rmtree(frames_dir, ignore_errors=True)
        shutil.rmtree(masked_dir, ignore_errors=True)

        return output_path


# ── Click-to-Segment Frame Widget ──

_MARKER_COLORS = [
    "#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8", "#cba6f7",
    "#94e2d5", "#fab387", "#eba0ac", "#74c7ec", "#b4befe",
]


class _ClickableFrame(QLabel):
    """QLabel that shows a video frame and records click positions for multiple objects."""
    clicked = pyqtSignal(int, int)  # x, y in original video coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._original_size: tuple[int, int] = (0, 0)  # (w, h)
        self._click_points: list[QPoint] = []  # all object markers
        self._mask_overlay: QImage | None = None
        self._clicks_locked = False
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #181825; border: 1px solid #333344;")

    def set_clicks_locked(self, locked: bool):
        """Lock/unlock click interaction (lock during processing)."""
        self._clicks_locked = locked
        self.setCursor(
            Qt.CursorShape.WaitCursor if locked else Qt.CursorShape.CrossCursor)

    def set_frame(self, image: QImage, original_w: int, original_h: int):
        self._original_size = (original_w, original_h)
        self._pixmap = QPixmap.fromImage(image)
        self._click_points.clear()
        self._mask_overlay = None
        self._update_display()

    def set_mask_overlay(self, mask_image: QImage):
        self._mask_overlay = mask_image
        self._update_display()

    def add_marker(self, x: int, y: int):
        """Add a marker at original video coordinates."""
        self._click_points.append(QPoint(x, y))
        self._update_display()

    def remove_last_marker(self):
        """Remove the most recently added marker."""
        if self._click_points:
            self._click_points.pop()
            self._update_display()

    def clear_markers(self):
        """Remove all markers."""
        self._click_points.clear()
        self._update_display()

    @property
    def markers(self) -> list[tuple[int, int]]:
        return [(p.x(), p.y()) for p in self._click_points]

    def _update_display(self):
        if self._pixmap is None:
            return
        display = self._pixmap.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)

        painter = QPainter(display)
        if self._mask_overlay:
            overlay_scaled = QPixmap.fromImage(self._mask_overlay).scaled(
                display.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            painter.setOpacity(0.4)
            painter.drawPixmap(0, 0, overlay_scaled)
            painter.setOpacity(1.0)

        # Draw all markers with distinct colors
        scale_x = display.width() / self._original_size[0] if self._original_size[0] else 1
        scale_y = display.height() / self._original_size[1] if self._original_size[1] else 1

        for i, pt in enumerate(self._click_points):
            color = _MARKER_COLORS[i % len(_MARKER_COLORS)]
            dx = int(pt.x() * scale_x)
            dy = int(pt.y() * scale_y)

            painter.setPen(QPen(QColor(color), 2))
            painter.setBrush(QColor(color))
            painter.setOpacity(0.3)
            painter.drawEllipse(QPoint(dx, dy), 12, 12)
            painter.setOpacity(1.0)
            painter.setPen(QPen(QColor(color), 2))
            painter.drawEllipse(QPoint(dx, dy), 12, 12)
            painter.setPen(QPen(QColor("#cdd6f4"), 2))
            painter.drawLine(dx - 6, dy, dx + 6, dy)
            painter.drawLine(dx, dy - 6, dx, dy + 6)
            # Object number label
            painter.setPen(QPen(QColor(color), 1))
            painter.drawText(dx + 14, dy - 6, str(i + 1))

        painter.end()
        super().setPixmap(display)

    def mousePressEvent(self, event):
        if (self._pixmap is None or self._clicks_locked
                or event.button() != Qt.MouseButton.LeftButton):
            return
        pm = self.pixmap()
        if pm is None:
            return
        ox = (self.width() - pm.width()) // 2
        oy = (self.height() - pm.height()) // 2
        px = event.pos().x() - ox
        py = event.pos().y() - oy
        if px < 0 or py < 0 or px >= pm.width() or py >= pm.height():
            return
        orig_x = int(px * self._original_size[0] / pm.width())
        orig_y = int(py * self._original_size[1] / pm.height())
        self.add_marker(orig_x, orig_y)
        self.clicked.emit(orig_x, orig_y)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()


# ── Visual Prompt Dialog ──

class VisualPromptDialog(QDialog):
    """Dialog for selecting an object in a video frame for visual prompting.

    Two views:
    - Frame selection: click on object in a frame, choose scale, generate mask
    - Mask preview: play back the masked video, accept or regenerate
    """

    def __init__(self, video_path: str, output_dir: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Visual Prompt — Select Object")
        self.setMinimumSize(750, 600)
        self._video_path = video_path
        self._output_dir = output_dir
        self._click_point: tuple[int, int] | None = None
        self._masked_video_path: str | None = None
        self._worker: MaskGenerationWorker | None = None
        self._processing = False
        self._frame_w = 0
        self._frame_h = 0

        # Apply dark theme
        self.setStyleSheet("""
            QDialog { background-color: #1e1e2e; color: #cdd6f4; }
            QLabel { color: #cdd6f4; }
            QPushButton {
                background-color: #313244; color: #cdd6f4;
                border: 1px solid #45475a; border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton:hover { background-color: #45475a; border-color: #89b4fa; }
            QPushButton:disabled { background-color: #1e1e2e; color: #585b70; }
            QComboBox {
                background-color: #313244; color: #cdd6f4;
                border: 1px solid #45475a; border-radius: 4px; padding: 6px;
            }
            QProgressBar {
                background-color: #313244; border: 1px solid #45475a;
                border-radius: 4px; text-align: center; color: #cdd6f4;
            }
            QProgressBar::chunk { background-color: #89b4fa; border-radius: 3px; }
            QGroupBox {
                color: #cdd6f4; border: 1px solid #45475a;
                border-radius: 4px; margin-top: 8px; padding-top: 12px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; }
        """)

        layout = QVBoxLayout(self)

        # ── Top row: backend + scale ──
        top_row = QHBoxLayout()

        # Backend selector
        top_row.addWidget(QLabel("Segmentation model:"))
        self._backend_combo = QComboBox()
        backend = check_vision_available()
        if backend == "sam2":
            self._backend_combo.addItem("SAM2 (installed)")
        elif backend == "sam3":
            self._backend_combo.addItem("SAM3 (installed)")
        else:
            self._backend_combo.addItem("Not installed")
        self._backend_combo.setEnabled(False)
        top_row.addWidget(self._backend_combo)

        top_row.addSpacing(16)

        # Video scale selector
        top_row.addWidget(QLabel("Processing scale:"))
        self._scale_combo = QComboBox()
        self._scale_combo.addItems(["1:1 (full)", "1/2", "1/3", "1/4"])
        self._scale_combo.setToolTip(
            "Downscale video before mask generation to reduce VRAM usage.\n"
            "Lower scale = faster + less memory, but lower mask resolution.")
        top_row.addWidget(self._scale_combo)

        top_row.addStretch()
        layout.addLayout(top_row)

        # Install instructions if no backend
        if backend is None:
            not_installed_label = QLabel(
                "Install ONE of:\n"
                "  SAM3 (recommended): git clone https://github.com/facebookresearch/sam3 && "
                "cd sam3 && pip install -e .\n"
                "  SAM2: git clone https://github.com/facebookresearch/sam2 && "
                "cd sam2 && pip install -e .\n"
                "Do NOT install both at the same time.")
            not_installed_label.setWordWrap(True)
            not_installed_label.setStyleSheet("color: #f9e2af; font-size: 10px; padding: 4px;")
            layout.addWidget(not_installed_label)

        # Instructions
        instructions = QLabel(
            "1. Navigate to the frame showing the object(s) you want to isolate.\n"
            "2. Click on each object to place a marker (one click per object).\n"
            "3. Click 'Generate Mask' — all marked objects will be segmented and tracked.")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #a6adc8; font-size: 11px; padding: 4px;")
        layout.addWidget(instructions)

        # ── Stacked widget: frame selector / video player ──
        self._stack = QStackedWidget()

        # Page 0: Frame selection
        frame_page = QWidget()
        frame_layout = QVBoxLayout(frame_page)
        frame_layout.setContentsMargins(0, 0, 0, 0)

        self._frame_widget = _ClickableFrame()
        self._frame_widget.clicked.connect(self._on_frame_clicked)
        frame_layout.addWidget(self._frame_widget, stretch=1)

        # Frame navigation slider
        nav_row = QHBoxLayout()
        nav_row.addWidget(QLabel("Frame:"))
        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setRange(0, 0)
        self._frame_slider.setValue(0)
        self._frame_slider.valueChanged.connect(self._on_frame_changed)
        nav_row.addWidget(self._frame_slider, stretch=1)
        self._frame_num_label = QLabel("0 / 0")
        nav_row.addWidget(self._frame_num_label)
        frame_layout.addLayout(nav_row)

        # Marker management row
        marker_row = QHBoxLayout()
        self._marker_label = QLabel("Markers: 0")
        self._marker_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
        marker_row.addWidget(self._marker_label)

        self._undo_marker_btn = QPushButton("Undo Last")
        self._undo_marker_btn.setEnabled(False)
        self._undo_marker_btn.clicked.connect(self._undo_last_marker)
        self._undo_marker_btn.setStyleSheet(
            "QPushButton { padding: 4px 10px; font-size: 11px; }")
        marker_row.addWidget(self._undo_marker_btn)

        self._clear_markers_btn = QPushButton("Clear All")
        self._clear_markers_btn.setEnabled(False)
        self._clear_markers_btn.clicked.connect(self._clear_all_markers)
        self._clear_markers_btn.setStyleSheet(
            "QPushButton { padding: 4px 10px; font-size: 11px; }")
        marker_row.addWidget(self._clear_markers_btn)

        marker_row.addStretch()
        frame_layout.addLayout(marker_row)

        self._stack.addWidget(frame_page)

        # Page 1: Video player for masked video preview
        player_page = QWidget()
        player_layout = QVBoxLayout(player_page)
        player_layout.setContentsMargins(0, 0, 0, 0)

        self._video_widget = QVideoWidget()
        self._video_widget.setMinimumSize(400, 300)
        self._video_widget.setStyleSheet("background-color: #181825;")
        player_layout.addWidget(self._video_widget, stretch=1)

        self._media_player = QMediaPlayer()
        self._audio_output = QAudioOutput()
        self._audio_output.setVolume(0.5)
        self._media_player.setAudioOutput(self._audio_output)
        self._media_player.setVideoOutput(self._video_widget)
        self._media_player.mediaStatusChanged.connect(self._on_media_status)

        # Playback controls
        playback_row = QHBoxLayout()
        self._play_btn = QPushButton("Play")
        self._play_btn.clicked.connect(self._toggle_playback)
        playback_row.addWidget(self._play_btn)

        self._back_to_frames_btn = QPushButton("Back to Frame Selection")
        self._back_to_frames_btn.clicked.connect(self._show_frame_selection)
        playback_row.addWidget(self._back_to_frames_btn)
        playback_row.addStretch()
        player_layout.addLayout(playback_row)

        self._stack.addWidget(player_page)
        layout.addWidget(self._stack, stretch=1)

        # ── Progress ──
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet("color: #a6adc8;")
        self._progress_label.setVisible(False)
        layout.addWidget(self._progress)
        layout.addWidget(self._progress_label)

        # ── Action buttons ──
        btn_row = QHBoxLayout()

        self._generate_btn = QPushButton("Generate Mask")
        self._generate_btn.setEnabled(False)
        self._generate_btn.clicked.connect(self._generate_mask)
        self._generate_btn.setStyleSheet(
            "QPushButton { background-color: #89b4fa; color: #1e1e2e; font-weight: bold; }"
            "QPushButton:hover { background-color: #b4d0fb; }"
            "QPushButton:disabled { background-color: #45475a; color: #585b70; }")
        btn_row.addWidget(self._generate_btn)

        self._accept_btn = QPushButton("Accept Mask")
        self._accept_btn.setStyleSheet(
            "QPushButton { background-color: #a6e3a1; color: #1e1e2e; font-weight: bold; }"
            "QPushButton:hover { background-color: #c6f0c6; }"
            "QPushButton:disabled { background-color: #45475a; color: #585b70; }")
        self._accept_btn.clicked.connect(self.accept)
        self._accept_btn.setEnabled(False)
        btn_row.addWidget(self._accept_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        # Extract frames for navigation
        self._frames_cache: list[str] = []
        self._extract_first_frames()

    def _extract_first_frames(self):
        """Extract frames from video for preview."""
        if not get_ffmpeg_executable():
            QMessageBox.critical(self, "FFmpeg Required",
                                 "FFmpeg is needed for video frame extraction.")
            return

        frames_dir = Path(self._output_dir) / "_preview_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Extract 1 frame per second for navigation
        try:
            subprocess.run(
                ffmpeg_command(
                    "-y", "-i", self._video_path,
                    "-vf", "fps=1", "-q:v", "2",
                    str(frames_dir / "preview_%04d.jpg"),
                ),
                capture_output=True, check=True, timeout=120,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            log.error(f"Frame extraction failed: {e}")
            try:
                subprocess.run(
                    ffmpeg_command(
                        "-y", "-i", self._video_path,
                        "-frames:v", "1",
                        str(frames_dir / "preview_0001.jpg"),
                    ),
                    capture_output=True, check=True,
                )
            except Exception:
                return

        self._frames_cache = sorted(str(p) for p in frames_dir.glob("preview_*.jpg"))
        if self._frames_cache:
            self._frame_slider.setRange(0, len(self._frames_cache) - 1)
            self._frame_num_label.setText(f"1 / {len(self._frames_cache)}")
            self._load_frame(0)

    def _load_frame(self, index: int):
        if index < 0 or index >= len(self._frames_cache):
            return
        img = QImage(self._frames_cache[index])
        if img.isNull():
            return
        self._frame_w = img.width()
        self._frame_h = img.height()
        self._frame_widget.set_frame(img, img.width(), img.height())

    def _on_frame_changed(self, index: int):
        if self._processing:
            return
        self._frame_num_label.setText(f"{index + 1} / {len(self._frames_cache)}")
        self._load_frame(index)
        self._click_point = None
        self._update_marker_ui()
        self._generate_btn.setEnabled(False)

    def _on_frame_clicked(self, x: int, y: int):
        self._click_point = (x, y)
        self._update_marker_ui()
        self._generate_btn.setEnabled(check_vision_available() is not None)

    def _update_marker_ui(self):
        """Update marker count label and button states."""
        n = len(self._frame_widget.markers)
        self._marker_label.setText(f"Markers: {n}")
        self._undo_marker_btn.setEnabled(n > 0 and not self._processing)
        self._clear_markers_btn.setEnabled(n > 0 and not self._processing)
        self._generate_btn.setEnabled(n > 0 and check_vision_available() is not None
                                      and not self._processing)

    def _undo_last_marker(self):
        self._frame_widget.remove_last_marker()
        markers = self._frame_widget.markers
        self._click_point = markers[-1] if markers else None
        self._update_marker_ui()

    def _clear_all_markers(self):
        self._frame_widget.clear_markers()
        self._click_point = None
        self._update_marker_ui()

    def _get_video_scale(self) -> int:
        """Return scale factor from combo: 1, 2, 3, or 4."""
        idx = self._scale_combo.currentIndex()
        return [1, 2, 3, 4][idx]

    def _generate_mask(self):
        all_markers = self._frame_widget.markers
        if not all_markers:
            return
        backend = check_vision_available()
        if backend is None:
            QMessageBox.warning(self, "No Backend",
                                "Install SAM2 or SAM3 for visual prompting.")
            return

        frame_idx = self._frame_slider.value()

        if backend == "sam3":
            model_id = "jetjodh/sam3"
        else:
            model_id = "facebook/sam2.1-hiera-large"

        # Lock UI during processing
        self._processing = True
        self._frame_widget.set_clicks_locked(True)
        self._frame_slider.setEnabled(False)
        self._scale_combo.setEnabled(False)
        self._generate_btn.setEnabled(False)
        self._accept_btn.setEnabled(False)
        self._undo_marker_btn.setEnabled(False)
        self._clear_markers_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._progress_label.setVisible(True)

        self._worker = MaskGenerationWorker(
            video_path=self._video_path,
            click_point=all_markers[0],
            click_points=all_markers,
            frame_size=(self._frame_w, self._frame_h),
            output_dir=self._output_dir,
            backend=backend,
            model_id=model_id,
            frame_index=frame_idx,
            video_scale=self._get_video_scale(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_mask_done)
        self._worker.error.connect(self._on_mask_error)
        self._worker.start()

    def _on_progress(self, pct: int, msg: str):
        self._progress.setValue(pct)
        self._progress_label.setText(msg)

    def _unlock_ui(self):
        """Restore UI after processing completes."""
        self._processing = False
        self._frame_widget.set_clicks_locked(False)
        self._frame_slider.setEnabled(True)
        self._scale_combo.setEnabled(True)
        self._generate_btn.setEnabled(True)
        self._update_marker_ui()

    def _on_mask_done(self, masked_path: str):
        self._masked_video_path = masked_path
        self._progress.setVisible(False)
        self._progress_label.setText("Mask ready! Play the preview or accept the mask.")
        self._unlock_ui()
        self._generate_btn.setText("Regenerate Mask")
        self._accept_btn.setEnabled(True)

        # Save mask project for reuse
        all_markers = self._frame_widget.markers
        if all_markers:
            try:
                project_dir = str(Path(self._output_dir) / "mask_project")
                backend = check_vision_available() or "unknown"
                saved_path = save_mask_project(
                    project_dir=project_dir,
                    video_path=self._video_path,
                    click_point=all_markers[0],
                    click_points=all_markers,
                    frame_index=self._frame_slider.value(),
                    video_scale=self._get_video_scale(),
                    backend=backend,
                    masked_video_path=masked_path,
                )
                self._masked_video_path = saved_path
            except Exception as e:
                log.warning("Failed to save mask project: %s", e)

        # Switch to video player to preview the masked video
        self._show_mask_preview(self._masked_video_path)

    def _show_mask_preview(self, masked_path: str):
        """Load the masked video into the video player for preview."""
        self._media_player.setSource(QUrl.fromLocalFile(masked_path))
        self._stack.setCurrentIndex(1)  # switch to video player page
        self._play_btn.setText("Play")

    def _toggle_playback(self):
        if self._media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._media_player.pause()
            self._play_btn.setText("Play")
        else:
            # Restart from beginning if at end
            if self._media_player.mediaStatus() == QMediaPlayer.MediaStatus.EndOfMedia:
                self._media_player.setPosition(0)
            self._media_player.play()
            self._play_btn.setText("Pause")

    def _on_media_status(self, status):
        """Reset play button text when media finishes."""
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self._play_btn.setText("Play")

    def _show_frame_selection(self):
        """Switch back to frame selection view."""
        self._media_player.stop()
        self._play_btn.setText("Play")
        self._stack.setCurrentIndex(0)

    def _on_mask_error(self, error_msg: str):
        self._progress.setVisible(False)
        self._progress_label.setVisible(False)
        self._unlock_ui()
        QMessageBox.critical(self, "Mask Generation Failed", error_msg)

    @property
    def masked_video_path(self) -> str | None:
        return self._masked_video_path

    def closeEvent(self, event):
        self._media_player.stop()
        # Cleanup preview frames on close
        frames_dir = Path(self._output_dir) / "_preview_frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir, ignore_errors=True)
        super().closeEvent(event)
