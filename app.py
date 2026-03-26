"""
app.py  –  MediaPipe Pose Detection  |  Main GUI entry point
Uses CustomTkinter for a modern dark-themed desktop UI.
"""

import threading
import time
import tkinter as tk
from tkinter import filedialog
from typing import Optional

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from pose_detector import PoseDetector
from video_source import VideoSource
from model_3d import Model3DRenderer

# ─────────────────────────────────────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

ACCENT      = "#00C878"       # neon green
BG_DARK     = "#0D1117"
BG_PANEL    = "#161B22"
BG_SIDEBAR  = "#0D1117"
TEXT_MUTED  = "#8B949E"
TEXT_LIGHT  = "#E6EDF3"
DANGER      = "#F85149"


# ─────────────────────────────────────────────────────────────────────────────
# Main App Window
# ─────────────────────────────────────────────────────────────────────────────
class PoseApp(ctk.CTk):
    """Root application window."""

    CANVAS_W = 900
    CANVAS_H = 560

    def __init__(self):
        super().__init__()

        self.title("Pose Detector  •  MediaPipe")
        self.geometry("1260x700")
        self.minsize(1000, 620)
        self.configure(fg_color=BG_DARK)

        # ── State ──────────────────────────────────────────────────────────
        self._running   = False
        self._thread: Optional[threading.Thread] = None
        self._source: Optional[VideoSource] = None
        self._detector: Optional[PoseDetector] = None
        self._last_image_ref = None           # prevent GC of PhotoImage
        self._last_3d_ref    = None           # prevent GC of 3D PhotoImage
        self._last_3d_video_ref = None        # prevent GC of 3D side-by-side video
        self._frame_count = 0
        self._fps_display = 0.0
        self._t_last = time.time()
        self._anim_job   = None               # webcam idle animation after-id
        self._anim_phase = 0                  # animation tick counter
        self._full_path: Optional[str] = None
        self._active_tab  = "video"           # "video" or "3d"
        self._renderer    = Model3DRenderer()
        self._yaw_var     = tk.DoubleVar(value=0.0)
        self._pitch_var   = tk.DoubleVar(value=15.0)
        self._zoom_var    = tk.DoubleVar(value=1.0)
        self._3d_idle_job = None
        self._3d_idle_tick = 0
        self._drag_start: Optional[tuple[int, int]] = None
        self._drag_yaw0  = 0.0
        self._drag_pitch0 = 0.0
        self._latest_landmarks_3d: list = []  # latest 3D landmarks for 3D tab

        # ── Layout ─────────────────────────────────────────────────────────
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main_panel()
        self._build_status_bar()

        # ── App Icon ───────────────────────────────────────────────────────
        try:
            import os
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
            if os.path.exists(icon_path):
                self.iconbitmap(icon_path)
            else:
                # Fallback to png
                png_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
                if os.path.exists(png_path):
                    self._app_icon_img = ImageTk.PhotoImage(file=png_path)
                    self.iconphoto(False, self._app_icon_img)
        except Exception as e:
            print("Could not load app icon:", e)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ══════════════════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════

    def _build_sidebar(self) -> None:
        sb = ctk.CTkFrame(self, width=270, corner_radius=0, fg_color=BG_SIDEBAR)
        sb.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        sb.grid_propagate(False)
        sb.grid_rowconfigure(20, weight=1)   # spacer

        pad = {"padx": 18, "pady": 6}

        # ── Logo ────────────────────────────────────────────────────────
        logo_lbl = ctk.CTkLabel(
            sb, text="🦾  Pose Detector",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=ACCENT,
        )
        logo_lbl.grid(row=0, column=0, sticky="w", padx=18, pady=(24, 4))

        sub_lbl = ctk.CTkLabel(
            sb, text="Powered by Google MediaPipe",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
        )
        sub_lbl.grid(row=1, column=0, sticky="w", **pad)

        _sep(sb, row=2)

        # ── Mode selector ────────────────────────────────────────────────
        ctk.CTkLabel(sb, text="INPUT SOURCE", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=TEXT_MUTED).grid(row=3, column=0, sticky="w", **pad)

        self._mode_var = ctk.StringVar(value="video")

        self._rb_video = ctk.CTkRadioButton(
            sb, text="📂  Video File", variable=self._mode_var,
            value="video", font=ctk.CTkFont(size=14),
            command=self._on_mode_change,
            fg_color=ACCENT, hover_color="#00B060",
        )
        self._rb_video.grid(row=4, column=0, sticky="w", **pad)

        self._rb_cam = ctk.CTkRadioButton(
            sb, text="📷  Live Webcam", variable=self._mode_var,
            value="webcam", font=ctk.CTkFont(size=14),
            command=self._on_mode_change,
            fg_color=ACCENT, hover_color="#00B060",
        )
        self._rb_cam.grid(row=5, column=0, sticky="w", **pad)

        _sep(sb, row=6)

        # ── Video file section ───────────────────────────────────────────
        self._video_frame = ctk.CTkFrame(sb, fg_color="transparent")
        self._video_frame.grid(row=7, column=0, sticky="ew", padx=14, pady=4)

        ctk.CTkLabel(self._video_frame, text="VIDEO FILE",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=TEXT_MUTED).pack(anchor="w", pady=(0, 4))

        self._file_path_var = ctk.StringVar(value="No file selected")
        self._file_label = ctk.CTkLabel(
            self._video_frame, textvariable=self._file_path_var,
            text_color=TEXT_MUTED, font=ctk.CTkFont(size=11),
            wraplength=220, justify="left",
        )
        self._file_label.pack(anchor="w", pady=(0, 6))

        self._browse_btn = ctk.CTkButton(
            self._video_frame, text="Browse…", height=32,
            font=ctk.CTkFont(size=13),
            fg_color=BG_PANEL, hover_color="#21262D",
            border_color=ACCENT, border_width=1,
            command=self._browse_file,
        )
        self._browse_btn.pack(fill="x")

        # ── Webcam section ───────────────────────────────────────────────
        self._cam_frame = ctk.CTkFrame(sb, fg_color="transparent")
        self._cam_frame.grid(row=8, column=0, sticky="ew", padx=14, pady=4)

        ctk.CTkLabel(self._cam_frame, text="CAMERA DEVICE",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=TEXT_MUTED).pack(anchor="w", pady=(0, 4))

        self._cam_var = ctk.StringVar(value="Camera 0 (Default)")
        self._cam_menu = ctk.CTkOptionMenu(
            self._cam_frame,
            values=["Camera 0 (Default)", "Camera 1", "Camera 2", "Camera 3"],
            variable=self._cam_var,
            height=32,
            fg_color=BG_PANEL, button_color=ACCENT,
            button_hover_color="#00B060",
        )
        self._cam_menu.pack(fill="x")

        _sep(sb, row=9)

        # ── Detection settings ───────────────────────────────────────────
        ctk.CTkLabel(sb, text="DETECTION SETTINGS",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=TEXT_MUTED).grid(row=10, column=0, sticky="w", **pad)

        # Detection confidence
        ctk.CTkLabel(sb, text="Detection Confidence",
                     font=ctk.CTkFont(size=12),
                     text_color=TEXT_LIGHT).grid(row=11, column=0, sticky="w", padx=18, pady=(6, 0))

        self._det_conf_var = ctk.DoubleVar(value=0.5)
        self._det_conf_label = ctk.CTkLabel(sb, text="0.50", text_color=ACCENT,
                                            font=ctk.CTkFont(size=12))
        self._det_conf_label.grid(row=12, column=0, sticky="e", padx=18)

        self._det_slider = ctk.CTkSlider(
            sb, from_=0.1, to=1.0, variable=self._det_conf_var,
            command=self._on_det_slider,
            progress_color=ACCENT, button_color=ACCENT,
        )
        self._det_slider.grid(row=13, column=0, sticky="ew", padx=18, pady=(0, 6))

        # Tracking confidence
        ctk.CTkLabel(sb, text="Tracking Confidence",
                     font=ctk.CTkFont(size=12),
                     text_color=TEXT_LIGHT).grid(row=14, column=0, sticky="w", padx=18, pady=(4, 0))

        self._trk_conf_var = ctk.DoubleVar(value=0.5)
        self._trk_conf_label = ctk.CTkLabel(sb, text="0.50", text_color=ACCENT,
                                            font=ctk.CTkFont(size=12))
        self._trk_conf_label.grid(row=15, column=0, sticky="e", padx=18)

        self._trk_slider = ctk.CTkSlider(
            sb, from_=0.1, to=1.0, variable=self._trk_conf_var,
            command=self._on_trk_slider,
            progress_color=ACCENT, button_color=ACCENT,
        )
        self._trk_slider.grid(row=16, column=0, sticky="ew", padx=18, pady=(0, 6))

        _sep(sb, row=17)

        # ── Start / Stop button ──────────────────────────────────────────
        self._start_btn = ctk.CTkButton(
            sb, text="▶   Start Detection",
            height=44, font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=ACCENT, hover_color="#00B060",
            text_color="#000000",
            command=self._toggle_detection,
        )
        self._start_btn.grid(row=18, column=0, sticky="ew", padx=18, pady=(8, 4))

        # ── Info card ────────────────────────────────────────────────────
        info = ctk.CTkFrame(sb, fg_color=BG_PANEL, corner_radius=10)
        info.grid(row=19, column=0, sticky="ew", padx=18, pady=(8, 18))

        self._fps_info   = ctk.CTkLabel(info, text="FPS: —",
                                        font=ctk.CTkFont(size=13, weight="bold"),
                                        text_color=ACCENT)
        self._fps_info.pack(anchor="w", padx=12, pady=(8, 2))

        self._lm_info    = ctk.CTkLabel(info, text="Landmarks: —",
                                        font=ctk.CTkFont(size=12),
                                        text_color=TEXT_LIGHT)
        self._lm_info.pack(anchor="w", padx=12, pady=(0, 8))

        # Initial state
        self._on_mode_change()

    def _build_main_panel(self) -> None:
        panel = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=0)
        panel.grid(row=0, column=1, sticky="nsew")
        panel.grid_rowconfigure(1, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        # ── Header bar with tab buttons ──────────────────────────────────
        header = ctk.CTkFrame(panel, fg_color="#161B22", height=48, corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)

        self._tab_video_btn = ctk.CTkButton(
            header, text="📹  Video Feed", width=140, height=34,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=ACCENT, hover_color="#00B060", text_color="#000000",
            corner_radius=6, command=lambda: self._switch_tab("video"),
        )
        self._tab_video_btn.pack(side="left", padx=(16, 4), pady=6)

        self._tab_3d_btn = ctk.CTkButton(
            header, text="🧍  3D Model", width=140, height=34,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#21262D", hover_color="#30363D", text_color=TEXT_MUTED,
            corner_radius=6, command=lambda: self._switch_tab("3d"),
        )
        self._tab_3d_btn.pack(side="left", padx=4, pady=6)

        self._source_badge = ctk.CTkLabel(
            header, text="● IDLE", font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        )
        self._source_badge.pack(side="right", padx=20)

        # ── Video canvas (tab 1) ─────────────────────────────────────────
        self._video_wrap = ctk.CTkFrame(panel, fg_color="#0D1117", corner_radius=12)
        self._video_wrap.grid(row=1, column=0, sticky="nsew", padx=16, pady=16)
        self._video_wrap.grid_rowconfigure(0, weight=1)
        self._video_wrap.grid_columnconfigure(0, weight=1)

        self._canvas = tk.Canvas(
            self._video_wrap, bg="#0D1117", highlightthickness=0,
            cursor="crosshair",
        )
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._canvas.bind("<Configure>", self._on_canvas_resize)

        # Placeholder text
        self._placeholder_id = self._canvas.create_text(
            self.CANVAS_W // 2, self.CANVAS_H // 2,
            text="No source loaded.\nChoose a video file or webcam and press  ▶ Start.",
            fill=TEXT_MUTED, font=("Consolas", 14), justify="center",
        )

        # ── 3D canvas (tab 2) ────────────────────────────────────────────
        self._3d_wrap = ctk.CTkFrame(panel, fg_color="#0D1117", corner_radius=12)
        # Initially hidden; same grid cell as video_wrap
        self._3d_wrap.grid(row=1, column=0, sticky="nsew", padx=16, pady=16)
        self._3d_wrap.grid_rowconfigure(0, weight=1)
        self._3d_wrap.grid_columnconfigure(0, weight=1)
        self._3d_wrap.grid_columnconfigure(1, weight=1)
        self._3d_wrap.grid_remove()  # hidden by default

        # Side-by-side video feed canvas
        self._canvas_3d_video = tk.Canvas(
            self._3d_wrap, bg="#0D1117", highlightthickness=0, cursor="crosshair",
        )
        self._canvas_3d_video.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        
        self._canvas_3d_video.create_text(
            400, 300, text="Original Video Feed",
            fill=TEXT_MUTED, font=("Consolas", 14), justify="center", tags="placeholder"
        )

        self._canvas_3d = tk.Canvas(
            self._3d_wrap, bg="#0D1117", highlightthickness=0, cursor="hand2",
        )
        self._canvas_3d.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        # Mouse drag and wheel for 3D view
        self._canvas_3d.bind("<ButtonPress-1>", self._on_3d_drag_start)
        self._canvas_3d.bind("<B1-Motion>", self._on_3d_drag_move)
        self._canvas_3d.bind("<ButtonRelease-1>", self._on_3d_drag_end)
        self._canvas_3d.bind("<MouseWheel>", self._on_3d_mouse_wheel)

        # ── 3D Controls (Sliders + Reset) ──────────────────────────────────
        self._3d_wrap.grid_rowconfigure(1, weight=0)
        ctrl_frame = ctk.CTkFrame(self._3d_wrap, fg_color="transparent")
        ctrl_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 16))

        # Yaw
        ctk.CTkLabel(ctrl_frame, text="Yaw", font=ctk.CTkFont(size=11), text_color=TEXT_MUTED).pack(side="left", padx=(0, 4))
        self._yaw_slider = ctk.CTkSlider(ctrl_frame, variable=self._yaw_var, from_=-180, to=180, width=100, command=self._on_3d_slider_change)
        self._yaw_slider.pack(side="left", padx=(0, 16))

        # Pitch
        ctk.CTkLabel(ctrl_frame, text="Pitch", font=ctk.CTkFont(size=11), text_color=TEXT_MUTED).pack(side="left", padx=(0, 4))
        self._pitch_slider = ctk.CTkSlider(ctrl_frame, variable=self._pitch_var, from_=-80, to=80, width=100, command=self._on_3d_slider_change)
        self._pitch_slider.pack(side="left", padx=(0, 16))

        # Zoom
        ctk.CTkLabel(ctrl_frame, text="Zoom", font=ctk.CTkFont(size=11), text_color=TEXT_MUTED).pack(side="left", padx=(0, 4))
        self._zoom_slider = ctk.CTkSlider(ctrl_frame, variable=self._zoom_var, from_=0.5, to=3.0, width=100, command=self._on_3d_slider_change)
        self._zoom_slider.pack(side="left", padx=(0, 16))

        # Reset button
        ctk.CTkButton(ctrl_frame, text="Reset View", width=80, height=24, fg_color="#30363D", hover_color="#40464D", font=ctk.CTkFont(size=11), command=self._reset_3d_view).pack(side="right")

    def _build_status_bar(self) -> None:
        self._status_var = tk.StringVar(value="Ready  •  Select a source and press Start")
        bar = ctk.CTkFrame(self, height=32, corner_radius=0, fg_color="#010409")
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        bar.grid_propagate(False)

        ctk.CTkLabel(
            bar, textvariable=self._status_var,
            font=ctk.CTkFont(size=11), text_color=TEXT_MUTED,
        ).pack(side="left", padx=16)

        ctk.CTkLabel(
            bar, text="MediaPipe • OpenCV • CustomTkinter",
            font=ctk.CTkFont(size=11), text_color="#30363D",
        ).pack(side="right", padx=16)

    # ══════════════════════════════════════════════════════════════════════
    #  EVENT HANDLERS
    # ══════════════════════════════════════════════════════════════════════

    def _on_mode_change(self) -> None:
        # Called during sidebar init before canvas exists – skip visual updates
        if not hasattr(self, "_canvas"):
            mode = self._mode_var.get()
            if mode == "video":
                self._video_frame.grid()
                self._cam_frame.grid_remove()
            else:
                self._video_frame.grid_remove()
                self._cam_frame.grid()
            return

        if self._running:
            return
        mode = self._mode_var.get()
        if mode == "video":
            self._video_frame.grid()
            self._cam_frame.grid_remove()
            self._stop_webcam_anim()
            # Restore first-frame preview if a file was already chosen
            if self._full_path:
                self._show_video_first_frame(self._full_path)
            else:
                self._show_placeholder()
        else:
            self._video_frame.grid_remove()
            self._cam_frame.grid()
            self._start_webcam_anim()

    def _browse_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
                ("All files", "*.*"),
            ],
        )
        if path:
            short = path if len(path) < 35 else "…" + path[-32:]
            self._file_path_var.set(short)
            self._full_path = path
            self._set_status(f"File selected: {path.split('/')[-1]}  •  Press ▶ Start to begin")
            self._show_video_first_frame(path)
        else:
            if not self._full_path:
                self._full_path = None

    def _on_det_slider(self, val) -> None:
        self._det_conf_label.configure(text=f"{val:.2f}")

    def _on_trk_slider(self, val) -> None:
        self._trk_conf_label.configure(text=f"{val:.2f}")

    def _on_canvas_resize(self, event) -> None:
        # Recentre placeholder text
        self._canvas.coords(
            self._placeholder_id, event.width // 2, event.height // 2
        )

    def _toggle_detection(self) -> None:
        if self._running:
            self._stop_detection()
        else:
            self._start_detection()

    # ══════════════════════════════════════════════════════════════════════
    #  DETECTION CONTROL
    # ══════════════════════════════════════════════════════════════════════

    def _start_detection(self) -> None:
        mode = self._mode_var.get()

        if mode == "video":
            path = getattr(self, "_full_path", None)
            if not path:
                self._set_status("⚠  Please select a video file first.", error=True)
                return
            source_arg = path
            badge_text = f"● VIDEO  {path.split('/')[-1]}"
        else:
            cam_idx = int(self._cam_var.get().split()[1])
            source_arg = cam_idx
            badge_text = f"● WEBCAM  (index {cam_idx})"

        # Disable controls immediately
        self._start_btn.configure(text="⏳   Opening…",
                                  fg_color="#30363D", hover_color="#30363D",
                                  text_color=TEXT_MUTED, state="disabled")
        self._det_slider.configure(state="disabled")
        self._trk_slider.configure(state="disabled")
        self._rb_video.configure(state="disabled")
        self._rb_cam.configure(state="disabled")
        self._browse_btn.configure(state="disabled")
        self._cam_menu.configure(state="disabled")

        # Stop webcam idle animation
        self._stop_webcam_anim()

        # Start the loading animation immediately
        self._loading_active = True
        self._loading_phase  = 0
        self._loading_msg    = "Initializing…"
        self._canvas.itemconfigure(self._placeholder_id, state="hidden")
        self._loading_tick()

        # Heavy work (webcam open + model download) in a background thread
        self._init_args = (source_arg, badge_text)
        threading.Thread(target=self._loading_init_thread, daemon=True).start()

    def _loading_init_thread(self) -> None:
        """Background thread: open the video source & create the detector."""
        source_arg, badge_text = self._init_args

        self.after(0, self._update_loading_msg, "Opening camera…" if isinstance(source_arg, int) else "Opening video…")

        src = VideoSource(source_arg)
        ok  = src.open()

        if not ok:
            self.after(0, self._on_source_failed)
            return

        self.after(0, self._update_loading_msg, "Loading pose model…")

        det = PoseDetector(
            min_detection_confidence=self._det_conf_var.get(),
            min_tracking_confidence=self._trk_conf_var.get(),
            progress_cb=lambda msg: self.after(0, self._update_loading_msg, msg),
        )

        # Signal the main thread that we're ready
        self.after(0, self._on_source_ready, src, det, badge_text)

    def _update_loading_msg(self, msg: str) -> None:
        self._loading_msg = msg
        self._set_status(msg)

    def _on_source_failed(self) -> None:
        """Called on main thread when the source fails to open."""
        self._loading_active = False
        self._set_status("⚠  Failed to open video source.", error=True)
        # Re-enable controls
        self._start_btn.configure(text="▶   Start Detection",
                                  fg_color=ACCENT, hover_color="#00B060",
                                  text_color="#000000", state="normal")
        self._det_slider.configure(state="normal")
        self._trk_slider.configure(state="normal")
        self._rb_video.configure(state="normal")
        self._rb_cam.configure(state="normal")
        self._browse_btn.configure(state="normal")
        self._cam_menu.configure(state="normal")
        # Restore idle visuals
        if self._mode_var.get() == "webcam":
            self._start_webcam_anim()
        elif self._full_path:
            self._show_video_first_frame(self._full_path)
        else:
            self._show_placeholder()

    def _on_source_ready(self, src: "VideoSource", det: "PoseDetector", badge_text: str) -> None:
        """Called on main thread once source + detector are initialised."""
        self._loading_active = False
        self._source   = src
        self._detector  = det

        self._running      = True
        self._frame_count  = 0
        self._t_last       = time.time()

        self._start_btn.configure(text="⏹   Stop Detection",
                                  fg_color=DANGER, hover_color="#C84040",
                                  text_color=TEXT_LIGHT, state="normal")
        self._source_badge.configure(text=badge_text, text_color=ACCENT)
        self._set_status(f"Running  •  {badge_text.strip('● ')}")

        # Start processing thread
        self._thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._thread.start()

    # ── Loading animation ────────────────────────────────────────────────

    def _loading_tick(self) -> None:
        """Draw one frame of the loading animation, schedule next."""
        if not self._loading_active:
            return
        frame = self._draw_loading_frame()
        if self._active_tab == "3d":
            self._render_video_on_3d_canvas(frame)
            self._render_3d_on_canvas([], self._loading_phase)
        else:
            self._show_frame_on_canvas(frame)
        self._loading_phase += 1
        self.after(35, self._loading_tick)

    def _draw_loading_frame(self) -> "np.ndarray":
        """Render a spinning-ring + message loading screen."""
        import math
        W, H = 960, 540
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        cx, cy = W // 2, H // 2
        t = self._loading_phase

        # Dark radial background
        for rad in range(220, 0, -8):
            a = int(14 * (1 - rad / 220))
            cv2.circle(frame, (cx, cy), rad, (0, a, a // 2), -1)

        # Spinning arcs (multiple concentric rings)
        for ring_i, (radius, thick, speed) in enumerate([
            (80, 4, 5), (60, 3, -7), (100, 2, 3),
        ]):
            start_angle = (t * speed) % 360
            sweep = 90 + 30 * math.sin(t * 0.08 + ring_i)
            brightness = int(140 + 80 * math.sin(t * 0.1 + ring_i))
            color = (0, brightness, brightness // 2)
            cv2.ellipse(frame, (cx, cy - 30), (radius, radius),
                        0, start_angle, start_angle + sweep,
                        color, thick, cv2.LINE_AA)

        # Pulsing inner dot
        dot_r = int(6 + 3 * math.sin(t * 0.15))
        cv2.circle(frame, (cx, cy - 30), dot_r, (0, 255, 140), -1, cv2.LINE_AA)

        # Status message
        msg = self._loading_msg
        # Animated trailing dots
        dots = "." * ((t // 8) % 4)
        display_msg = msg.rstrip(".…") + dots

        text_size = cv2.getTextSize(display_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        tx = cx - text_size[0] // 2
        ty = cy + 80
        cv2.putText(frame, display_msg, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 120), 1, cv2.LINE_AA)

        # Subtle scanning line
        scan_y = int(H * 0.2 + H * 0.6 * ((t * 4) % 120) / 120)
        cv2.line(frame, (int(W * 0.2), scan_y), (int(W * 0.8), scan_y),
                 (0, 40, 20), 1, cv2.LINE_AA)

        return frame

    def _stop_detection(self) -> None:
        self._running = False
        self._loading_active = False
        # Thread will exit on its own
        if self._source:
            self._source.release()
            self._source = None
        if self._detector:
            self._detector.close()
            self._detector = None

        self._start_btn.configure(text="▶   Start Detection",
                                  fg_color=ACCENT, hover_color="#00B060",
                                  text_color="#000000")
        self._source_badge.configure(text="● IDLE", text_color=TEXT_MUTED)
        self._fps_info.configure(text="FPS: —")
        self._lm_info.configure(text="Landmarks: —")
        self._set_status("Stopped  •  Select a source and press Start")

        # Re-enable settings
        self._det_slider.configure(state="normal")
        self._trk_slider.configure(state="normal")
        self._rb_video.configure(state="normal")
        self._rb_cam.configure(state="normal")
        self._browse_btn.configure(state="normal")
        self._cam_menu.configure(state="normal")

        # Restore idle visuals
        if self._mode_var.get() == "webcam":
            self._start_webcam_anim()
        elif self._full_path:
            self._show_video_first_frame(self._full_path)
        else:
            self._show_placeholder()
        # Restart 3D idle if on 3D tab
        if self._active_tab == "3d":
            self._start_3d_idle()

    # ══════════════════════════════════════════════════════════════════════
    #  BACKGROUND PROCESSING LOOP
    # ══════════════════════════════════════════════════════════════════════

    def _processing_loop(self) -> None:
        """Runs in a background thread; pushes frames to the Tk canvas."""
        tick = 0
        while self._running:
            frame = self._source.get_frame() if self._source else None

            if frame is None:
                # Video file ended or error
                self.after(0, self._on_stream_ended)
                break

            annotated, lm_count, landmarks_3d = self._detector.detect(frame)

            # FPS calculation
            self._frame_count += 1
            now = time.time()
            elapsed = now - self._t_last
            if elapsed >= 0.5:
                self._fps_display = self._frame_count / elapsed
                self._frame_count = 0
                self._t_last = now

            tick += 1
            # Schedule UI update on main thread
            self.after(0, self._update_canvas, annotated, lm_count, landmarks_3d, tick)

            # Small sleep to be kind to the event loop (≈1 ms)
            time.sleep(0.001)

    def _on_stream_ended(self) -> None:
        self._set_status("▣  Video finished.  Press ▶ Start to replay.")
        self._stop_detection()
        # show first frame again as "end card"
        if self._full_path:
            self._show_video_first_frame(self._full_path)

    # ══════════════════════════════════════════════════════════════════════
    #  CANVAS UPDATE  (main thread)
    # ══════════════════════════════════════════════════════════════════════

    def _update_canvas(self, frame: np.ndarray, lm_count: int,
                       landmarks_3d: list = None, tick: int = 0) -> None:
        if not self._running:
            return

        # Store latest 3D landmarks for tab switching
        if landmarks_3d:
            self._latest_landmarks_3d = landmarks_3d

        # --- Video tab ---
        if self._active_tab == "video":
            cw = self._canvas.winfo_width()
            ch = self._canvas.winfo_height()
            if cw < 2 or ch < 2:
                return

            fh, fw = frame.shape[:2]
            scale = min(cw / fw, ch / fh)
            nw, nh = int(fw * scale), int(fh * scale)
            resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))

            x_off = (cw - nw) // 2
            y_off = (ch - nh) // 2

            self._canvas.delete("frame")
            self._canvas.create_image(x_off, y_off, anchor="nw",
                                      image=photo, tags="frame")
            self._last_image_ref = photo

        # --- 3D tab ---
        elif self._active_tab == "3d" and landmarks_3d:
            self._render_3d_on_canvas(landmarks_3d, tick)
            self._render_video_on_3d_canvas(frame)

        # Update sidebar info
        self._fps_info.configure(text=f"FPS: {self._fps_display:.1f}")
        self._lm_info.configure(
            text=f"Landmarks: {lm_count}" if lm_count else "Landmarks: none"
        )

    # ══════════════════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _show_placeholder(self) -> None:
        """Show the default 'no source' text on the canvas."""
        self._canvas.delete("frame")
        self._canvas.itemconfigure(self._placeholder_id, state="normal")
        self._canvas.itemconfig(
            self._placeholder_id,
            text="No source loaded.\nChoose a video file or webcam and press  ▶ Start.",
        )

    def _show_frame_on_canvas(self, frame: np.ndarray) -> None:
        """Resize a BGR frame to fit the canvas and display it."""
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 2 or ch < 2:
            cw, ch = self.CANVAS_W, self.CANVAS_H

        fh, fw = frame.shape[:2]
        scale = min(cw / fw, ch / fh)
        nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb   = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))

        self._canvas.itemconfigure(self._placeholder_id, state="hidden")
        self._canvas.delete("frame")
        self._canvas.create_image(
            (cw - nw) // 2, (ch - nh) // 2,
            anchor="nw", image=photo, tags="frame",
        )
        self._last_image_ref = photo

    def _show_video_first_frame(self, path: str) -> None:
        """Extract and display the first frame of a video file."""
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            # Draw a semi-transparent 'preview' banner
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 38), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            cv2.putText(
                frame, "  PREVIEW  –  Press  Start  to begin",
                (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 200, 120), 1, cv2.LINE_AA,
            )
            self._show_frame_on_canvas(frame)

    # ── Webcam idle animation ──────────────────────────────────────────────

    # Stick-figure skeleton: (x_norm, y_norm) for 17 key points
    _DEMO_PTS: list[tuple[float, float]] = [
        (0.50, 0.12),  # 0  nose
        (0.47, 0.10),  # 1  left eye
        (0.53, 0.10),  # 2  right eye
        (0.44, 0.11),  # 3  left ear
        (0.56, 0.11),  # 4  right ear
        (0.38, 0.28),  # 5  left shoulder
        (0.62, 0.28),  # 6  right shoulder
        (0.30, 0.46),  # 7  left elbow
        (0.70, 0.46),  # 8  right elbow
        (0.25, 0.62),  # 9  left wrist
        (0.75, 0.62),  # 10 right wrist
        (0.42, 0.58),  # 11 left hip
        (0.58, 0.58),  # 12 right hip
        (0.40, 0.76),  # 13 left knee
        (0.60, 0.76),  # 14 right knee
        (0.38, 0.94),  # 15 left ankle
        (0.62, 0.94),  # 16 right ankle
    ]
    _DEMO_LINKS: list[tuple[int, int]] = [
        (0,1),(0,2),(1,3),(2,4),            # face
        (5,6),(5,7),(7,9),(6,8),(8,10),     # arms
        (5,11),(6,12),(11,12),              # torso
        (11,13),(13,15),(12,14),(14,16),    # legs
    ]

    def _start_webcam_anim(self) -> None:
        """Begin the pulsing stick-figure animation."""
        self._stop_webcam_anim()
        self._anim_phase = 0
        self._tick_webcam_anim()

    def _stop_webcam_anim(self) -> None:
        if self._anim_job is not None:
            try:
                self.after_cancel(self._anim_job)
            except Exception:
                pass
            self._anim_job = None

    def _tick_webcam_anim(self) -> None:
        """Draw one frame of the idle animation, then schedule the next."""
        if self._running:
            return
        if self._mode_var.get() != "webcam":
            return

        frame = self._draw_webcam_anim_frame()
        self._show_frame_on_canvas(frame)
        self._anim_phase += 1
        self._anim_job = self.after(45, self._tick_webcam_anim)

    def _draw_webcam_anim_frame(self) -> np.ndarray:
        """Render an animated demo-pose skeleton onto a dark canvas."""
        W, H = 960, 540
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        # Subtle radial background glow
        cx, cy, r = W // 2, H // 2, min(W, H) // 2
        for rad in range(r, 0, -6):
            alpha = int(18 * (1 - rad / r))
            color = (0, alpha, alpha // 2)
            cv2.circle(frame, (cx, cy), rad, color, -1)

        # Breathing / pulsing scale driven by sine
        import math
        t = self._anim_phase
        scale = 1.0 + 0.03 * math.sin(t * 0.12)        # slow pulse
        arm_swing = 0.06 * math.sin(t * 0.18)           # arm pendulum

        # Build pixel coords with wave offsets
        pts_px: dict[int, tuple[int, int]] = {}
        for i, (nx, ny) in enumerate(self._DEMO_PTS):
            # Apply arm swing to elbows/wrists
            dx = 0.0
            if i in (7, 9):   dx = -arm_swing
            if i in (8, 10):  dx =  arm_swing

            sx = cx + (nx - 0.5 + dx) * W * 0.55 * scale
            sy = cy + (ny - 0.5)      * H * 0.78 * scale
            pts_px[i] = (int(sx), int(sy))

        # Glow: draw thick dim lines first
        for a, b in self._DEMO_LINKS:
            cv2.line(frame, pts_px[a], pts_px[b], (0, 100, 60), 8, cv2.LINE_AA)
        # Draw bright lines on top
        for a, b in self._DEMO_LINKS:
            cv2.line(frame, pts_px[a], pts_px[b], (0, 220, 120), 2, cv2.LINE_AA)

        # Landmark dots with subtle glow ring
        for (px, py) in pts_px.values():
            cv2.circle(frame, (px, py), 7, (0, 80, 40), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 4, (0, 255, 140), -1, cv2.LINE_AA)

        # Pulsing ring around head (node 0)
        ring_r = int(18 + 4 * math.sin(t * 0.18))
        ring_a = int(120 + 80 * math.sin(t * 0.18))
        cv2.circle(frame, pts_px[0], ring_r, (0, ring_a, ring_a // 2), 1, cv2.LINE_AA)

        # Label
        dot_clr = (0, 200 + int(55 * math.sin(t * 0.15)), 120)
        cv2.putText(frame, "WEBCAM READY  –  Press  Start",
                    (W // 2 - 180, H - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, dot_clr, 1, cv2.LINE_AA)

        # Scanning line
        scan_y = int((H * 0.15) + (H * 0.70) * ((t * 3) % 100) / 100)
        cv2.line(frame, (int(W * 0.15), scan_y), (int(W * 0.85), scan_y),
                 (0, 60, 30), 1, cv2.LINE_AA)

        return frame

    def _set_status(self, msg: str, error: bool = False) -> None:
        self._status_var.set(msg)

    # ══════════════════════════════════════════════════════════════════════
    #  TAB SWITCHING
    # ══════════════════════════════════════════════════════════════════════

    def _switch_tab(self, tab: str) -> None:
        if tab == self._active_tab:
            return
        self._active_tab = tab

        if tab == "video":
            self._tab_video_btn.configure(fg_color=ACCENT, text_color="#000000")
            self._tab_3d_btn.configure(fg_color="#21262D", text_color=TEXT_MUTED)
            self._3d_wrap.grid_remove()
            self._video_wrap.grid()
            self._stop_3d_idle()
            # If not running, restore video idle visuals
            if not self._running and not getattr(self, '_loading_active', False):
                if self._mode_var.get() == "webcam":
                    self._start_webcam_anim()
                elif self._full_path:
                    self._show_video_first_frame(self._full_path)
                else:
                    self._show_placeholder()
        else:  # "3d"
            self._tab_3d_btn.configure(fg_color=ACCENT, text_color="#000000")
            self._tab_video_btn.configure(fg_color="#21262D", text_color=TEXT_MUTED)
            self._video_wrap.grid_remove()
            self._3d_wrap.grid()
            self._stop_webcam_anim()
            # If not running, show idle 3D rotation
            if not self._running and not getattr(self, '_loading_active', False):
                self._start_3d_idle()

    # ══════════════════════════════════════════════════════════════════════
    #  3D CANVAS HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _render_video_on_3d_canvas(self, frame: np.ndarray) -> None:
        cw = self._canvas_3d_video.winfo_width()
        ch = self._canvas_3d_video.winfo_height()
        if cw < 2 or ch < 2:
            return

        h, w = frame.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)

        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))

        self._canvas_3d_video.delete("all")
        self._canvas_3d_video.create_image(
            (cw - nw) // 2, (ch - nh) // 2,
            anchor="nw", image=photo,
            tags="frame"
        )
        self._last_3d_video_ref = photo

    def _update_3d_tab(self, frame: np.ndarray, landmarks_3d: list, tick: int = 0) -> None:
        self._render_3d_on_canvas(landmarks_3d, tick)
        self._render_video_on_3d_canvas(frame)

    def _render_video_on_3d_canvas(self, frame: np.ndarray) -> None:
        cw = self._canvas_3d_video.winfo_width()
        ch = self._canvas_3d_video.winfo_height()
        if cw < 2 or ch < 2:
            return

        h, w = frame.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)

        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))

        self._canvas_3d_video.delete("all")
        self._canvas_3d_video.create_image(
            (cw - nw) // 2, (ch - nh) // 2,
            anchor="nw", image=photo,
        )
        self._last_3d_video_ref = photo

    def _render_3d_on_canvas(self, landmarks_3d: list, tick: int = 0) -> None:
        """Render the 3D model and blit it to the 3D canvas."""
        cw = self._canvas_3d.winfo_width()
        ch = self._canvas_3d.winfo_height()
        if cw < 2 or ch < 2:
            return

        self._renderer.W = cw
        self._renderer.H = ch
        rendered = self._renderer.render(landmarks_3d, tick)

        rgb   = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))

        self._canvas_3d.delete("all")
        self._canvas_3d.create_image(0, 0, anchor="nw", image=photo)
        self._last_3d_ref = photo

    # ── Mouse drag for 3D rotation ──────────────────────────────────────

    def _on_3d_drag_start(self, event) -> None:
        self._drag_start = (event.x, event.y)
        self._drag_yaw0  = self._renderer.yaw
        self._drag_pitch0 = self._renderer.pitch

    def _on_3d_drag_move(self, event) -> None:
        if self._drag_start is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._renderer.yaw   = self._drag_yaw0   + dx * 0.5
        self._renderer.pitch = self._drag_pitch0  - dy * 0.5
        # Clamp pitch
        self._renderer.pitch = max(-80, min(80, self._renderer.pitch))

        # Sync variables
        y = self._renderer.yaw % 360
        if y > 180: y -= 360
        self._yaw_var.set(y)
        self._pitch_var.set(self._renderer.pitch)

    def _on_3d_drag_end(self, event) -> None:
        self._drag_start = None

    def _on_3d_mouse_wheel(self, event) -> None:
        # Standardize delta (windows gives multiples of 120, others might give 1)
        direction = 1 if event.delta > 0 else -1
        self._zoom_var.set(max(0.5, min(3.0, self._zoom_var.get() + direction * 0.1)))
        self._on_3d_slider_change()

    def _on_3d_slider_change(self, *args) -> None:
        self._renderer.yaw = self._yaw_var.get()
        self._renderer.pitch = self._pitch_var.get()
        self._renderer.zoom = self._zoom_var.get()
        
        # If running, the processing loop updates the view automatically.
        # If we are strictly idle (not running, and have recent landmarks), render the latest frame
        if not self._running and self._active_tab == "3d" and self._latest_landmarks_3d:
            self._render_3d_on_canvas(self._latest_landmarks_3d, self._3d_idle_tick)

    def _reset_3d_view(self) -> None:
        self._yaw_var.set(0.0)
        self._pitch_var.set(15.0)
        self._zoom_var.set(1.0)
        self._on_3d_slider_change()

    # ── 3D idle animation (auto-rotating T-pose) ───────────────────────

    def _start_3d_idle(self) -> None:
        self._stop_3d_idle()
        self._3d_idle_tick = 0
        self._tick_3d_idle()

    def _stop_3d_idle(self) -> None:
        if self._3d_idle_job is not None:
            try:
                self.after_cancel(self._3d_idle_job)
            except Exception:
                pass
            self._3d_idle_job = None

    def _tick_3d_idle(self) -> None:
        if self._running:
            return
        if self._active_tab != "3d":
            return
        cw = self._canvas_3d.winfo_width()
        ch = self._canvas_3d.winfo_height()
        if cw < 2 or ch < 2:
            cw, ch = self.CANVAS_W, self.CANVAS_H
        self._renderer.W = cw
        self._renderer.H = ch
        rendered = self._renderer.render_idle(self._3d_idle_tick)

        rgb   = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))

        self._canvas_3d.delete("all")
        self._canvas_3d.create_image(0, 0, anchor="nw", image=photo)
        self._last_3d_ref = photo

        self._3d_idle_tick += 1
        self._3d_idle_job = self.after(45, self._tick_3d_idle)

    def _on_close(self) -> None:
        self._running = False
        self._loading_active = False
        if self._source:
            self._source.release()
        if self._detector:
            self._detector.close()
        if self._renderer:
            self._renderer.close()
        self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _sep(parent, row: int) -> None:
    """Thin horizontal separator."""
    ctk.CTkFrame(parent, height=1, fg_color="#21262D").grid(
        row=row, column=0, sticky="ew", padx=10, pady=8
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = PoseApp()
    app.mainloop()
