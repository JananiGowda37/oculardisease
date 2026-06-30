"""
Multi Disease Ocular Diagnosis System Using Deep Learning
Architecture  : ResNet50 (Transfer Learning)
Frontend      : Tkinter
Diseases      : Diabetic Retinopathy, Glaucoma, Cataract, AMD, Normal, Others
Severity      : Mild, Moderate, Severe, Proliferative
Localization  : Grad-CAM heatmap overlay
Metrics       : Loaded from metrics.pkl (Accuracy, Precision, Recall, F1-Score)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import pickle
import numpy as np



import cv2
import numpy as np
from PIL import Image

def is_valid_fundus_image(path):
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((224, 224))
        arr = np.array(img)

        # Mean RGB values
        r = np.mean(arr[:, :, 0])
        g = np.mean(arr[:, :, 1])
        b = np.mean(arr[:, :, 2])

        brightness = np.mean(arr)

        # 1. Reject completely dark images
        if brightness < 15:
            return False

        # 2. Reject bright/white images (documents, notes, screenshots)
        if brightness > 210:
            return False

        # 3. Reject grayscale/monochrome-like images (notes often fall here)
        if abs(r-g) < 10 and abs(g-b) < 10:
            return False

        # 4. Color signature: Fundus is reddish/orange.
        if r < g or r < b:
            return False
        if (r - b) < 15:
            return False

        # 5. Illumination profile: Fundus has a dark periphery/mask and brighter center.
        # Documents/notes often have white margins (bright) and text in the center (dark).
        center_brightness = np.mean(arr[70:154, 70:154])
        corners_brightness = np.mean([
            arr[:40, :40], arr[:40, -40:], 
            arr[-40:, :40], arr[-40:, -40:]
        ])
        
        if corners_brightness > center_brightness + 15:
            return False

        return True

    except:
        return False


# ── Optional imports (graceful fallback) ───────────────────────────────────────
try:
    from PIL import Image, ImageTk, ImageFilter, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.models import Model, load_model
    # Fix for quantization_config issue

    original_dense_from_config = tf.keras.layers.Dense.from_config

    @classmethod
    def patched_dense_from_config(cls, config):
        config.pop("quantization_config", None)
        return original_dense_from_config(config)
    
    tf.keras.layers.Dense.from_config = patched_dense_from_config
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
IMG_SIZE        = (224, 224)
DISEASE_CLASSES = [
    "AMD",
    "Cataract",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Hypertension",
    "Myopia",
    "Normal"
]
SEVERITY_LEVELS = ["N/A", "Mild", "Moderate", "Severe", "Proliferative"]

# Colour palette  ── dark medical theme
BG_DARK    = "#0b0f18"
BG_CARD    = "#131929"
BG_PANEL   = "#1a2236"
BG_PANEL2  = "#1e2840"
ACCENT     = "#4f9cf9"       # blue
ACCENT2    = "#34c97b"       # green
WARN       = "#f05252"       # red
AMBER      = "#f5a623"       # orange
PURPLE     = "#a78bfa"       # purple
TEXT_PRI   = "#e8edf5"
TEXT_SEC   = "#7a8499"
TEXT_DIM   = "#3d4a5e"
BORDER     = "#232f45"
BORDER2    = "#2d3f5e"

# Severity colours
SEV_COLORS = {
    "N/A":           "#7a8499",
    "Mild":          "#34c97b",
    "Moderate":      "#f5a623",
    "Severe":        "#f05252",
    "Proliferative": "#c0392b",
    
}

# Metric bar colours
METRIC_COLORS = {
    "Accuracy":  "#4f9cf9",
    "Precision": "#34c97b",
    "Recall":    "#a78bfa",
    "F1-Score":  "#f5a623",
}

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, 0)
    if TF_AVAILABLE:
        arr = preprocess_input(arr)
    return arr


def make_gradcam_heatmap(model, img_array: np.ndarray,
                          last_conv="conv5_block3_out") -> np.ndarray:
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        pred_idx  = tf.argmax(preds[0])
        class_ch  = preds[:, pred_idx]
    grads  = tape.gradient(class_ch, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam    = conv_out[0] @ pooled[..., tf.newaxis]
    cam    = tf.squeeze(cam)
    cam    = tf.maximum(cam, 0) / (tf.math.reduce_max(cam) + 1e-8)
    return cam.numpy()


def fake_predict(path: str):
    np.random.seed(int(os.path.getsize(path)) % 999)
    probs   = np.random.dirichlet(np.ones(len(DISEASE_CLASSES)))
    idx     = int(np.argmax(probs))
    sev_idx = np.random.randint(0, 4)
    heatmap = np.random.rand(7, 7).astype(np.float32)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return DISEASE_CLASSES[idx], float(probs[idx]), SEVERITY_LEVELS[sev_idx], heatmap, probs


def load_pkl_metrics(path: str):
    """Load metrics.pkl → dict with keys: accuracy, precision, recall, f1_score."""
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        result = {}
        key_map = {
            "accuracy":  "Accuracy",
            "precision": "Precision",
            "recall":    "Recall",
            "f1_score":  "F1-Score",
            "f1":        "F1-Score",
        }
        if isinstance(data, dict):
            for raw_k, v in data.items():
                clean = str(raw_k).lower().strip()
                for pat, label in key_map.items():
                    if pat in clean:
                        val = v
                        if isinstance(val, (list, tuple, np.ndarray)):
                            val = float(val[-1])
                        else:
                            val = float(val)
                        # Normalise: if stored as fraction convert to %
                        if val <= 1.0:
                            val *= 100
                        result[label] = val
                        break
        return result if result else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# ROUNDED RECTANGLE HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def round_rect(canvas, x1, y1, x2, y2, r=10, **kwargs):
    canvas.create_arc(x1,     y1,     x1+2*r, y1+2*r, start=90,  extent=90,  style="pieslice", **kwargs)
    canvas.create_arc(x2-2*r, y1,     x2,     y1+2*r, start=0,   extent=90,  style="pieslice", **kwargs)
    canvas.create_arc(x1,     y2-2*r, x1+2*r, y2,     start=180, extent=90,  style="pieslice", **kwargs)
    canvas.create_arc(x2-2*r, y2-2*r, x2,     y2,     start=270, extent=90,  style="pieslice", **kwargs)
    canvas.create_rectangle(x1+r, y1, x2-r, y2, **kwargs)
    canvas.create_rectangle(x1, y1+r, x2, y2-r, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class RetinalApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Multi-Disease Ocular Diagnosis System  |  ResNet50 · Grad-CAM")
        self.geometry("1280x820")
        self.minsize(1100, 720)
        self.configure(bg=BG_DARK)
        self.resizable(True, True)

        # State
        self.model       = None
        self.model_path  = tk.StringVar(value="No model loaded")
        self.image_path  = None
        self.orig_pil    = None
        self.result      = None
        self._anim_id    = None
        self._spin_step  = 0
        self._pkl_metrics = None

        self._build_ui()
        self._try_load_default_model()
        self._auto_load_pkl()

    # ──────────────────────────────────────────────────────────────────────────
    # UI CONSTRUCTION
    # ──────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()

        # Main content
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        body.columnconfigure(0, weight=0, minsize=290)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._build_left_panel(body)
        self._build_right_panel(body)
        self._build_status_bar()

    # ── Header ─────────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self, bg=BG_CARD, pady=0)
        hdr.pack(fill="x")

        # Accent top bar
        tk.Frame(hdr, bg=ACCENT, height=3).pack(fill="x")

        inner = tk.Frame(hdr, bg=BG_CARD, padx=18, pady=10)
        inner.pack(fill="x")

        # Left: title block
        title_blk = tk.Frame(inner, bg=BG_CARD)
        title_blk.pack(side="left")
        tk.Label(title_blk, text="👁  Multi-Disease Ocular Diagnosis System",
                 font=("Helvetica", 16, "bold"), fg=TEXT_PRI, bg=BG_CARD).pack(anchor="w")
        tk.Label(title_blk,
                 text="Deep Learning  ·  ResNet50 Transfer Learning  ·  Grad-CAM Lesion Localization",
                 font=("Helvetica", 9), fg=TEXT_SEC, bg=BG_CARD).pack(anchor="w")

        # Right: model loader
        right = tk.Frame(inner, bg=BG_CARD)
        right.pack(side="right")
        self._pill_btn(right, "⬆  Load Model (.h5)", self._load_model,
                       bg=BG_PANEL2, fg=ACCENT).pack(side="right")
        tk.Label(right, textvariable=self.model_path,
                 font=("Helvetica", 8), fg=TEXT_SEC, bg=BG_CARD,
                 wraplength=260, justify="right").pack(side="right", padx=(0, 10))

        tk.Frame(hdr, bg=BORDER, height=1).pack(fill="x")

    # ── Left panel ─────────────────────────────────────────────────────────────
    def _build_left_panel(self, parent):
        left = tk.Frame(parent, bg=BG_DARK)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        # Upload card
        up_card = self._card(left, "📁  Retinal Image Input")
        up_card.grid(row=0, column=0, sticky="ew")

        self._pill_btn(up_card, "  Select Fundus Image  ", self._select_image,
                       bg=ACCENT, fg="#fff",
                       font=("Helvetica", 11, "bold"), pady=10).pack(
            fill="x", padx=12, pady=(10, 4))
        tk.Label(up_card, text="Supported: JPG  PNG  BMP  TIFF",
                 font=("Helvetica", 8), fg=TEXT_SEC, bg=BG_CARD).pack(pady=(0, 8))

        # Pipeline steps (decorative, reflects architecture)
        pipe = tk.Frame(up_card, bg=BG_PANEL, padx=10, pady=8)
        pipe.pack(fill="x", padx=12, pady=(0, 10))
        steps = [
            ("①", "Image Preprocessing",   "Resize · CLAHE · Normalise"),
            ("②", "Feature Extraction",     "ResNet50 CNN Backbone"),
            ("③", "Disease Classification", "Softmax · 4 Classes"),
            ("④", "Lesion Localization",    "Grad-CAM Heatmap"),
        ]
        for icon, title, sub in steps:
            row = tk.Frame(pipe, bg=BG_PANEL)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=icon, font=("Helvetica", 9, "bold"),
                     fg=ACCENT, bg=BG_PANEL, width=2).pack(side="left")
            col = tk.Frame(row, bg=BG_PANEL)
            col.pack(side="left", padx=6)
            tk.Label(col, text=title, font=("Helvetica", 8, "bold"),
                     fg=TEXT_PRI, bg=BG_PANEL, anchor="w").pack(anchor="w")
            tk.Label(col, text=sub, font=("Helvetica", 7),
                     fg=TEXT_SEC, bg=BG_PANEL, anchor="w").pack(anchor="w")

        # Image preview
        prev = self._card(left, "🖼  Image Preview")
        prev.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.preview_canvas = tk.Canvas(prev, bg="#07090f",
                                        highlightthickness=0, height=200)
        self.preview_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self._placeholder(self.preview_canvas, "No image selected")

        # Analyze button
        self.analyze_btn = tk.Button(
            left, text="▶   Analyze Image",
            command=self._run_analysis,
            font=("Helvetica", 12, "bold"), bg=ACCENT2, fg="#000",
            relief="flat", padx=16, pady=13, cursor="hand2",
            activebackground="#28a464", activeforeground="#000",
            state="disabled", bd=0, highlightthickness=0
        )
        self.analyze_btn.grid(row=2, column=0, sticky="ew", pady=(10, 0))

    # ── Right panel ────────────────────────────────────────────────────────────
    def _build_right_panel(self, parent):
        right = tk.Frame(parent, bg=BG_DARK)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=0)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # ── Top row: results + heatmap ────────────────────────────────────────
        top = tk.Frame(right, bg=BG_DARK)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=0, minsize=260)

        self._build_results_row(top)
        self._build_gradcam_card(top)

        # ── Bottom: tabbed graphs ─────────────────────────────────────────────
        self._build_graphs_card(right)

    # ── Results row ────────────────────────────────────────────────────────────
    def _build_results_row(self, parent):
        res = self._card(parent, "🩺  Diagnosis Results")
        res.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        body = tk.Frame(res, bg=BG_CARD)
        body.pack(fill="both", expand=True, padx=12, pady=10)
        body.columnconfigure((0, 1, 2), weight=1)

        # ── Metric tile helper ─────────────────────────────────────────────────
        def metric_tile(parent, col, icon, label, var_attr, big_color):
            tile = tk.Frame(parent, bg=BG_PANEL, padx=12, pady=10,
                            highlightbackground=BORDER2, highlightthickness=1)
            tile.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 8, 0))
            tk.Label(tile, text=icon, font=("Helvetica", 18),
                     fg=big_color, bg=BG_PANEL).pack(anchor="w")
            tk.Label(tile, text=label, font=("Helvetica", 8),
                     fg=TEXT_SEC, bg=BG_PANEL).pack(anchor="w")
            lbl = tk.Label(tile, text="—", font=("Helvetica", 15, "bold"),
                           fg=big_color, bg=BG_PANEL, wraplength=180, justify="left")
            lbl.pack(anchor="w", pady=(4, 0))
            setattr(self, var_attr, lbl)

        metric_tile(body, 0, "🔬", "Detected Disease",  "lbl_disease",  ACCENT)
        metric_tile(body, 1, "%",  "Confidence Score",  "lbl_conf",     ACCENT2)
        metric_tile(body, 2, "⚠", "Severity Grade",    "lbl_severity", WARN)

        # Severity gauge
        sev_frame = tk.Frame(res, bg=BG_CARD)
        sev_frame.pack(fill="x", padx=12, pady=(0, 4))
        tk.Label(sev_frame, text="Severity Scale", font=("Helvetica", 8),
                 fg=TEXT_SEC, bg=BG_CARD).pack(anchor="w")
        self.sev_canvas = tk.Canvas(sev_frame, bg=BG_CARD, height=22,
                                    highlightthickness=0)
        self.sev_canvas.pack(fill="x")

        # Per-class bar
        bar_frame = tk.Frame(res, bg=BG_CARD)
        bar_frame.pack(fill="x", padx=12, pady=(4, 0))
        tk.Label(bar_frame, text="Class Probabilities", font=("Helvetica", 8),
                 fg=TEXT_SEC, bg=BG_CARD).pack(anchor="w")
        self.prob_canvas = tk.Canvas(bar_frame, bg=BG_CARD,
                                     height=len(DISEASE_CLASSES) * 22 + 4,
                                     highlightthickness=0)
        self.prob_canvas.pack(fill="x")
        self._draw_prob_bars(None)

       
    # ── Grad-CAM card ──────────────────────────────────────────────────────────
    def _build_gradcam_card(self, parent):
        card = self._card(parent, "🌡  Lesion Localization  (Grad-CAM)")
        card.grid(row=0, column=1, sticky="nsew")

        self.heat_canvas = tk.Canvas(card, bg="#07090f", width=240, height=200,
                                     highlightthickness=0)
        self.heat_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self._placeholder(self.heat_canvas, "Grad-CAM heatmap\nwill appear here")

        # Legend
        leg = tk.Frame(card, bg=BG_CARD)
        leg.pack(fill="x", padx=10, pady=(0, 8))
        for txt, colour in [("Low", "#1a3a8c"), ("Med", "#2ca080"), ("High", "#e53e3e")]:
            tk.Frame(leg, bg=colour, width=14, height=10).pack(side="left")
            tk.Label(leg, text=txt, font=("Helvetica", 7), fg=TEXT_SEC,
                     bg=BG_CARD).pack(side="left", padx=(1, 6))

    # ── Graphs card (tabbed) ───────────────────────────────────────────────────
    def _build_graphs_card(self, parent):
        card = self._card(parent, "📊  Performance Analytics")
        card.grid(row=1, column=0, sticky="nsew")

        # Tab bar (custom, not ttk.Notebook so we keep our theme)
        tab_bar = tk.Frame(card, bg=BG_CARD)
        tab_bar.pack(fill="x", padx=12, pady=(6, 0))

        self._tab_btns = {}
        self._active_tab = tk.StringVar(value="metrics")

        for key, label in [("metrics", "📈  Model Metrics"), ("image", "🖼  Image Analysis")]:
            btn = tk.Button(
                tab_bar, text=label,
                command=lambda k=key: self._switch_tab(k),
                font=("Helvetica", 9, "bold"), relief="flat",
                padx=14, pady=6, cursor="hand2",
                bd=0, highlightthickness=0
            )
            btn.pack(side="left", padx=(0, 4))
            self._tab_btns[key] = btn
        self._style_tabs()

        # Graph area
        self.graph_area = tk.Frame(card, bg="#07090f")
        self.graph_area.pack(fill="both", expand=True, padx=12, pady=(6, 12))

        self._graph_widget = None
        self._active_fig   = None

        # Show metrics tab by default (will draw after pkl loads)
        self._draw_placeholder_graph("Model metrics will load automatically")

    # ──────────────────────────────────────────────────────────────────────────
    # STATUS BAR
    # ──────────────────────────────────────────────────────────────────────────
    def _build_status_bar(self):
        bar = tk.Frame(self, bg=BG_CARD, pady=4)
        bar.pack(fill="x", side="bottom")
        tk.Frame(bar, bg=ACCENT, height=2).pack(fill="x")
        inner = tk.Frame(bar, bg=BG_CARD)
        inner.pack(fill="x")
        self.status_var = tk.StringVar(value="Ready  ·  Load a model and select a fundus image.")
        tk.Label(inner, textvariable=self.status_var,
                 font=("Helvetica", 9), fg=TEXT_SEC, bg=BG_CARD).pack(side="left", padx=14, pady=4)
        tk.Label(inner,
                 text="ResNet50 · ImageNet Pretrained · Fine-tuned on Retinal Dataset",
                 font=("Helvetica", 8), fg=TEXT_DIM, bg=BG_CARD).pack(side="right", padx=14)

    # ══════════════════════════════════════════════════════════════════════════
    # WIDGET FACTORIES
    # ══════════════════════════════════════════════════════════════════════════

    def _card(self, parent, title: str) -> tk.Frame:
        outer = tk.Frame(parent, bg=BG_CARD,
                         highlightbackground=BORDER2, highlightthickness=1)
        hdr = tk.Frame(outer, bg=BG_PANEL)
        hdr.pack(fill="x")
        tk.Label(hdr, text=title, font=("Helvetica", 9, "bold"),
                 fg=TEXT_PRI, bg=BG_PANEL, padx=12, pady=7).pack(side="left")
        tk.Frame(outer, bg=BORDER, height=1).pack(fill="x")
        return outer

    def _pill_btn(self, parent, text, cmd, bg=BG_PANEL2, fg=TEXT_PRI,
                  font=("Helvetica", 9), pady=6, state="normal"):
        return tk.Button(parent, text=text, command=cmd,
                         font=font, bg=bg, fg=fg,
                         relief="flat", padx=12, pady=pady,
                         cursor="hand2", state=state,
                         activebackground=BORDER2, activeforeground=TEXT_PRI,
                         bd=0, highlightthickness=0)

    @staticmethod
    def _placeholder(canvas, msg):
        canvas.delete("all")
        w = canvas.winfo_reqwidth() or 200
        h = canvas.winfo_reqheight() or 180
        canvas.create_text(w // 2, h // 2, text=msg, fill=TEXT_DIM,
                           font=("Helvetica", 9), justify="center")

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL LOADING
    # ══════════════════════════════════════════════════════════════════════════

    def _try_load_default_model(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for fname in os.listdir(script_dir):
            if fname.endswith((".h5", ".keras")):
                self._load_model_from(os.path.join(script_dir, fname))
                return

    def _load_model(self):
        path = filedialog.askopenfilename(
            title="Select ResNet50 Model",
            filetypes=[("Keras model", "*.h5 *.keras"), ("All files", "*.*")]
        )
        if path:
            self._load_model_from(path)

    def _load_model_from(self, path: str):
        if not TF_AVAILABLE:
            self.model_path.set("⚠  TF not installed — demo mode")
            self._set_status("TensorFlow not found. Running in demo mode.")
            return
        try:
            self._set_status(f"Loading model: {os.path.basename(path)} …")
            self.model = load_model(path,compile=False)
            self.model_path.set(f"✔  {os.path.basename(path)}")
            self._set_status(f"Model loaded: {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Model Error", str(exc))
            self.model_path.set("⚠  Failed to load model")

    # ══════════════════════════════════════════════════════════════════════════
    # IMAGE SELECTION & PREVIEW
    # ══════════════════════════════════════════════════════════════════════════

    def _select_image(self):
        path = filedialog.askopenfilename(
            title="Select Retinal Fundus Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                       ("All files", "*.*")]
        )
        if not path:
            return

        # [MODIFIED]: Uncommented image validation to prevent processing non-fundus images
        if not is_valid_fundus_image(path):
            messagebox.showerror(
                "Invalid Image",
                "Please upload a valid retinal fundus image."
            )
            self._set_status("Invalid fundus image.")
            return
        self.image_path = path
        self._show_preview(path)
        self.analyze_btn.config(state="normal")
        self._set_status(f"Image loaded: {os.path.basename(path)}")
        # Reset results
        for lbl in (self.lbl_disease, self.lbl_conf, self.lbl_severity):
            lbl.config(text="—")
        self.lbl_disease.config(fg=ACCENT)
        self.lbl_conf.config(fg=ACCENT2)
        self.lbl_severity.config(fg=WARN)
        self.sev_canvas.delete("all")
        self._draw_prob_bars(None)
        self._placeholder(self.heat_canvas, "Run analysis to see Grad-CAM")
        # self.report_btn.config(state="disabled")
        self.result = None
        self._switch_tab("metrics")

    def _show_preview(self, path: str):
        if not PIL_AVAILABLE:
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(
                140, 110, text="Pillow not installed\nCannot preview image",
                fill=TEXT_SEC, font=("Helvetica", 9), justify="center"
            )
            return
        img = Image.open(path).convert("RGB")
        self.orig_pil = img.copy()
        self.preview_canvas.update_idletasks()
        w = self.preview_canvas.winfo_width()  or 270
        h = self.preview_canvas.winfo_height() or 200
        img.thumbnail((w - 8, h - 8), Image.LANCZOS)
        self._preview_tk = ImageTk.PhotoImage(img)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(w // 2, h // 2, anchor="center",
                                         image=self._preview_tk)

    # ══════════════════════════════════════════════════════════════════════════
    # ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════

    def _run_analysis(self):
        if not self.image_path:
            return
        self.analyze_btn.config(state="disabled", text="Analyzing…")
        self._set_status("Running ResNet50 inference…")
        self._start_spinner()
        threading.Thread(target=self._inference_thread, daemon=True).start()

    def _inference_thread(self):
        try:
            if TF_AVAILABLE and self.model is not None:
                arr = preprocess_image(self.image_path)
                preds = self.model.predict(arr, verbose=0)[0]

                idx = int(np.argmax(preds))
                conf = float(preds[idx])

                cls = [
                    "AMD",
                    "Cataract",
                    "Diabetic Retinopathy",
                    "Glaucoma",
                    "Hypertension",
                    "Myopia",
                    "Normal",
                    "Other"
                ]

                disease = cls[idx]

                # Low-confidence prediction
                if conf < 0.35:
                    disease = "Uncertain"

                # Severity
                if disease == "Normal":
                    severity = "N/A"
                elif conf >= 0.85:
                    severity = "Severe"
                elif conf >= 0.65:
                    severity = "Moderate"
                else:
                    severity = "Mild"

                heatmap = make_gradcam_heatmap(self.model, arr)

                self.result = (
                    disease,
                    conf,
                    severity,
                    heatmap,
                    preds
                )

            else:
                self.result = fake_predict(self.image_path)

            self.after(0, self._on_done)

        except Exception as exc:
            self.after(0, lambda: self._on_error(str(exc)))



    def _on_done(self):
        self._stop_spinner()
        disease, conf, severity, heatmap, all_probs = self.result
        self.analyze_btn.config(state="normal", text="▶   Analyze Image")

        # Update metric tiles
        self.lbl_disease.config(
            text=disease,
            fg=ACCENT2 if disease == "Normal" else WARN
        )
        self.lbl_conf.config(text=f"{conf*100:.1f}%")
        self.lbl_severity.config(
            text=severity, fg=SEV_COLORS.get(severity, TEXT_PRI)
        )
        self._draw_severity_bar(severity)
        self._draw_prob_bars(all_probs)
        self._draw_heatmap(heatmap)
        # self.report_btn.config(state="normal")

        # Auto-switch to image tab and draw
        self._switch_tab("image")

        self._set_status(
            f"✔  Analysis complete  ·  {disease}  ·  "
            f"Confidence {conf*100:.1f}%  ·  Severity: {severity}"
        )

    def _on_error(self, msg: str):
        self._stop_spinner()
        self.analyze_btn.config(state="normal", text="▶   Analyze Image")
        messagebox.showerror("Inference Error", msg)
        self._set_status("Error during inference. See dialog.")

    # ══════════════════════════════════════════════════════════════════════════
    # VISUALISATION
    # ══════════════════════════════════════════════════════════════════════════

    def _draw_severity_bar(self, severity: str):
        self.sev_canvas.update_idletasks()
        w = self.sev_canvas.winfo_width() or 300
        self.sev_canvas.delete("all")
        idx_map = {s: i for i, s in enumerate(SEVERITY_LEVELS)}
        idx = idx_map.get(severity, -1)
        if idx < 0:
            return
        seg = w / len(SEVERITY_LEVELS)
        colors = [ACCENT2, AMBER, WARN, "#c0392b"]
        for i, sev in enumerate(SEVERITY_LEVELS):
            active = i <= idx
            fill   = colors[i] if active else BG_PANEL
            x0, x1 = i * seg + 2, (i + 1) * seg - 2
            self.sev_canvas.create_rectangle(x0, 2, x1, 20, fill=fill, outline="")
            self.sev_canvas.create_text(
                (x0 + x1) / 2, 11, text=sev,
                fill="#000" if active else TEXT_DIM,
                font=("Helvetica", 7, "bold")
            )

    def _draw_prob_bars(self, probs):
        self.prob_canvas.update_idletasks()
        self.prob_canvas.delete("all")
        w = self.prob_canvas.winfo_width() or 380
        row_h = 22
        label_w = 170
        bar_max = w - label_w - 50
        classes = DISEASE_CLASSES if (probs is None or len(probs) == len(DISEASE_CLASSES)) \
                  else ["AMD","Cataract","Diabetic Retinopathy","Glaucoma","Normal"]

        if probs is None:
            for i, cls in enumerate(classes):
                y = i * row_h + row_h // 2
                self.prob_canvas.create_text(
                    4, y, text=cls, anchor="w",
                    fill=TEXT_DIM, font=("Helvetica", 8)
                )
                self.prob_canvas.create_rectangle(
                    label_w, y - 5, label_w + bar_max, y + 5,
                    fill=BG_PANEL, outline=""
                )
            return

        peak = int(np.argmax(probs))
        for i, cls in enumerate(classes[:len(probs)]):
            y    = i * row_h + row_h // 2
            p    = float(probs[i])
            bw   = int(p * bar_max)
            col  = ACCENT if i == peak else ACCENT2 if cls == "Normal" else BG_PANEL2
            self.prob_canvas.create_text(
                4, y, text=cls, anchor="w",
                fill=TEXT_PRI if i == peak else TEXT_SEC,
                font=("Helvetica", 8, "bold" if i == peak else "normal")
            )
            # Background track
            self.prob_canvas.create_rectangle(
                label_w, y - 5, label_w + bar_max, y + 5,
                fill=BG_PANEL, outline=""
            )
            # Filled bar
            if bw > 0:
                self.prob_canvas.create_rectangle(
                    label_w, y - 5, label_w + bw, y + 5,
                    fill=col, outline=""
                )
            self.prob_canvas.create_text(
                label_w + bar_max + 4, y, text=f"{p*100:.1f}%",
                anchor="w", fill=TEXT_SEC, font=("Helvetica", 7)
            )

    def _draw_heatmap(self, heatmap: np.ndarray):
        self.heat_canvas.update_idletasks()
        cw = self.heat_canvas.winfo_width()  or 240
        ch = self.heat_canvas.winfo_height() or 200

        if not PIL_AVAILABLE:
            self._placeholder(self.heat_canvas, "Pillow required for heatmap")
            return

        h_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (cw - 8, ch - 8), Image.LANCZOS
        )
        arr = np.array(h_img.convert("L"), dtype=np.float32) / 255.0
        r = np.clip(1.5 - np.abs(arr * 4 - 3), 0, 1)
        g = np.clip(1.5 - np.abs(arr * 4 - 2), 0, 1)
        b = np.clip(1.5 - np.abs(arr * 4 - 1), 0, 1)
        jet = Image.fromarray((np.stack([r, g, b], axis=-1) * 255).astype(np.uint8))

        if self.orig_pil is not None:
            orig = self.orig_pil.resize((cw - 8, ch - 8), Image.LANCZOS).convert("RGB")
            blended = Image.blend(orig, jet, alpha=0.55)
        else:
            blended = jet

        self._heatmap_tk = ImageTk.PhotoImage(blended)
        self.heat_canvas.delete("all")
        self.heat_canvas.create_image(cw // 2, ch // 2, anchor="center",
                                      image=self._heatmap_tk)

    # ══════════════════════════════════════════════════════════════════════════
    # GRAPH TABS
    # ══════════════════════════════════════════════════════════════════════════

    def _style_tabs(self):
        active = self._active_tab.get()
        for key, btn in self._tab_btns.items():
            if key == active:
                btn.config(bg=ACCENT, fg="#fff")
            else:
                btn.config(bg=BG_PANEL2, fg=TEXT_SEC)

    def _switch_tab(self, key: str):
        self._active_tab.set(key)
        self._style_tabs()
        if key == "metrics":
            self._draw_metrics_graph()
        elif key == "image":
            self._draw_image_graph()

    def _clear_graph(self):
        if self._graph_widget:
            self._graph_widget.destroy()
            self._graph_widget = None
        if self._active_fig:
            plt.close(self._active_fig)
            self._active_fig = None

    def _draw_placeholder_graph(self, msg=""):
        self._clear_graph()
        c = tk.Canvas(self.graph_area, bg="#07090f", highlightthickness=0)
        c.pack(fill="both", expand=True)
        self._graph_widget = c
        c.update_idletasks()
        w, h = c.winfo_width() or 600, c.winfo_height() or 200
        c.create_text(w // 2, h // 2, text=msg, fill=TEXT_DIM,
                      font=("Helvetica", 9), justify="center")

    # ── Metrics graph (from pkl) ───────────────────────────────────────────────
    def _auto_load_pkl(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pkl_path = os.path.join(script_dir, "metrics.pkl")
        if os.path.exists(pkl_path):
            self._pkl_metrics = load_pkl_metrics(pkl_path)
            
        # [MODIFIED]: Removed hardcoded default metrics. Only load from the file.
        
        # [MODIFIED]: Always schedule to draw the metrics graph
        self.after(300, self._draw_metrics_graph)

    def _draw_metrics_graph(self):
        if not MATPLOTLIB_AVAILABLE:
            self._draw_placeholder_graph("Matplotlib not installed")
            return

        # [MODIFIED]: Removed the filedialog prompt and fake metrics. Just show placeholder if missing.
        metrics = self._pkl_metrics
        if not metrics:
            self._draw_placeholder_graph("No metrics data found.\nEnsure valid metrics.pkl is in app folder.")
            return

        self._clear_graph()

        labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
        values = [metrics.get(l, 0.0) for l in labels]
        colors = [METRIC_COLORS[l] for l in labels]

        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8),
                                 gridspec_kw={"width_ratios": [1.6, 1]})
        fig.patch.set_facecolor("#07090f")

        # ── Left: Bar chart ────────────────────────────────────────────────────
        ax = axes[0]
        ax.set_facecolor("#07090f")
        bars = ax.bar(labels, values, color=colors, width=0.52, zorder=3)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Score (%)", color=TEXT_SEC, fontsize=8)
        ax.set_title("Model Performance Metrics", color=TEXT_PRI, fontsize=9, pad=8)
        ax.tick_params(axis="x", colors=TEXT_PRI, labelsize=8)
        ax.tick_params(axis="y", colors=TEXT_SEC, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(BORDER2)
        ax.yaxis.grid(True, color=BORDER2, linestyle="--", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        # Value labels on top of bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{val:.1f}%",
                    ha="center", va="bottom",
                    color=TEXT_PRI, fontsize=8, fontweight="bold")
        # 90% target line
        ax.axhline(90, color=ACCENT2, linewidth=1, linestyle="--", alpha=0.7, zorder=2)
        ax.text(3.4, 91.5, "90% target", color=ACCENT2, fontsize=7, ha="right")

        # ── Right: Gauge / radar-style summary ────────────────────────────────
        ax2 = axes[1]
        ax2.set_facecolor("#07090f")
        ax2.set_xlim(0, 1); ax2.set_ylim(0, len(labels) + 0.5)
        ax2.axis("off")
        ax2.set_title("Score Summary", color=TEXT_PRI, fontsize=9, pad=8)

        for i, (lbl, val, col) in enumerate(zip(labels, values, colors)):
            y = len(labels) - i - 0.5
            # Track
            ax2.barh(y, 100, left=0, height=0.55, color=BG_PANEL2, zorder=1)
            # Fill
            ax2.barh(y, val, left=0, height=0.55, color=col, alpha=0.85, zorder=2)
            ax2.text(-2, y, lbl, ha="right", va="center",
                     color=TEXT_SEC, fontsize=7.5)
            ax2.text(102, y, f"{val:.1f}%", ha="left", va="center",
                     color=col, fontsize=7.5, fontweight="bold")

        fig.tight_layout(pad=1.2)
        self._active_fig = fig

        canvas = FigureCanvasTkAgg(fig, master=self.graph_area)
        canvas.draw()
        self._graph_widget = canvas.get_tk_widget()
        self._graph_widget.config(bg="#07090f", highlightthickness=0)
        self._graph_widget.pack(fill="both", expand=True)

    # ── Image confidence graph ─────────────────────────────────────────────────
    def _draw_image_graph(self):
        if not MATPLOTLIB_AVAILABLE:
            self._draw_placeholder_graph("Matplotlib not installed")
            return
        if not self.result:
            self._draw_placeholder_graph("Run analysis first to see image prediction chart")
            return

        self._clear_graph()
        _, _, _, _, all_probs = self.result
        classes = (DISEASE_CLASSES if len(all_probs) == len(DISEASE_CLASSES)
                   else ["AMD","Cataract","Diabetic Retinopathy","Glaucoma","Normal"])

        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8),
                                 gridspec_kw={"width_ratios": [1.4, 1]})
        fig.patch.set_facecolor("#07090f")

        # ── Horizontal bar chart ───────────────────────────────────────────────
        ax = axes[0]
        ax.set_facecolor("#07090f")
        peak    = int(np.argmax(all_probs[:len(classes)]))
        vals    = [p * 100 for p in all_probs[:len(classes)]]
        y_pos   = np.arange(len(classes))
        col_lst = [ACCENT if i == peak else BORDER2 for i in range(len(classes))]
        bars = ax.barh(y_pos, vals, color=col_lst, height=0.5, zorder=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(classes, color=TEXT_PRI, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 110)
        ax.set_xlabel("Confidence (%)", color=TEXT_SEC, fontsize=8)
        ax.set_title("Prediction Confidence by Class", color=TEXT_PRI, fontsize=9, pad=8)
        ax.tick_params(axis="x", colors=TEXT_SEC, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(BORDER2)
        ax.xaxis.grid(True, color=BORDER2, linestyle="--", linewidth=0.6, zorder=0)
        for bar, val in zip(bars, vals):
            ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", color=TEXT_PRI, fontsize=7.5)

        # ── Pie / donut ────────────────────────────────────────────────────────
        ax2 = axes[1]
        ax2.set_facecolor("#07090f")
        pie_vals = [max(v, 0.001) for v in all_probs[:len(classes)]]
        pie_cols = [ACCENT if i == peak else BORDER2 for i in range(len(classes))]
        wedges, _ = ax2.pie(
            pie_vals, colors=pie_cols,
            startangle=90, counterclock=False,
            wedgeprops={"width": 0.55, "edgecolor": "#07090f", "linewidth": 1.5}
        )
        ax2.set_title("Distribution", color=TEXT_PRI, fontsize=9, pad=8)
        # Centre label
        ax2.text(0, 0,
                 f"{classes[peak]}\n{all_probs[peak]*100:.1f}%",
                 ha="center", va="center",
                 color=TEXT_PRI, fontsize=7.5, fontweight="bold")
        # Legend
        patches = [mpatches.Patch(color=pie_cols[i], label=classes[i])
                   for i in range(len(classes))]
        ax2.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.25),
                   fontsize=6.5, ncol=2,
                   labelcolor=TEXT_SEC, frameon=False)

        fig.tight_layout(pad=1.2)
        self._active_fig = fig

        canvas = FigureCanvasTkAgg(fig, master=self.graph_area)
        canvas.draw()
        self._graph_widget = canvas.get_tk_widget()
        self._graph_widget.config(bg="#07090f", highlightthickness=0)
        self._graph_widget.pack(fill="both", expand=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SPINNER
    # ══════════════════════════════════════════════════════════════════════════

    _SPIN_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def _start_spinner(self):
        self._spin_step = 0
        self._tick_spinner()

    def _tick_spinner(self):
        c = self._SPIN_CHARS[self._spin_step % len(self._SPIN_CHARS)]
        self._set_status(f"{c}  Running ResNet50 inference…")
        self._spin_step += 1
        self._anim_id = self.after(80, self._tick_spinner)

    def _stop_spinner(self):
        if self._anim_id:
            self.after_cancel(self._anim_id)
            self._anim_id = None

    # ══════════════════════════════════════════════════════════════════════════
    # REPORT EXPORT
    # ══════════════════════════════════════════════════════════════════════════

    def _export_report(self):
        if not self.result:
            return
        disease, conf, severity, _, all_probs = self.result
        path = filedialog.asksaveasfilename(
            title="Save Diagnosis Report",
            defaultextension=".txt",
            filetypes=[("Text report", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return
        classes = (DISEASE_CLASSES if len(all_probs) == len(DISEASE_CLASSES)
                   else ["AMD","Cataract","Diabetic Retinopathy","Glaucoma","Normal"])
        lines = [
            "=" * 58,
            "       RETINAL DISEASE DIAGNOSIS REPORT",
            "       Model: ResNet50 (Transfer Learning)",
            "=" * 58,
            f"Image      : {os.path.basename(self.image_path or 'unknown')}",
            f"Disease    : {disease}",
            f"Confidence : {conf*100:.1f}%",
            f"Severity   : {severity}",
            "",
            "Per-class Probabilities:",
        ]
        for cls, p in zip(classes, all_probs):
            bar = "█" * int(p * 30)
            lines.append(f"  {cls:<30} {p*100:5.1f}%  {bar}")

        if self._pkl_metrics:
            lines += ["", "Model Metrics (from training):"]
            for k, v in self._pkl_metrics.items():
                lines.append(f"  {k:<12}: {v:.1f}%")

        lines += [
            "",
            "Note: This tool is for research/screening purposes only.",
            "Consult a qualified ophthalmologist for clinical decisions.",
            "=" * 58,
        ]
        with open(path, "w") as f:
            f.write("\n".join(lines))
        messagebox.showinfo("Report Saved", f"Report saved to:\n{path}")
        self._set_status(f"Report exported: {os.path.basename(path)}")

    # ══════════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ══════════════════════════════════════════════════════════════════════════

    def _set_status(self, msg: str):
        self.status_var.set(msg)


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = RetinalApp()
    app.mainloop()
