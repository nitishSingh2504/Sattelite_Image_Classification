#!/usr/bin/env python3
"""
Satellite Image Classifier  —  Landsat × Sentinel-2
Supervised + Unsupervised land-cover classification with resolution comparison.
Run:  python satellite_classifier.py
Deps: pip install rasterio scikit-learn scipy matplotlib numpy
"""

import os, threading, traceback
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# ── Optional deps ─────────────────────────────────────────────────────────────
try:
    import rasterio
    HAS_RIO = True
except ImportError:
    HAS_RIO = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from scipy.optimize import linear_sum_assignment
    HAS_SK = True
except ImportError:
    HAS_SK = False

# ── Theme ─────────────────────────────────────────────────────────────────────
BG     = "#1e1e2e"
PANEL  = "#181825"
FG     = "#cdd6f4"
DIM    = "#6c7086"
ACCENT = "#89b4fa"
BORDER = "#313244"
GREEN  = "#a6e3a1"

PALETTE = ["#1E90FF","#228B22","#FF4500","#D2691E","#9ACD32",
           "#FFD700","#A9A9A9","#8B008B","#FF69B4","#20B2AA",
           "#DC143C","#FFA500","#00CED1","#ADFF2F","#4B0082"]

DEFAULT_CLASSES = [
    ("Water",          "#1E90FF"),
    ("Vegetation",     "#228B22"),
    ("Urban/Built-up", "#FF4500"),
    ("Bare Soil",      "#D2691E"),
    ("Cropland",       "#9ACD32"),
]

SUP_ALGOS = {
    "Random Forest": lambda: RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
    "SVM (RBF)":     lambda: SVC(kernel="rbf", C=10, gamma="scale"),
    "K-Nearest (5)": lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=20, random_state=42),
    "Naive Bayes":   lambda: GaussianNB(),
} if HAS_SK else {}


# ══════════════════════════════════════════════════════════════════════════════
class App:
    def __init__(self, root):
        self.root = root
        root.title("Satellite Image Classifier  |  Landsat × Sentinel-2")
        root.geometry("1440x860")
        root.minsize(1100, 680)
        root.configure(bg=BG)

        # state
        self.ls = None
        self.st = None
        self.classes = []
        self.cur_cls  = 0
        self.sampling = False
        self.results  = {}

        # tk vars
        self.mode        = tk.StringVar(value="supervised")
        self.sup_algo    = tk.StringVar(value="Random Forest")
        self.unsup_algo  = tk.StringVar(value="K-Means")
        self.n_clusters  = tk.IntVar(value=5)
        self.max_iter    = tk.IntVar(value=500)
        self.n_init      = tk.IntVar(value=20)
        self.ls_r = tk.IntVar(value=4); self.ls_g = tk.IntVar(value=3); self.ls_b = tk.IntVar(value=2)
        self.st_r = tk.IntVar(value=4); self.st_g = tk.IntVar(value=3); self.st_b = tk.IntVar(value=2)
        self.status_var  = tk.StringVar(value="Ready — load satellite images to begin.")

        self._style()
        self._build_ui()

        for name, col in DEFAULT_CLASSES:
            self.classes.append({"name": name, "color": col, "ls": [], "st": []})
        self._refresh_cls()

        if not HAS_RIO or not HAS_SK:
            pkgs = []
            if not HAS_RIO: pkgs.append("rasterio")
            if not HAS_SK:  pkgs.append("scikit-learn scipy")
            root.after(300, lambda: messagebox.showwarning("Missing packages",
                "Please install:\n\n  pip install " + " ".join(pkgs)))

    # ── Style ─────────────────────────────────────────────────────────────────
    def _style(self):
        s = ttk.Style()
        try:
            s.theme_use("clam")
            s.configure(".",              background=BG, foreground=FG,
                         fieldbackground=PANEL, bordercolor=BORDER,
                         lightcolor=BORDER, darkcolor=BORDER)
            s.configure("TFrame",         background=BG)
            s.configure("TLabel",         background=BG, foreground=FG)
            s.configure("TLabelframe",    background=BG, bordercolor=BORDER)
            s.configure("TLabelframe.Label", background=BG, foreground=ACCENT,
                         font=("Segoe UI", 9, "bold"))
            s.configure("TButton",        background=PANEL, foreground=FG,
                         bordercolor=BORDER, padding=5)
            s.map("TButton",              background=[("active", BORDER)])
            s.configure("Run.TButton",    background=ACCENT, foreground=BG,
                         font=("Segoe UI", 9, "bold"), padding=7)
            s.map("Run.TButton",          background=[("active","#74c7ec"),("disabled",PANEL)],
                   foreground=[("disabled",DIM)])
            s.configure("TRadiobutton",   background=BG, foreground=FG)
            s.map("TRadiobutton",         background=[("active", BG)])
            s.configure("TNotebook",      background=BG, borderwidth=0)
            s.configure("TNotebook.Tab",  background=PANEL, foreground=DIM, padding=[14,6])
            s.map("TNotebook.Tab",        background=[("selected",BG)],
                   foreground=[("selected",ACCENT)])
            s.configure("TEntry",         fieldbackground=PANEL, foreground=FG,
                         insertcolor=FG, bordercolor=BORDER)
            s.configure("TSpinbox",       fieldbackground=PANEL, foreground=FG,
                         arrowcolor=FG, bordercolor=BORDER)
            s.configure("TCombobox",      fieldbackground=PANEL, foreground=FG,
                         arrowcolor=FG, bordercolor=BORDER)
            s.configure("TProgressbar",   background=ACCENT, troughcolor=PANEL)
            s.configure("TSeparator",     background=BORDER)
        except tk.TclError:
            pass

    # ── UI Layout ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)

        self._sb = tk.Frame(body, bg=BG, width=320)
        self._sb.pack(side="left", fill="y", padx=4, pady=4)
        self._sb.pack_propagate(False)
        self._build_sidebar()

        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(0,4), pady=4)
        self._build_main(right)

        bar = tk.Frame(self.root, bg=PANEL, height=26)
        bar.pack(side="bottom", fill="x")
        tk.Label(bar, textvariable=self.status_var, bg=PANEL, fg=DIM,
                  font=("Segoe UI",9), anchor="w").pack(side="left", padx=10, fill="x", expand=True)
        self.prog = ttk.Progressbar(bar, mode="determinate", length=200)
        self.prog.pack(side="right", padx=8, pady=4)

    def _build_sidebar(self):
        s = self._sb
        tk.Label(s, text="🛰  Satellite Classifier", bg=BG, fg=ACCENT,
                  font=("Segoe UI",12,"bold")).pack(anchor="w", padx=6, pady=(4,0))
        tk.Label(s, text="Landsat × Sentinel-2  |  land-cover analysis",
                  bg=BG, fg=DIM, font=("Segoe UI",8)).pack(anchor="w", padx=6)

        container = tk.Frame(s, bg=BG)
        container.pack(fill="both", expand=True)

        sc = tk.Canvas(container, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(container, orient="vertical", command=sc.yview)
        sc.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        sc.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(sc, bg=BG)
        win_id = sc.create_window((0,0), window=inner, anchor="nw")

        def _on_canvas_configure(e):
            sc.itemconfig(win_id, width=e.width)
        sc.bind("<Configure>", _on_canvas_configure)

        inner.bind("<Configure>", lambda e: sc.configure(scrollregion=sc.bbox("all")))

        def _scroll(e):
            sc.yview_scroll(int(-1*(e.delta/120)), "units")
        sc.bind_all("<MouseWheel>", _scroll)

        self._build_sidebar_content(inner)

    def _build_sidebar_content(self, s):
        # 1 Load
        f = ttk.LabelFrame(s, text="  1. Load Images  ", padding=6)
        f.pack(fill="x", padx=4, pady=(10,3))
        self.ls_lbl = tk.Label(f, text="Landsat: not loaded", bg=BG, fg=DIM,
                                 font=("Segoe UI",8), anchor="w", wraplength=290)
        self.ls_lbl.pack(fill="x")
        r = tk.Frame(f, bg=BG); r.pack(fill="x", pady=3)
        ttk.Button(r, text="Load Landsat", command=self._load_ls).pack(side="left")
        tk.Label(r, text=" RGB:", bg=BG, fg=DIM, font=("Segoe UI",8)).pack(side="left", padx=(6,0))
        for v in (self.ls_r, self.ls_g, self.ls_b):
            sb = ttk.Spinbox(r, from_=1, to=20, textvariable=v, width=3, command=self._redraw_prev)
            sb.pack(side="left", padx=1)
            sb.bind("<Return>", lambda *_: self._redraw_prev())

        self.st_lbl = tk.Label(f, text="Sentinel: not loaded", bg=BG, fg=DIM,
                                 font=("Segoe UI",8), anchor="w", wraplength=290)
        self.st_lbl.pack(fill="x", pady=(6,0))
        r2 = tk.Frame(f, bg=BG); r2.pack(fill="x", pady=3)
        ttk.Button(r2, text="Load Sentinel", command=self._load_st).pack(side="left")
        tk.Label(r2, text=" RGB:", bg=BG, fg=DIM, font=("Segoe UI",8)).pack(side="left", padx=(6,0))
        for v in (self.st_r, self.st_g, self.st_b):
            sb = ttk.Spinbox(r2, from_=1, to=20, textvariable=v, width=3, command=self._redraw_prev)
            sb.pack(side="left", padx=1)
            sb.bind("<Return>", lambda *_: self._redraw_prev())

        # 2 Classes
        f2 = ttk.LabelFrame(s, text="  2. Classes  ", padding=6)
        f2.pack(fill="x", padx=4, pady=3)
        self.cls_lb = tk.Listbox(f2, height=5, bg=PANEL, fg=FG,
                                   selectbackground=ACCENT, selectforeground=BG,
                                   relief="flat", borderwidth=1,
                                   highlightthickness=1, highlightbackground=BORDER,
                                   highlightcolor=ACCENT, font=("Segoe UI",9))
        self.cls_lb.pack(fill="x", pady=(0,4))
        self.cls_lb.bind("<<ListboxSelect>>", self._on_cls_sel)
        self.cls_lb.bind("<Double-Button-1>", lambda _: self._edit_color())
        br = tk.Frame(f2, bg=BG); br.pack(fill="x")
        for txt, cmd in [("+ Add",self._add_class),("🎨 Color",self._edit_color),
                          ("✏ Rename",self._rename_class),("✕ Del",self._del_class)]:
            ttk.Button(br, text=txt, command=cmd).pack(side="left", padx=1, expand=True, fill="x")

        # 3 Method
        f3 = ttk.LabelFrame(s, text="  3. Method & Parameters  ", padding=6)
        f3.pack(fill="x", padx=4, pady=3)
        ttk.Radiobutton(f3, text="Supervised  (requires training samples)",
                        value="supervised", variable=self.mode,
                        command=self._mode_changed).pack(anchor="w")
        ttk.Radiobutton(f3, text="Unsupervised  (automatic clustering)",
                        value="unsupervised", variable=self.mode,
                        command=self._mode_changed).pack(anchor="w")
        ttk.Separator(f3).pack(fill="x", pady=5)

        self.sup_pane = ttk.Frame(f3)
        tk.Label(self.sup_pane, text="Classifier:", bg=BG, fg=FG,
                  font=("Segoe UI",8)).pack(anchor="w")
        ttk.Combobox(self.sup_pane, textvariable=self.sup_algo,
                      values=list(SUP_ALGOS.keys()), state="readonly",
                      width=28).pack(fill="x", pady=2)

        self.uns_pane = ttk.Frame(f3)
        tk.Label(self.uns_pane, text="Algorithm:", bg=BG, fg=FG,
                  font=("Segoe UI",8)).pack(anchor="w")
        ttk.Combobox(self.uns_pane, textvariable=self.unsup_algo,
                      values=["K-Means","Gaussian Mixture (GMM)"],
                      state="readonly", width=28).pack(fill="x", pady=2)
        for lbl, var, lo, hi, inc in [("Clusters:", self.n_clusters, 2, 15, 1),
                                       ("Max iters:", self.max_iter, 50, 2000, 50),
                                       ("n_init runs:", self.n_init, 1, 30, 1)]:
            rr = tk.Frame(self.uns_pane, bg=BG); rr.pack(fill="x", pady=1)
            tk.Label(rr, text=lbl, bg=BG, fg=FG, width=13,
                      font=("Segoe UI",8)).pack(side="left")
            ttk.Spinbox(rr, from_=lo, to=hi, increment=inc,
                        textvariable=var, width=8).pack(side="left")
        self._mode_changed()

        # 4 Sample / Run
        f4 = ttk.LabelFrame(s, text="  4. Sample & Run  ", padding=6)
        f4.pack(fill="x", padx=4, pady=3)
        self.samp_btn = ttk.Button(f4, text="🖱  Enable Sampling",
                                    command=self._toggle_sample)
        self.samp_btn.pack(fill="x", pady=2)
        self.run_btn = ttk.Button(f4, text="▶  Run Classification",
                                   command=self._run, style="Run.TButton")
        self.run_btn.pack(fill="x", pady=2)
        ttk.Button(f4, text="💾  Save Results",
                   command=self._save).pack(fill="x", pady=2)
        ttk.Separator(f4).pack(fill="x", pady=4)
        ttk.Button(f4, text="📤  Save Samples",
                   command=self._save_model).pack(fill="x", pady=2)
        ttk.Button(f4, text="📥  Load Samples",
                   command=self._load_model).pack(fill="x", pady=2)

        self.summary = tk.Label(s, text="", bg=BG, fg=GREEN,
                                  font=("Segoe UI",9), justify="left",
                                  wraplength=290, anchor="w")
        self.summary.pack(fill="x", padx=6, pady=(6,4))

    def _build_main(self, parent):
        self.nb = ttk.Notebook(parent)
        self.nb.pack(fill="both", expand=True)

        self.tab_prev = ttk.Frame(self.nb)
        self.tab_res  = ttk.Frame(self.nb)
        self.tab_ana  = ttk.Frame(self.nb)
        self.nb.add(self.tab_prev, text="  🖼  Preview  ")
        self.nb.add(self.tab_res,  text="  🗺  Classification  ")
        self.nb.add(self.tab_ana,  text="  📊  Analysis  ")

        self._build_preview()
        self._build_results()
        self._build_analysis()

    def _build_preview(self):
        self.fig_p = Figure(facecolor=BG, constrained_layout=True)
        self.ax_ls = self.fig_p.add_subplot(1, 2, 1)
        self.ax_st = self.fig_p.add_subplot(1, 2, 2)
        for ax, t in [(self.ax_ls,"Landsat — Preview"),(self.ax_st,"Sentinel-2 — Preview")]:
            self._sax(ax, t)
            ax.text(0.5,0.5,"Load image (.tif)",ha="center",va="center",
                     color=DIM,transform=ax.transAxes)
        self.cv_p = FigureCanvasTkAgg(self.fig_p, self.tab_prev)
        tb = NavigationToolbar2Tk(self.cv_p, self.tab_prev, pack_toolbar=False)
        tb.update(); tb.pack(side="bottom", fill="x")
        self.cv_p.get_tk_widget().pack(fill="both", expand=True)
        self.cv_p.draw()
        self.cv_p.mpl_connect("button_press_event", self._on_click)

    def _build_results(self):
        self.fig_r = Figure(facecolor=BG, constrained_layout=True)
        self.ax_rls = self.fig_r.add_subplot(1, 2, 1)
        self.ax_rst = self.fig_r.add_subplot(1, 2, 2)
        for ax, t in [(self.ax_rls,"Landsat — Classified"),(self.ax_rst,"Sentinel-2 — Classified")]:
            self._sax(ax, t)
            ax.text(0.5,0.5,"Run classification first",ha="center",va="center",
                     color=DIM,transform=ax.transAxes)
        self.cv_r = FigureCanvasTkAgg(self.fig_r, self.tab_res)
        tb2 = NavigationToolbar2Tk(self.cv_r, self.tab_res, pack_toolbar=False)
        tb2.update(); tb2.pack(side="bottom", fill="x")
        self.cv_r.get_tk_widget().pack(fill="both", expand=True)
        self.cv_r.draw()

    def _build_analysis(self):
        self.fig_a = Figure(facecolor=BG, constrained_layout=True)
        self.cv_a  = FigureCanvasTkAgg(self.fig_a, self.tab_ana)
        tb3 = NavigationToolbar2Tk(self.cv_a, self.tab_ana, pack_toolbar=False)
        tb3.update(); tb3.pack(side="bottom", fill="x")
        self.cv_a.get_tk_widget().pack(fill="both", expand=True)
        ax = self.fig_a.add_subplot(111); self._sax(ax, "Analysis")
        ax.text(0.5,0.5,"Run classification to see analysis here.",
                 ha="center",va="center",color=DIM,transform=ax.transAxes)
        self.cv_a.draw()

    def _sax(self, ax, title=""):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=FG, fontsize=10, pad=6)
        ax.tick_params(colors=DIM, labelsize=7)
        for sp in ax.spines.values(): sp.set_color(BORDER)

    # ── Load (synchronous) ────────────────────────────────────────────────────
    def _load_ls(self): self._load("ls")
    def _load_st(self): self._load("st")

    def _load(self, kind):
        if not HAS_RIO:
            messagebox.showerror("Missing", "pip install rasterio"); return
        path = filedialog.askopenfilename(
            title=f"Select {'Landsat' if kind=='ls' else 'Sentinel-2'} GeoTIFF",
            filetypes=[("GeoTIFF","*.tif *.tiff"),("All","*.*")])
        if not path: return
        self._set_status(f"Loading {os.path.basename(path)}…", 20)
        self.root.update_idletasks()
        try:
            with rasterio.open(path) as src:
                raw = src.read().astype(np.float32)
                meta = dict(fname=os.path.basename(path), path=path,
                             W=src.width, H=src.height, bands=src.count,
                             dtype=str(src.dtypes[0]), crs=str(src.crs or "unknown"),
                             res=src.res, nodata=src.nodata)
                transform = src.transform
                crs_obj   = src.crs
            obj = dict(data=self._norm(raw), meta=meta,
                       transform=transform, crs_obj=crs_obj)
            if kind == "ls": self.ls = obj
            else:            self.st = obj
            res = max(abs(meta["res"][0]), abs(meta["res"][1]))
            lbl = self.ls_lbl if kind == "ls" else self.st_lbl
            lbl.config(fg=FG, text=(
                f"{'Landsat' if kind=='ls' else 'Sentinel'}: {meta['fname']}\n"
                f"{meta['bands']} bands  ·  {meta['W']}×{meta['H']} px  ·  {res:.0f} m/px"))
            self._redraw_prev()
            self._set_status(f"Loaded {meta['fname']}", 100)
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            self._set_status("Load failed.", 0)

    @staticmethod
    def _norm(arr):
        out = np.zeros_like(arr, dtype=np.float32)
        for i in range(arr.shape[0]):
            b = arr[i]; v = b[np.isfinite(b) & (b > 0)]
            if len(v):
                lo, hi = np.percentile(v, [2,98])
                out[i] = np.clip((b-lo)/max(hi-lo,1e-6), 0, 1)
        return out

    def _add_map_decor(self, ax, obj):
        """Add lat/lon tick labels (EPSG:4326), scale bar, and north arrow."""
        meta  = obj["meta"]
        H, W  = meta["H"], meta["W"]
        xform = obj.get("transform")
        crs   = obj.get("crs_obj")

        # ── 1. Lat / Lon tick labels ──────────────────────────────────────────
        lon_min = lon_max = lat_min = lat_max = None
        try:
            from rasterio.crs import CRS as RioCRS
            if crs is None:
                crs_str = meta.get("crs","")
                if crs_str and crs_str not in ("unknown","None","none",""):
                    crs = RioCRS.from_string(crs_str)

            if crs is not None and xform is not None:
                left  = xform.c
                top   = xform.f
                right = left  + W * xform.a
                bot   = top   + H * xform.e

                try:
                    from rasterio.warp import transform_bounds
                    lon_min, lat_min, lon_max, lat_max = \
                        transform_bounds(crs, "EPSG:4326", left, bot, right, top)
                except Exception:
                    from pyproj import Transformer
                    t = Transformer.from_crs(crs.to_epsg() or str(crs),
                                              "EPSG:4326", always_xy=True)
                    lon_min, lat_max = t.transform(left, top)
                    lon_max, lat_min = t.transform(right, bot)

        except Exception:
            pass

        if lon_min is not None:
            n = 5
            px_x = np.linspace(0, W-1, n)
            lons  = np.linspace(lon_min, lon_max, n)
            ax.set_xticks(px_x)
            ax.set_xticklabels(
                [f"{abs(v):.3f}°{'E' if v>=0 else 'W'}" for v in lons],
                fontsize=6, color=FG, rotation=25, ha="right")

            px_y = np.linspace(0, H-1, n)
            lats  = np.linspace(lat_max, lat_min, n)
            ax.set_yticks(px_y)
            ax.set_yticklabels(
                [f"{abs(v):.3f}°{'N' if v>=0 else 'S'}" for v in lats],
                fontsize=6, color=FG)

            ax.tick_params(axis="both", length=3, width=0.5,
                            labelsize=6, colors=FG, pad=2)
        else:
            ax.set_xticks([]); ax.set_yticks([])

        # ── 2. Scale bar (top-left, compact) ─────────────────────────────────
        try:
            res_m = max(abs(meta["res"][0]), abs(meta["res"][1]))
            if res_m <= 0:
                raise ValueError("zero resolution")
            # Largest "round" distance that occupies 8–18 % of image width
            chosen_km = 1.0
            bar_frac  = 0.12
            for km in [200, 100, 50, 20, 10, 5, 2, 1, 0.5, 0.25, 0.1, 0.05]:
                frac = km * 1000 / res_m / W
                if 0.08 <= frac <= 0.18:
                    chosen_km = km
                    bar_frac  = frac
                    break
            else:
                # Fallback: target ~12 % width, pick nearest nice km value
                target_m = 0.12 * W * res_m
                for km in [200, 100, 50, 20, 10, 5, 2, 1, 0.5, 0.25, 0.1]:
                    if km * 1000 <= target_m * 1.5:
                        chosen_km = km
                        break
                bar_frac = min(0.18, max(0.04, chosen_km * 1000 / res_m / W))

            bx = 0.03   # left edge (axes fraction)
            by = 0.905  # near top
            bh = 0.016  # bar height — thin
            pad = 0.006

            ax.add_patch(plt.Rectangle(                           # semi-transparent backing
                (bx - pad, by - pad * 2.5),
                bar_frac + pad * 2 + 0.028, bh + pad * 6 + 0.018,
                transform=ax.transAxes, color="black", alpha=0.50,
                zorder=8, clip_on=True))
            ax.add_patch(plt.Rectangle(                           # white left half
                (bx, by), bar_frac / 2, bh,
                transform=ax.transAxes, color="white", zorder=9, clip_on=True))
            ax.add_patch(plt.Rectangle(                           # dark right half
                (bx + bar_frac / 2, by), bar_frac / 2, bh,
                transform=ax.transAxes, color="#333", zorder=9, clip_on=True))
            ax.add_patch(plt.Rectangle(                           # border outline
                (bx, by), bar_frac, bh, fill=False,
                transform=ax.transAxes, edgecolor="white",
                linewidth=0.7, zorder=10, clip_on=True))
            kw = dict(transform=ax.transAxes, ha="center", va="bottom",
                      color="white", fontsize=6, fontweight="bold", zorder=11)
            ax.text(bx,                 by + bh + 0.002, "0",                        **kw)
            ax.text(bx + bar_frac / 2,  by + bh + 0.002,
                    f"{chosen_km/2:.3g}".rstrip("0").rstrip("."),                     **kw)
            ax.text(bx + bar_frac,      by + bh + 0.002, f"{chosen_km:.4g} km",      **kw)
        except Exception:
            pass

        # ── 3. North arrow (xycoords="axes fraction" string — NOT transform ──
        try:
            ax.add_patch(plt.Rectangle(
                (0.870, 0.78), 0.115, 0.195,
                transform=ax.transAxes, color="black", alpha=0.55,
                zorder=9, clip_on=True))
            ax.annotate("",
                xy    =(0.928, 0.945),
                xytext=(0.928, 0.845),
                xycoords="axes fraction",
                textcoords="axes fraction",
                arrowprops=dict(
                    arrowstyle="->, head_width=0.25, head_length=0.15",
                    color="white", lw=2.0, mutation_scale=14),
                zorder=12)
            ax.text(0.928, 0.805, "N",
                    transform=ax.transAxes, ha="center", va="top",
                    color="white", fontsize=9, fontweight="bold", zorder=12)
        except Exception:
            pass

    def _redraw_prev(self):
        cfg = [(self.ax_ls,self.ls,self.ls_r,self.ls_g,self.ls_b,"Landsat — Preview","ls"),
               (self.ax_st,self.st,self.st_r,self.st_g,self.st_b,"Sentinel-2 — Preview","st")]
        for ax, obj, rv, gv, bv, title, key in cfg:
            ax.clear(); self._sax(ax, title)
            if obj is None:
                ax.text(0.5,0.5,"No image loaded",ha="center",va="center",
                         color=DIM,transform=ax.transAxes); continue
            d = obj["data"]
            ri=min(max(rv.get()-1,0),d.shape[0]-1)
            gi=min(max(gv.get()-1,0),d.shape[0]-1)
            bi=min(max(bv.get()-1,0),d.shape[0]-1)
            ax.imshow(np.clip(np.dstack([d[ri],d[gi],d[bi]]),0,1),
                       aspect="equal", interpolation="nearest")
            m = obj["meta"]; res=max(abs(m["res"][0]),abs(m["res"][1]))
            ax.set_xlabel(f"{m['bands']} bands · {res:.0f} m/px · {m['W']}×{m['H']}",
                           color=DIM, fontsize=8)
            self._add_map_decor(ax, obj)
            for cls in self.classes:
                for (r,c) in cls[key]:
                    ax.plot(c,r,"o",color=cls["color"],ms=5,mew=0.8,mec="white")
        self.cv_p.draw_idle()

    # ── Class management ──────────────────────────────────────────────────────
    def _refresh_cls(self):
        self.cls_lb.delete(0,"end")
        for i,c in enumerate(self.classes):
            self.cls_lb.insert("end",
                f"  ●  {c['name']:16s}  [LS:{len(c['ls'])} ST:{len(c['st'])}]")
            self.cls_lb.itemconfig(i, fg=c["color"])
        if self.classes:
            idx=min(self.cur_cls,len(self.classes)-1)
            self.cls_lb.selection_set(idx); self.cur_cls=idx

    def _on_cls_sel(self, *_):
        sel=self.cls_lb.curselection()
        if sel: self.cur_cls=sel[0]

    def _dialog(self, title, prompt, default="", pick_color=False):
        dlg=tk.Toplevel(self.root); dlg.title(title); dlg.configure(bg=BG)
        dlg.geometry("320x170"); dlg.transient(self.root)
        dlg.grab_set(); dlg.resizable(False,False)
        tk.Label(dlg,text=prompt,bg=BG,fg=FG).pack(padx=12,pady=(12,4),anchor="w")
        var=tk.StringVar(value=default)
        ent=ttk.Entry(dlg,textvariable=var,width=36)
        ent.pack(padx=12,fill="x"); ent.select_range(0,"end"); ent.focus_set()
        col_ref=[PALETTE[len(self.classes)%len(PALETTE)]]
        if pick_color:
            cr=tk.Frame(dlg,bg=BG); cr.pack(padx=12,pady=6,fill="x")
            tk.Label(cr,text="Colour:",bg=BG,fg=FG).pack(side="left")
            cb=tk.Button(cr,bg=col_ref[0],width=4,relief="flat",
                          command=lambda: self._pick(col_ref,cb))
            cb.pack(side="left",padx=8)
        res={"ok":False}
        def ok():  res["ok"]=True; dlg.destroy()
        def cxl(): dlg.destroy()
        br=tk.Frame(dlg,bg=BG); br.pack(pady=10)
        ttk.Button(br,text="OK",    command=ok, width=10).pack(side="left",padx=4)
        ttk.Button(br,text="Cancel",command=cxl,width=10).pack(side="left",padx=4)
        ent.bind("<Return>", lambda _: ok())
        dlg.wait_window()
        return (var.get().strip(), col_ref[0]) if res["ok"] else (None,None)

    def _pick(self, holder, btn):
        c=colorchooser.askcolor(color=holder[0],title="Pick colour")[1]
        if c: holder[0]=c; btn.config(bg=c)

    def _add_class(self):
        name,col=self._dialog("Add Class","Class name:","New Class",pick_color=True)
        if name:
            self.classes.append({"name":name,"color":col,"ls":[],"st":[]})
            self.cur_cls=len(self.classes)-1; self._refresh_cls()

    def _edit_color(self):
        sel=self.cls_lb.curselection()
        if not sel or not self.classes: return
        i=sel[0]
        c=colorchooser.askcolor(color=self.classes[i]["color"],
                                  title=f"Colour for '{self.classes[i]['name']}'")[1]
        if c:
            self.classes[i]["color"]=c; self._refresh_cls(); self._redraw_prev()
            if self.results: self._redraw_result(); self._redraw_analysis()

    def _rename_class(self):
        sel=self.cls_lb.curselection()
        if not sel or not self.classes: return
        i=sel[0]
        name,_=self._dialog("Rename","New name:",self.classes[i]["name"])
        if name:
            self.classes[i]["name"]=name; self._refresh_cls()
            if self.results: self._redraw_result(); self._redraw_analysis()

    def _del_class(self):
        sel=self.cls_lb.curselection()
        if not sel or not self.classes: return
        self.classes.pop(sel[0]); self.cur_cls=max(0,sel[0]-1)
        self._refresh_cls(); self._redraw_prev()

    # ── Sampling ──────────────────────────────────────────────────────────────
    def _toggle_sample(self):
        self.sampling=not self.sampling
        if self.sampling:
            self.samp_btn.config(text="🔴  Sampling ON — click to stop")
            self._set_status("Sampling ON — click on the Preview images.",0)
        else:
            self.samp_btn.config(text="🖱  Enable Sampling")
            self._set_status("Sampling OFF.",0)

    def _on_click(self, event):
        if not self.sampling or event.inaxes is None or not self.classes: return
        if   event.inaxes is self.ax_ls and self.ls: obj,key=self.ls,"ls"
        elif event.inaxes is self.ax_st and self.st: obj,key=self.st,"st"
        else: return
        c,r=int(round(event.xdata)),int(round(event.ydata))
        H,W=obj["data"].shape[1:]
        if 0<=r<H and 0<=c<W:
            self.classes[self.cur_cls][key].append((r,c))
            self._refresh_cls(); self._redraw_prev()

    # ── Mode ──────────────────────────────────────────────────────────────────
    def _mode_changed(self):
        if self.mode.get()=="supervised":
            self.uns_pane.pack_forget(); self.sup_pane.pack(fill="x",pady=2)
        else:
            self.sup_pane.pack_forget(); self.uns_pane.pack(fill="x",pady=2)

    # ── Run dispatcher ────────────────────────────────────────────────────────
    def _run(self):
        if not HAS_SK:
            messagebox.showerror("Missing","pip install scikit-learn scipy"); return
        if self.ls is None and self.st is None:
            messagebox.showerror("No image","Load at least one image first."); return
        sup = self.mode.get()=="supervised"
        if sup:
            if not self.classes:
                messagebox.showerror("No classes","Add at least one class."); return
            if sum(len(c["ls"])+len(c["st"]) for c in self.classes)==0:
                messagebox.showerror("No samples",
                    "Enable sampling and click on the images to collect training pixels."); return
            if sum(1 for c in self.classes if c["ls"] or c["st"])<2:
                messagebox.showwarning("Warning","Sample at least 2 classes."); return
        else:
            k=self.n_clusters.get()
            self.classes = [{"name": f"Cluster {i+1}",
                              "color": PALETTE[i % len(PALETTE)],
                              "ls": [], "st": []}
                             for i in range(k)]
            self._refresh_cls()
        self.run_btn.config(state="disabled"); self.results={}
        fn=self._thread_sup if sup else self._thread_uns
        threading.Thread(target=fn, daemon=True).start()

    # ── Build training arrays ─────────────────────────────────────────────────
    def _build_xy(self, data, key):
        X,y=[],[]
        H,W=data.shape[1],data.shape[2]
        for i,cls in enumerate(self.classes):
            for (r,c) in cls[key]:
                r0,r1=max(0,r-1),min(H-1,r+1)
                c0,c1=max(0,c-1),min(W-1,c+1)
                patch=data[:,r0:r1+1,c0:c1+1]
                for pr in range(patch.shape[1]):
                    for pc in range(patch.shape[2]):
                        X.append(patch[:,pr,pc]); y.append(i)
        return np.array(X,dtype=np.float32), np.array(y)

    # ── Supervised thread ─────────────────────────────────────────────────────
    def _thread_sup(self):
        try:
            algo=self.sup_algo.get(); out={}
            for key,obj,label in [("ls",self.ls,"Landsat"),("st",self.st,"Sentinel-2")]:
                if obj is None: continue
                if sum(len(c[key]) for c in self.classes)==0: continue
                self._set_status(f"Training {algo} on {label}…",25)
                X,y=self._build_xy(obj["data"],key)
                if len(X)<4 or len(np.unique(y))<2: continue
                # 70 / 30 stratified split — test set never seen during training
                try:
                    X_tr,X_te,y_tr,y_te=train_test_split(
                        X,y,test_size=0.30,random_state=42,stratify=y)
                except ValueError:
                    X_tr,X_te,y_tr,y_te=train_test_split(
                        X,y,test_size=0.30,random_state=42)
                scaler=StandardScaler()
                X_trs=scaler.fit_transform(X_tr)
                X_tes=scaler.transform(X_te)
                clf=SUP_ALGOS[algo](); clf.fit(X_trs,y_tr)
                # Accuracy metrics on held-out 30 % test set
                yp_te=clf.predict(X_tes)
                test_acc=accuracy_score(y_te,yp_te)
                cm_tr=confusion_matrix(y_te,yp_te)
                n_te=cm_tr.sum()
                pe=sum(cm_tr[i,:].sum()*cm_tr[:,i].sum()
                       for i in range(len(np.unique(y_te))))/max(n_te**2,1)
                kappa=(test_acc-pe)/(1-pe) if pe<1 else 0.0
                n_splits=max(2,min(5,int(np.bincount(y_tr).min())))
                try:
                    cv=cross_val_score(clf,X_trs,y_tr,
                        cv=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=42),
                        scoring="accuracy")
                except Exception:
                    cv=np.array([test_acc])
                self._set_status(f"Predicting {label}…",65)
                H,W=obj["data"].shape[1:]
                flat=obj["data"].reshape(obj["data"].shape[0],-1).T
                pred=clf.predict(scaler.transform(flat)).reshape(H,W)
                out[key]=dict(supervised=True,algo=algo,pred=pred,
                               test_acc=test_acc,kappa=kappa,cm_tr=cm_tr,cv=cv,
                               clf=clf,scaler=scaler,X=X_te,y=y_te,
                               n_train=len(X_tr),n_test=len(X_te))
            self.results=out; self.root.after(0,self._run_done)
        except Exception:
            tb=traceback.format_exc(); self.root.after(0,lambda:self._run_error(tb))

    # ── Unsupervised thread ───────────────────────────────────────────────────
    def _thread_uns(self):
        try:
            algo=self.unsup_algo.get(); k=self.n_clusters.get()
            max_it=self.max_iter.get(); n_in=self.n_init.get()
            out,raw={},{}
            for key,obj,label in [("ls",self.ls,"Landsat"),("st",self.st,"Sentinel-2")]:
                if obj is None: continue
                self._set_status(f"{algo} on {label}…",30)
                data=obj["data"]; H,W=data.shape[1:]
                flat=data.reshape(data.shape[0],-1).T
                if flat.shape[0]>150_000:
                    idx=np.random.default_rng(42).choice(flat.shape[0],150_000,replace=False)
                    train=flat[idx]
                else:
                    train=flat
                scaler=StandardScaler()
                train_s=scaler.fit_transform(train); flat_s=scaler.transform(flat)
                if "GMM" in algo:
                    m=GaussianMixture(n_components=k,random_state=42,n_init=n_in,
                                       max_iter=max_it,init_params="k-means++",
                                       covariance_type="full")
                    m.fit(train_s); pred=m.predict(flat_s).reshape(H,W)
                    score,sname,n_it=m.bic(train_s),"BIC",m.n_iter_
                else:
                    m=KMeans(n_clusters=k,random_state=42,n_init=n_in,
                              max_iter=max_it,init="k-means++")
                    m.fit(train_s); pred=m.predict(flat_s).reshape(H,W)
                    score,sname,n_it=m.inertia_,"Inertia",m.n_iter_
                counts=np.bincount(pred.flatten(),minlength=k)
                out[key]=dict(supervised=False,algo=algo,k=k,pred=pred,
                               score=score,score_name=sname,n_iter=n_it,
                               counts=counts,pct=100.0*counts/counts.sum())
                raw[key]=dict(data=data,pred=pred)
            if "ls" in raw and "st" in raw:
                self._set_status("Synchronising cluster labels…",88)
                remap=self._sync(raw["ls"]["data"],raw["ls"]["pred"],
                                  raw["st"]["data"],raw["st"]["pred"],k)
                np_=remap[out["st"]["pred"]]
                out["st"]["pred"]=np_
                cnt=np.bincount(np_.flatten(),minlength=k)
                out["st"]["counts"]=cnt; out["st"]["pct"]=100.0*cnt/cnt.sum()
            self.results=out; self.root.after(0,self._run_done)
        except Exception:
            tb=traceback.format_exc(); self.root.after(0,lambda:self._run_error(tb))

    def _sync(self, ls_d, ls_p, st_d, st_p, k):
        def gi(d,rv,gv,bv):
            return [min(max(v.get()-1,0),d.shape[0]-1) for v in (rv,gv,bv)]
        li=gi(ls_d,self.ls_r,self.ls_g,self.ls_b)
        si=gi(st_d,self.st_r,self.st_g,self.st_b)
        def cents(data,pred,idx):
            c=np.zeros((k,3))
            for cid in range(k):
                m=pred==cid
                if m.any():
                    for ch,b in enumerate(idx): c[cid,ch]=data[b][m].mean()
            return c
        lc,sc=cents(ls_d,ls_p,li),cents(st_d,st_p,si)
        cost=np.linalg.norm(lc[:,None]-sc[None,:],axis=2)
        la,sa=linear_sum_assignment(cost)
        remap=np.arange(k)
        for a,b in zip(la,sa): remap[b]=a
        return remap

    def _run_done(self):
        self.run_btn.config(state="normal")
        self._set_status("Classification complete.",100)
        self._redraw_result(); self._redraw_analysis(); self.nb.select(1)
        lines=[]
        for k,lbl in [("ls","Landsat"),("st","Sentinel-2")]:
            if k not in self.results: continue
            r=self.results[k]
            if r["supervised"]:
                lines.append(f"✔ {lbl}: OA {r['test_acc']:.1%} (test)  CV {r['cv'].mean():.1%}")
            else:
                lines.append(f"✔ {lbl}: {r['algo']} k={r['k']} ({r['n_iter']} iters)")
        self.summary.config(text="\n".join(lines))
        r0 = next(iter(self.results.values()), None)
        if r0 and not r0["supervised"]:
            self.root.after(400, self._rename_clusters_prompt)

    def _rename_clusters_prompt(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Rename Clusters (optional)")
        dlg.configure(bg=BG)
        dlg.geometry("380x" + str(60 + len(self.classes)*38))
        dlg.transient(self.root); dlg.grab_set(); dlg.resizable(False, False)
        tk.Label(dlg,
                  text="Optionally rename clusters based on what you see in the map.\n"
                       "Leave as-is and click OK to keep default names.",
                  bg=BG, fg=DIM, font=("Segoe UI",8), justify="left",
                  wraplength=350).pack(padx=12, pady=(10,6), anchor="w")
        entries = []
        for i, cls in enumerate(self.classes):
            row = tk.Frame(dlg, bg=BG); row.pack(fill="x", padx=12, pady=2)
            b = tk.Button(row, bg=cls["color"], width=3, relief="flat",
                           command=lambda idx=i: self._edit_color_by_idx(idx, entries))
            b.pack(side="left", padx=(0,6))
            entries.append(b)
            var = tk.StringVar(value=cls["name"])
            cls["_rename_var"] = var
            ttk.Entry(row, textvariable=var, width=30).pack(side="left", fill="x", expand=True)
        def apply():
            for cls in self.classes:
                if "_rename_var" in cls:
                    name = cls["_rename_var"].get().strip()
                    if name: cls["name"] = name
                    del cls["_rename_var"]
            self._refresh_cls(); self._redraw_result(); self._redraw_analysis()
            dlg.destroy()
        def cancel():
            for cls in self.classes:
                cls.pop("_rename_var", None)
            dlg.destroy()
        br = tk.Frame(dlg, bg=BG); br.pack(pady=10)
        ttk.Button(br, text="Apply Names", command=apply, width=14).pack(side="left", padx=4)
        ttk.Button(br, text="Keep Defaults", command=cancel, width=14).pack(side="left", padx=4)

    def _edit_color_by_idx(self, idx, _entries=None):
        c = colorchooser.askcolor(color=self.classes[idx]["color"],
                                    title=f"Colour for '{self.classes[idx]['name']}'")[1]
        if c:
            self.classes[idx]["color"] = c
            if _entries and idx < len(_entries):
                _entries[idx].config(bg=c)

    def _run_error(self, tb):
        self.run_btn.config(state="normal")
        self._set_status("Classification failed.",0)
        messagebox.showerror("Error",tb)

    # ── Result rendering ──────────────────────────────────────────────────────
    def _cmap(self):
        cols=[c["color"] for c in self.classes] or ["#fff"]
        return (ListedColormap(cols), max(len(cols)-1,1),
                [mpatches.Patch(color=c["color"],label=c["name"]) for c in self.classes])

    def _redraw_result(self):
        cmap,vmax,pats=self._cmap()
        for ax,key,title in [(self.ax_rls,"ls","Landsat — Classified"),
                              (self.ax_rst,"st","Sentinel-2 — Classified")]:
            ax.clear(); self._sax(ax,title)
            if key not in self.results:
                ax.text(0.5,0.5,"No result",ha="center",va="center",
                         color=DIM,transform=ax.transAxes); continue
            r=self.results[key]
            ax.imshow(r["pred"],cmap=cmap,vmin=0,vmax=vmax,
                       aspect="equal",interpolation="nearest")
            obj = self.ls if key=="ls" else self.st
            if obj: self._add_map_decor(ax, obj)
            if r["supervised"]:
                sub=f"{r['algo']}  ·  OA {r['test_acc']:.1%}  ·  Kappa {r['kappa']:.3f}  ·  CV {r['cv'].mean():.1%}"
                ax.text(0.01, 0.99,
                        f"OA:    {r['test_acc']:.2%}  (test 30%)\nKappa: {r['kappa']:.4f}\nCV:    {r['cv'].mean():.2%}",
                        transform=ax.transAxes, va="top", ha="left", fontsize=8, color=FG,
                        bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL,
                                  alpha=0.88, edgecolor=BORDER))
            else:
                sub=f"{r['algo']}  k={r['k']}  ·  {r['score_name']} {r['score']:.0f}  ·  {r['n_iter']} iters"
            ax.set_xlabel(sub,color=DIM,fontsize=8)
            ax.legend(handles=pats,loc="lower right",fontsize=6,
                       facecolor=PANEL,labelcolor=FG,framealpha=0.9,edgecolor=BORDER)
        self.cv_r.draw_idle()

    def _redraw_analysis(self):
        from matplotlib.gridspec import GridSpec
        self.fig_a.clear()
        keys=[(k,l) for k,l in [("ls","Landsat"),("st","Sentinel-2")] if k in self.results]
        if not keys:
            ax=self.fig_a.add_subplot(111); self._sax(ax)
            ax.text(0.5,0.5,"No results",ha="center",va="center",
                     color=DIM,transform=ax.transAxes)
            self.cv_a.draw_idle(); return
        sup=self.results[keys[0][0]]["supervised"]; n=len(keys)
        if sup:
            gs=GridSpec(2,2*n,figure=self.fig_a,height_ratios=[2,1.3],
                        hspace=0.65,wspace=0.35,
                        left=0.10,right=0.97,top=0.95,bottom=0.07)
            for i,(k,title) in enumerate(keys):
                ax=self.fig_a.add_subplot(gs[0,i*2:(i+1)*2]); self._plot_cm(ax,k,title)
            ax_tbl=self.fig_a.add_subplot(gs[1,0:n]); self._plot_acc_table(ax_tbl,keys)
            ax2=self.fig_a.add_subplot(gs[1,n:2*n]); self._plot_acc(ax2,keys)
        else:
            ax=self.fig_a.add_subplot(111); self._plot_cluster(ax,keys)
            self.fig_a.subplots_adjust(0.07,0.18,0.97,0.93)
        self.cv_a.draw_idle()

    def _plot_cm(self, ax, key, title):
        r=self.results[key]
        names=[c["name"] for c in self.classes]
        uniq=sorted(np.unique(r["y"]))
        tn=[names[i] for i in uniq if i<len(names)]
        yp=r["clf"].predict(r["scaler"].transform(r["X"]))
        cm=confusion_matrix(r["y"],yp,labels=uniq)
        oa=accuracy_score(r["y"],yp)
        n=cm.sum()
        pe=sum(cm[i,:].sum()*cm[:,i].sum() for i in range(len(uniq)))/max(n**2,1)
        kappa=(oa-pe)/(1-pe) if pe<1 else 0.0
        pa=cm.diagonal()/cm.sum(axis=1).clip(1)   # Producer's Accuracy (recall)
        ua=cm.diagonal()/cm.sum(axis=0).clip(1)   # User's Accuracy (precision)
        ax.set_facecolor(PANEL); ax.imshow(cm,cmap="Blues",aspect="auto")
        ax.set_title(f"{title}\nOA {oa:.1%}  ·  Kappa {kappa:.3f}  (test 30%)",color=FG,fontsize=9,pad=4)
        t=range(len(tn))
        ax.set_xticks(t)
        ax.set_xticklabels([f"{n}\nUA {u:.0%}" for n,u in zip(tn,ua)],
                            rotation=35,ha="right",color=FG,fontsize=6)
        ax.set_yticks(t)
        ax.set_yticklabels([f"{n}  PA {p:.0%}" for n,p in zip(tn,pa)],
                            color=FG,fontsize=6)
        ax.set_xlabel("Predicted",color=DIM,fontsize=8)
        ax.set_ylabel("True",color=DIM,fontsize=8)
        thr=cm.max()/2 if cm.max() else 1
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=8,
                         color="white" if cm[i,j]<thr else "#111")
        for sp in ax.spines.values(): sp.set_color(BORDER)

    def _plot_acc(self, ax, keys):
        labels=[l for _,l in keys]
        test=[self.results[k]["test_acc"] for k,_ in keys]
        cv  =[self.results[k]["cv"].mean() for k,_ in keys]
        x=np.arange(len(labels)); w=0.32
        b1=ax.bar(x-w/2,test,w,label="Test set (30%)",color=ACCENT,edgecolor="white",lw=0.5)
        b2=ax.bar(x+w/2,cv,  w,label="Cross-validation",color=GREEN,edgecolor="white",lw=0.5)
        for bars in (b1,b2):
            for b in bars:
                ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.01,
                         f"{b.get_height():.1%}",ha="center",color=FG,fontsize=9)
        ax.set_facecolor(PANEL); ax.set_xticks(x); ax.set_xticklabels(labels,color=FG)
        ax.set_ylim(0,1.15); ax.set_ylabel("Accuracy",color=FG)
        ax.set_title(f"Accuracy Comparison — {self.results[keys[0][0]]['algo']}",
                      color=FG,fontsize=11,pad=6)
        ax.legend(facecolor=PANEL,labelcolor=FG,edgecolor=BORDER)
        ax.tick_params(colors=DIM); ax.grid(axis="y",linestyle=":",color=BORDER,alpha=0.5)
        for sp in ax.spines.values(): sp.set_color(BORDER)

    def _plot_acc_table(self, ax, keys):
        ax.set_facecolor(PANEL); ax.axis("off")
        ax.set_title("Accuracy Assessment", color=FG, fontsize=9, pad=6)
        names=[c["name"] for c in self.classes]
        lines=[]
        for k,label in keys:
            r=self.results[k]
            uniq=list(sorted(np.unique(r["y"])))
            lines.append(f"── {label} ──────────────────")
            lines.append(f"  OA    : {r['test_acc']:>7.2%}  (test 30%)")
            lines.append(f"  Kappa : {r['kappa']:>7.4f}")
            lines.append(f"  CV    : {r['cv'].mean():>7.2%}")
            lines.append("")
            lines.append(f"  {'Class':<14}  {'PA':>5}  {'UA':>5}")
            lines.append(f"  {'─'*30}")
            for i,name in enumerate(names):
                if i in uniq:
                    idx=uniq.index(i); cm=r["cm_tr"]
                    pa=cm[idx,idx]/max(cm[idx,:].sum(),1)
                    ua=cm[idx,idx]/max(cm[:,idx].sum(),1)
                    lines.append(f"  {name[:14]:<14}  {pa:>5.1%}  {ua:>5.1%}")
            lines.append("")
        ax.text(0.04,0.96,"\n".join(lines),transform=ax.transAxes,
                va="top",ha="left",fontsize=7.5,color=FG,family="monospace")

    def _plot_cluster(self, ax, keys):
        names   = [c["name"]  for c in self.classes]
        colors  = [c["color"] for c in self.classes]
        # Landsat = solid fill, Sentinel-2 = hatched — visually distinct
        hatches = ["", "///"]
        x = np.arange(len(names))
        w = 0.35
        for i, (key, label) in enumerate(keys):
            pct = self.results[key]["pct"]
            pct = np.pad(pct, (0, max(0, len(names) - len(pct))))[:len(names)]
            off = (i - (len(keys) - 1) / 2) * w
            hatch = hatches[i % len(hatches)]
            bars = ax.bar(x + off, pct, w, edgecolor="white", lw=0.5,
                          hatch=hatch, zorder=3)
            for b, col in zip(bars, colors):
                b.set_facecolor(col)
                b.set_alpha(0.95 - 0.25 * i)
            for b, v in zip(bars, pct):
                if v > 0:
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.3,
                            f"{v:.1f}%", ha="center", color=FG, fontsize=8)
        ax.set_facecolor(PANEL)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", color=FG, fontsize=9)
        ax.set_ylabel("Coverage (%)", color=FG)
        r0 = self.results[keys[0][0]]
        ax.set_title(f"Cluster Coverage — {r0['algo']} (k={r0['k']})",
                     color=FG, fontsize=11, pad=6)

        # ── Legend: cluster colours (left column) + sensor fill style (right) ──
        cluster_handles = [
            mpatches.Patch(facecolor=col, edgecolor="white", linewidth=0.5, label=name)
            for name, col in zip(names, colors)
        ]
        sensor_handles = [
            mpatches.Patch(facecolor="#999999", hatch=hatches[i % len(hatches)],
                           edgecolor="white", linewidth=0.5, label=label)
            for i, (_, label) in enumerate(keys)
        ]
        ax.legend(handles=cluster_handles + sensor_handles,
                  facecolor=PANEL, labelcolor=FG, edgecolor=BORDER,
                  fontsize=8, ncol=2, loc="upper right")
        ax.tick_params(colors=DIM)
        ax.grid(axis="y", linestyle=":", color=BORDER, alpha=0.5)
        for sp in ax.spines.values(): sp.set_color(BORDER)

    # ── Save ──────────────────────────────────────────────────────────────────
    def _save_model(self):
        if not any(c["ls"] or c["st"] for c in self.classes):
            messagebox.showinfo("Nothing yet","Add training samples first."); return
        path=filedialog.asksaveasfilename(
            title="Save training samples",
            defaultextension=".json",
            filetypes=[("Training Samples","*.json"),("All","*.*")])
        if not path: return
        import json
        payload={"classes":[{"name":c["name"],"color":c["color"],
                              "ls":c["ls"],"st":c["st"]}
                             for c in self.classes]}
        try:
            with open(path,"w",encoding="utf-8") as f:
                json.dump(payload,f,indent=2)
            self._set_status(f"Samples saved: {os.path.basename(path)}",100)
        except Exception:
            messagebox.showerror("Save error",traceback.format_exc())

    def _load_model(self):
        path=filedialog.askopenfilename(
            title="Load training samples",
            filetypes=[("Training Samples","*.json"),("All","*.*")])
        if not path: return
        import json
        try:
            with open(path,"r",encoding="utf-8") as f:
                payload=json.load(f)
            self.classes=[]
            for c in payload["classes"]:
                self.classes.append({"name":c["name"],"color":c["color"],
                                     "ls":[tuple(p) for p in c["ls"]],
                                     "st":[tuple(p) for p in c["st"]]})
            self.cur_cls=0
            self._refresh_cls()
            self._redraw_prev()
            self._set_status(f"Samples loaded: {os.path.basename(path)}",100)
        except Exception:
            messagebox.showerror("Load error",traceback.format_exc())

    def _save(self):
        if not self.results:
            messagebox.showinfo("Nothing yet","Run classification first."); return
        folder=filedialog.askdirectory(title="Select output folder")
        if not folder: return
        try:
            ts=datetime.now().strftime("%Y%m%d_%H%M%S"); saved=[]
            p=os.path.join(folder,f"report_{ts}.txt")
            with open(p,"w",encoding="utf-8") as f: f.write(self._report())
            saved.append(os.path.basename(p))
            cmap,vmax,pats=self._cmap()
            for key,label in [("ls","Landsat"),("st","Sentinel-2")]:
                if key not in self.results: continue
                r=self.results[key]
                # Use constrained_layout so tick labels are never clipped on save
                fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
                fig.patch.set_facecolor(BG)
                self._sax(ax, "")
                ax.imshow(r["pred"],cmap=cmap,vmin=0,vmax=vmax,
                           aspect="equal",interpolation="nearest")
                ax.legend(handles=pats,loc="lower right",fontsize=9,
                           facecolor=PANEL,labelcolor=FG,framealpha=0.9,edgecolor=BORDER)
                obj2 = self.ls if key=="ls" else self.st
                if obj2: self._add_map_decor(ax, obj2)
                if r["supervised"]:
                    sub=f"OA {r['test_acc']:.2%} (test 30%)  Kappa {r['kappa']:.4f}  CV {r['cv'].mean():.2%}"
                else:
                    sub=f"{r['algo']}  {r['score_name']} {r['score']:.0f}"
                ax.set_title(f"{label} Land-Cover Classification\n{sub}",fontsize=12,color=FG)
                fp=os.path.join(folder,f"{key}_classified_{ts}.png")
                fig.savefig(fp, dpi=150, bbox_inches="tight", facecolor=BG); plt.close(fig)
                saved.append(os.path.basename(fp))
            fp2=os.path.join(folder,f"analysis_{ts}.png")
            self.fig_a.savefig(fp2,dpi=150,bbox_inches="tight",facecolor=BG)
            saved.append(os.path.basename(fp2))
            messagebox.showinfo("Saved ✓",
                f"Saved to:\n{folder}\n\n"+"\n".join(f"  • {s}" for s in saved))
            self._set_status(f"Saved {len(saved)} files.",100)
        except Exception:
            messagebox.showerror("Save error",traceback.format_exc())

    def _report(self):
        L=[]; w=L.append; sep="="*68
        w(sep); w("  SATELLITE IMAGE CLASSIFICATION REPORT")
        w(f"  Generated : {datetime.now():%Y-%m-%d %H:%M:%S}")
        r0=next(iter(self.results.values()),None)
        if r0:
            if r0["supervised"]: w(f"  Mode      : Supervised — {r0['algo']}")
            else: w(f"  Mode      : Unsupervised — {r0['algo']} (k={r0['k']}, "
                    f"max_iter={self.max_iter.get()}, n_init={self.n_init.get()})")
        w(sep); w("")

        w("1. DATASETS"); w("-"*50)
        for kind,obj,sensor,rad in [
            ("ls",self.ls,"Landsat 8/9 OLI","16-bit DN (0-65535)"),
            ("st",self.st,"Sentinel-2 MSI L2A","12-bit reflectance (0-10000)")]:
            if obj is None: continue
            m=obj["meta"]; res=max(abs(m["res"][0]),abs(m["res"][1]))
            w(f"\n  {sensor}")
            w(f"    File       : {m['fname']}")
            w(f"    Size       : {m['W']} x {m['H']} pixels")
            w(f"    Bands      : {m['bands']}")
            w(f"    Spatial    : {res:.1f} m/pixel")
            w(f"    Data type  : {m['dtype']}")
            w(f"    CRS        : {m['crs']}")

        w(""); w("2. CLASSES"); w("-"*50)
        for i,c in enumerate(self.classes):
            w(f"  [{i}] {c['name']:20s}  LS:{len(c['ls']):3d}  ST:{len(c['st']):3d}")

        w(""); w("3. RESULTS"); w("-"*50)
        for key,label in [("ls","LANDSAT"),("st","SENTINEL-2")]:
            if key not in self.results: continue
            r=self.results[key]; w(f"\n  {label}")
            if r["supervised"]:
                w(f"    Split             : 70% train / 30% test (stratified)")
                w(f"    Training pixels   : {r['n_train']}")
                w(f"    Test pixels       : {r['n_test']}")
                w(f"    CV accuracy       : {r['cv'].mean():.4f} +/- {r['cv'].std():.4f}")
                w(f"    CV folds          : {' | '.join(f'{s:.3f}' for s in r['cv'])}")
                cn=[c["name"] for c in self.classes]
                uniq=sorted(np.unique(r["y"]))
                tn=[cn[i] for i in uniq if i<len(cn)]
                yp=r["clf"].predict(r["scaler"].transform(r["X"]))

                w(""); w("    Accuracy Assessment  (evaluated on held-out 30% test set):")
                w(f"      Overall Accuracy (OA) : {r['test_acc']:.4f}  ({r['test_acc']:.2%})")
                w(f"      Kappa Coefficient     : {r['kappa']:.4f}")
                w(f"      CV Accuracy (mean)    : {r['cv'].mean():.4f} +/- {r['cv'].std():.4f}")

                w(""); w("    Error Matrix  (rows = True class,  cols = Predicted class)")
                w("      PA = Producer's Accuracy (Recall)   UA = User's Accuracy (Precision)")
                cw=12
                hdr="      "+f"{'Class':<20s}"+"".join(f"{t[:cw]:>{cw}s}" for t in tn)+f"  {'PA':>6s}"
                w(hdr); w("      "+"-"*(22+len(tn)*cw+8))
                cm=r["cm_tr"]
                pa_arr=cm.diagonal()/cm.sum(axis=1).clip(1)
                ua_arr=cm.diagonal()/cm.sum(axis=0).clip(1)
                for i,name in enumerate(tn):
                    row="      "+f"{name:<20s}"+"".join(f"{cm[i,j]:>{cw}d}" for j in range(len(tn)))+f"  {pa_arr[i]:>5.1%}"
                    w(row)
                w("      "+"-"*(22+len(tn)*cw+8))
                w("      "+f"{'UA':<20s}"+"".join(f"{ua_arr[j]:>{cw}.1%}" for j in range(len(tn))))

                w(""); w("    Per-class report (sklearn):")
                for line in classification_report(r["y"],yp,labels=uniq,
                            target_names=tn,zero_division=0).splitlines():
                    w("      "+line)
            else:
                w(f"    Clusters  : {r['k']}")
                w(f"    Iterations: {r['n_iter']}")
                w(f"    {r['score_name']:10s}: {r['score']:.2f}")
                w(""); w("    Cluster coverage:")
                for i,c in enumerate(self.classes):
                    if i<len(r["pct"]):
                        w(f"      {c['name']:22s}  {r['pct'][i]:6.2f}%  ({r['counts'][i]:>10,} px)")

        w(""); w("4. RESOLUTION EFFECTS"); w("-"*50)
        if self.ls and self.st:
            lm=self.ls["meta"]; sm=self.st["meta"]
            lr=max(abs(lm["res"][0]),abs(lm["res"][1]))
            sr=max(abs(sm["res"][0]),abs(sm["res"][1]))
            w(f"\n  4.1  SPATIAL RESOLUTION")
            w(f"    Landsat    : {lr:.0f} m/pixel")
            w(f"    Sentinel-2 : {sr:.0f} m/pixel  (10/20/60 m by band)")
            if sr>0: w(f"    Ratio      : Sentinel-2 is ~{lr/sr:.1f}x finer")
            w("\n    Effect: Sentinel-2's finer pixels resolve small features")
            w("    (field edges, roads, small water) that Landsat merges")
            w("    into mixed pixels, reducing class separability.")
            w(f"\n  4.2  SPECTRAL RESOLUTION")
            w(f"    Landsat    : {lm['bands']} bands (VNIR + SWIR + Thermal)")
            w(f"    Sentinel-2 : {sm['bands']} bands (VNIR + 3 Red-Edge + SWIR)")
            w("\n    Effect: Sentinel-2's 3 Red-Edge bands (705-783 nm)")
            w("    enable finer vegetation separation. Landsat's Thermal")
            w("    bands (B10/B11) are unique - not available in Sentinel-2.")
            w("\n  4.3  RADIOMETRIC RESOLUTION")
            w("    Landsat    : 16-bit => 65,536 levels")
            w("    Sentinel-2 : 12-bit =>  4,096 levels")
            w("\n    Effect: Landsat's wider range avoids saturation on bright")
            w("    targets. Sentinel-2 L2A is atmospherically corrected to")
            w("    surface reflectance, improving index-based separation.")


        w(""); w("5. CONCLUSION"); w("-"*50)
        if "ls" in self.results and "st" in self.results and self.results["ls"]["supervised"]:
            lv=self.results["ls"]["cv"].mean(); sv=self.results["st"]["cv"].mean()
            better="Sentinel-2" if sv>=lv else "Landsat"
            w(f"\n  Landsat CV     : {lv:.4f}  ({lv:.2%})")
            w(f"  Sentinel-2 CV  : {sv:.4f}  ({sv:.2%})")
            w(f"  Better result  : {better}  (diff {abs(sv-lv):.2%})")
        w("")
        w("  - Sentinel-2 generally achieves higher accuracy for fine-scale")
        w("    land-cover due to better spatial (10m) and spectral resolution.")
        w("  - Landsat is essential for thermal analysis and long time-series.")
        w("  - In homogeneous scenes the gap narrows; in heterogeneous scenes")
        w("    (mixed agriculture/urban/forest) it widens substantially.")
        w("  - Fusing both sensors typically maximises classification accuracy.")
        w(""); w(sep)
        return "\n".join(L)

    def _set_status(self, msg, prog=None):
        def _do():
            self.status_var.set(msg)
            if prog is not None: self.prog["value"]=prog
        self.root.after(0,_do)


def main():
    root=tk.Tk()
    App(root)
    root.mainloop()

if __name__=="__main__":
    main()