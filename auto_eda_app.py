"""
Auto EDA, Feature Engineering, Visualization & Hypothesis Testing
Includes: SweetViz, YData Profiling (pandas-profiling), custom EDA, Hypothesis Testing

Run with:
    streamlit run auto_eda_app.py

Install:
    pip install streamlit pandas numpy matplotlib seaborn scipy scikit-learn plotly \
                sweetviz ydata-profiling streamlit-pandas-profiling

Notes:
  • SweetViz renders an interactive HTML report embedded inside Streamlit.
  • YData Profiling (the modern successor to pandas-profiling) renders a rich
    interactive HTML profile using streamlit-pandas-profiling.
  • Both reports are also available as HTML downloads.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.stats import (
    ttest_1samp, ttest_ind, ttest_rel,
    mannwhitneyu, wilcoxon, kruskal,
    chi2_contingency, f_oneway,
    pearsonr, spearmanr, normaltest,
    shapiro, levene, bartlett,
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OrdinalEncoder, PowerTransformer,
)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")
import io
import base64
import tempfile
import os

# ── Optional heavy profiling libraries (graceful degradation) ──
try:
    import sweetviz as sv
    SWEETVIZ_OK = True
except ImportError:
    SWEETVIZ_OK = False

try:
    from ydata_profiling import ProfileReport
    try:
        from streamlit_pandas_profiling import st_profile_report
        SPP_OK = True
    except ImportError:
        SPP_OK = False
    YDATA_OK = True
except ImportError:
    try:
        from pandas_profiling import ProfileReport  # legacy name
        try:
            from streamlit_pandas_profiling import st_profile_report
            SPP_OK = True
        except ImportError:
            SPP_OK = False
        YDATA_OK = True
    except ImportError:
        YDATA_OK = False
        SPP_OK   = False

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Auto EDA Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ═══════════════════════════════════════════════════
       CSS CUSTOM PROPERTIES — Light defaults
       Streamlit dark mode flips [data-theme="dark"] on
       the root, so we override vars there too.
    ═══════════════════════════════════════════════════ */
    :root {
        --eda-bg-card:        #f8fafc;
        --eda-border:         #e2e8f0;
        --eda-text-primary:   #0f172a;
        --eda-text-secondary: #475569;
        --eda-text-muted:     #64748b;
        --eda-tab-strip:      #f1f5f9;
        --eda-tab-active-bg:  #ffffff;
        --eda-tab-active-fg:  #0f172a;
        --eda-tab-idle-fg:    #475569;
        --eda-accent:         #6366f1;
        --eda-section-fg:     #0f172a;

        /* info box */
        --eda-info-bg:        #f0f9ff;
        --eda-info-border:    #93c5fd;
        --eda-info-fg:        #1e3a5f;

        /* warn box */
        --eda-warn-bg:        #fffbeb;
        --eda-warn-border:    #fcd34d;
        --eda-warn-fg:        #78350f;

        /* badges */
        --eda-badge-num-bg:   #dbeafe; --eda-badge-num-fg:   #1d4ed8;
        --eda-badge-cat-bg:   #fce7f3; --eda-badge-cat-fg:   #be185d;
        --eda-badge-dt-bg:    #d1fae5; --eda-badge-dt-fg:    #065f46;
        --eda-badge-bool-bg:  #fef3c7; --eda-badge-bool-fg:  #92400e;

        /* divider */
        --eda-divider:        #e2e8f0;

        /* sidebar */
        --eda-sidebar-bg:     #0f172a;
        --eda-sidebar-fg:     #e2e8f0;
        --eda-sidebar-muted:  #94a3b8;
    }

    /* ── Dark theme overrides ── */
    [data-theme="dark"] {
        --eda-bg-card:        #1e293b;
        --eda-border:         #334155;
        --eda-text-primary:   #f1f5f9;
        --eda-text-secondary: #cbd5e1;
        --eda-text-muted:     #94a3b8;
        --eda-tab-strip:      #1e293b;
        --eda-tab-active-bg:  #334155;
        --eda-tab-active-fg:  #f1f5f9;
        --eda-tab-idle-fg:    #94a3b8;
        --eda-section-fg:     #e2e8f0;

        --eda-info-bg:        #0f2847;
        --eda-info-border:    #3b82f6;
        --eda-info-fg:        #bfdbfe;

        --eda-warn-bg:        #2d1f00;
        --eda-warn-border:    #d97706;
        --eda-warn-fg:        #fde68a;

        --eda-badge-num-bg:   #1e3a6e; --eda-badge-num-fg:   #93c5fd;
        --eda-badge-cat-bg:   #4a1040; --eda-badge-cat-fg:   #f9a8d4;
        --eda-badge-dt-bg:    #064e3b; --eda-badge-dt-fg:    #6ee7b7;
        --eda-badge-bool-bg:  #451a03; --eda-badge-bool-fg:  #fde68a;

        --eda-divider:        #334155;
        --eda-sidebar-bg:     #020617;
        --eda-sidebar-fg:     #e2e8f0;
        --eda-sidebar-muted:  #64748b;
    }

    /* Fallback: prefers-color-scheme for browsers that don't expose data-theme */
    @media (prefers-color-scheme: dark) {
        :root {
            --eda-bg-card:        #1e293b;
            --eda-border:         #334155;
            --eda-text-primary:   #f1f5f9;
            --eda-text-secondary: #cbd5e1;
            --eda-text-muted:     #94a3b8;
            --eda-tab-strip:      #1e293b;
            --eda-tab-active-bg:  #334155;
            --eda-tab-active-fg:  #f1f5f9;
            --eda-tab-idle-fg:    #94a3b8;
            --eda-section-fg:     #e2e8f0;
            --eda-info-bg:        #0f2847;
            --eda-info-border:    #3b82f6;
            --eda-info-fg:        #bfdbfe;
            --eda-warn-bg:        #2d1f00;
            --eda-warn-border:    #d97706;
            --eda-warn-fg:        #fde68a;
            --eda-badge-num-bg:   #1e3a6e; --eda-badge-num-fg:   #93c5fd;
            --eda-badge-cat-bg:   #4a1040; --eda-badge-cat-fg:   #f9a8d4;
            --eda-badge-dt-bg:    #064e3b; --eda-badge-dt-fg:    #6ee7b7;
            --eda-badge-bool-bg:  #451a03; --eda-badge-bool-fg:  #fde68a;
            --eda-divider:        #334155;
        }
    }

    /* ══ Sidebar ══ */
    [data-testid="stSidebar"] {
        background: var(--eda-sidebar-bg) !important;
    }
    [data-testid="stSidebar"] * {
        color: var(--eda-sidebar-fg) !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stFileUploader label {
        color: var(--eda-sidebar-muted) !important;
        font-size: 0.78rem;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stCaption {
        color: var(--eda-sidebar-muted) !important;
    }

    /* ══ Metric cards ══ */
    [data-testid="stMetric"] {
        background:    var(--eda-bg-card);
        border:        1px solid var(--eda-border);
        border-radius: 10px;
        padding:       12px 16px;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.78rem;
        color: var(--eda-text-muted) !important;
    }
    [data-testid="stMetricValue"] {
        font-size:   1.5rem;
        font-weight: 700;
        color: var(--eda-text-primary) !important;
    }
    [data-testid="stMetricDelta"] {
        color: var(--eda-text-secondary) !important;
    }

    /* ══ Tab strip ══ */
    .stTabs [data-baseweb="tab-list"] {
        gap:           4px;
        background:    var(--eda-tab-strip);
        padding:       4px 6px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background:  transparent;
        border-radius: 7px;
        color:       var(--eda-tab-idle-fg) !important;
        font-weight: 500;
        padding:     6px 18px;
    }
    .stTabs [aria-selected="true"] {
        background: var(--eda-tab-active-bg) !important;
        color:      var(--eda-tab-active-fg) !important;
        box-shadow: 0 1px 4px rgba(0,0,0,.15);
    }

    /* ══ Section titles ══ */
    .section-title {
        font-size:   1.05rem;
        font-weight: 700;
        color:       var(--eda-section-fg) !important;
        border-left: 4px solid var(--eda-accent);
        padding-left: 10px;
        margin:      1.2rem 0 0.6rem;
    }

    /* ══ Badges ══ */
    .badge {
        display:       inline-block;
        padding:       2px 10px;
        border-radius: 20px;
        font-size:     0.72rem;
        font-weight:   600;
        margin:        2px;
    }
    .badge-num  { background: var(--eda-badge-num-bg);  color: var(--eda-badge-num-fg); }
    .badge-cat  { background: var(--eda-badge-cat-bg);  color: var(--eda-badge-cat-fg); }
    .badge-dt   { background: var(--eda-badge-dt-bg);   color: var(--eda-badge-dt-fg);  }
    .badge-bool { background: var(--eda-badge-bool-bg); color: var(--eda-badge-bool-fg);}

    /* ══ Info / Warning boxes ══ */
    .info-box {
        background:    var(--eda-info-bg);
        border:        1px solid var(--eda-info-border);
        border-radius: 8px;
        padding:       10px 16px;
        font-size:     0.85rem;
        color:         var(--eda-info-fg) !important;
        margin:        8px 0;
    }
    .info-box * { color: var(--eda-info-fg) !important; }

    .warn-box {
        background:    var(--eda-warn-bg);
        border:        1px solid var(--eda-warn-border);
        border-radius: 8px;
        padding:       10px 16px;
        font-size:     0.85rem;
        color:         var(--eda-warn-fg) !important;
        margin:        8px 0;
    }
    .warn-box * { color: var(--eda-warn-fg) !important; }
    .warn-box code, .info-box code {
        background: rgba(255,255,255,0.12);
        padding:    1px 5px;
        border-radius: 4px;
        font-size:  0.82rem;
    }

    /* ══ Hypothesis result box ══ */
    .result-box {
        background:    var(--eda-bg-card);
        border:        1px solid var(--eda-border);
        border-radius: 10px;
        padding:       16px 20px;
        margin-top:    12px;
    }
    .result-box p { color: var(--eda-text-primary) !important; }

    /* ══ Download button ══ */
    .stDownloadButton button {
        background:    var(--eda-accent);
        color:         #fff !important;
        border:        none;
        border-radius: 7px;
        font-weight:   600;
    }
    .stDownloadButton button:hover { background: #4f46e5; }

    /* ══ Dataframe / Table text ══ */
    [data-testid="stDataFrame"] * {
        color: var(--eda-text-primary) !important;
    }

    /* ══ Widget labels (selectbox, slider, radio, etc.) ══ */
    .stSelectbox label,
    .stMultiSelect label,
    .stSlider label,
    .stRadio label,
    .stCheckbox label,
    .stNumberInput label,
    .stTextInput label,
    .stTextArea label,
    .stFileUploader label,
    label[data-testid="stWidgetLabel"] {
        color: var(--eda-text-secondary) !important;
        font-weight: 500;
    }

    /* ══ st.subheader / st.markdown text ══ */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: var(--eda-text-primary) !important;
    }
    .stMarkdown p, .stMarkdown li {
        color: var(--eda-text-secondary) !important;
    }

    /* ══ Expander ══ */
    details { border-radius: 8px !important; }
    summary { color: var(--eda-text-primary) !important; font-weight: 600; }

    /* ══ Divider ══ */
    hr { border-top: 1px solid var(--eda-divider); margin: 1rem 0; }

    /* ══ Code blocks ══ */
    code {
        background: var(--eda-bg-card) !important;
        color: var(--eda-accent) !important;
        border: 1px solid var(--eda-border);
        border-radius: 4px;
        padding: 1px 5px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_sample(name: str) -> pd.DataFrame:
    samples = {
        "Titanic":   "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "Iris":      "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        "Tips":      "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
        "Diamonds":  "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv",
        "Penguins":  "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv",
        "MPG":       "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv",
    }
    return pd.read_csv(samples[name])


def classify_cols(df: pd.DataFrame):
    num_cols  = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols  = df.select_dtypes(include=["object", "category"]).columns.tolist()
    dt_cols   = df.select_dtypes(include=["datetime"]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    return num_cols, cat_cols, dt_cols, bool_cols


def df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


PALETTE = px.colors.qualitative.Set2


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Auto EDA Studio")
    st.markdown("---")

    data_source = st.radio("Data source", ["Upload CSV", "Sample Dataset"], horizontal=True)

    df = None
    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload your CSV", type=["csv"])
        if uploaded:
            sep = st.selectbox("Separator", [",", ";", "\t", "|"], index=0)
            df  = pd.read_csv(uploaded, sep=sep)
    else:
        sample_name = st.selectbox("Choose dataset", ["Titanic", "Iris", "Tips", "Diamonds", "Penguins", "MPG"])
        df = load_sample(sample_name)

    if df is not None:
        st.markdown(f"**Shape:** `{df.shape[0]:,} rows × {df.shape[1]} cols`")
        num_cols, cat_cols, dt_cols, bool_cols = classify_cols(df)

        st.markdown("**Column types**")
        for c in num_cols:
            st.markdown(f'<span class="badge badge-num">🔢 {c}</span>', unsafe_allow_html=True)
        for c in cat_cols:
            st.markdown(f'<span class="badge badge-cat">🔤 {c}</span>', unsafe_allow_html=True)
        for c in bool_cols:
            st.markdown(f'<span class="badge badge-bool">☑ {c}</span>', unsafe_allow_html=True)
        for c in dt_cols:
            st.markdown(f'<span class="badge badge-dt">📅 {c}</span>', unsafe_allow_html=True)

    st.markdown("---")
    # Library status
    st.markdown("**Library status**")
    sv_badge  = "🟢 SweetViz"      if SWEETVIZ_OK else "🔴 SweetViz (missing)"
    yp_badge  = "🟢 YData Profiling" if YDATA_OK  else "🔴 YData Profiling (missing)"
    spp_badge = "🟢 streamlit-pandas-profiling" if SPP_OK else "🟡 streamlit-pandas-profiling (missing)"
    st.caption(sv_badge)
    st.caption(yp_badge)
    if YDATA_OK:
        st.caption(spp_badge)
    st.markdown("---")
    st.caption("Auto EDA Studio v2.1")


# ─────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────
if df is None:
    st.markdown("""
    <div style='text-align:center; padding: 80px 0;'>
        <h1 style='font-size:3rem;'>🔬 Auto EDA Studio</h1>
        <p style='font-size:1.2rem; color:#64748b;'>
            Upload a CSV or pick a sample dataset from the sidebar to begin.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

num_cols, cat_cols, dt_cols, bool_cols = classify_cols(df)

tabs = st.tabs([
    "📊 Overview",
    "📈 Univariate",
    "🔗 Bivariate",
    "🌐 Multivariate",
    "🛠 Feature Engineering",
    "🧪 Hypothesis Testing",
    "🍬 SweetViz",
    "🐼 YData Profiling",
    "📤 Export",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 ── OVERVIEW
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<p class="section-title">Dataset at a Glance</p>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows",        f"{df.shape[0]:,}")
    m2.metric("Columns",     df.shape[1])
    m3.metric("Numeric",     len(num_cols))
    m4.metric("Categorical", len(cat_cols))
    total_miss = df.isnull().sum().sum()
    m5.metric("Missing vals", f"{total_miss:,}  ({total_miss/df.size*100:.1f}%)")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="section-title">Preview (first 100 rows)</p>', unsafe_allow_html=True)
        st.dataframe(df.head(100), use_container_width=True, height=300)

    with c2:
        st.markdown('<p class="section-title">Descriptive Statistics</p>', unsafe_allow_html=True)
        st.dataframe(df.describe(include="all").T, use_container_width=True, height=300)

    st.markdown('<p class="section-title">Missing Values</p>', unsafe_allow_html=True)
    miss = df.isnull().sum().reset_index()
    miss.columns = ["Column", "Missing"]
    miss["Pct"] = (miss["Missing"] / len(df) * 100).round(2)
    miss = miss[miss["Missing"] > 0].sort_values("Missing", ascending=False)

    if miss.empty:
        st.markdown('<div class="info-box">✅ No missing values found.</div>', unsafe_allow_html=True)
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(miss, use_container_width=True)
        with c2:
            fig = px.bar(
                miss, x="Column", y="Pct", text="Pct",
                color="Pct", color_continuous_scale="Reds",
                labels={"Pct": "Missing %"},
                title="Missing % per column",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                              margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-title">Data Types & Unique Counts</p>', unsafe_allow_html=True)
    dtype_df = pd.DataFrame({
        "Column":  df.columns,
        "Dtype":   df.dtypes.values.astype(str),
        "Non-null": df.notnull().sum().values,
        "Null":    df.isnull().sum().values,
        "Unique":  df.nunique().values,
        "Sample":  [str(df[c].dropna().iloc[0]) if df[c].dropna().shape[0] > 0 else "—" for c in df.columns],
    })
    st.dataframe(dtype_df, use_container_width=True)

    st.markdown('<p class="section-title">Duplicate Rows</p>', unsafe_allow_html=True)
    n_dup = df.duplicated().sum()
    if n_dup:
        st.markdown(f'<div class="warn-box">⚠️ Found <b>{n_dup}</b> duplicate rows ({n_dup/len(df)*100:.2f}%)</div>', unsafe_allow_html=True)
        if st.button("Show duplicates"):
            st.dataframe(df[df.duplicated(keep=False)].head(200), use_container_width=True)
    else:
        st.markdown('<div class="info-box">✅ No duplicate rows.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 ── UNIVARIATE
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<p class="section-title">Univariate Analysis</p>', unsafe_allow_html=True)

    # ── Numeric
    if num_cols:
        st.subheader("Numeric columns")
        sel_num = st.selectbox("Select numeric column", num_cols, key="uni_num")
        s = df[sel_num].dropna()

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Mean",   f"{s.mean():.4g}")
        c2.metric("Median", f"{s.median():.4g}")
        c3.metric("Std",    f"{s.std():.4g}")
        c4.metric("Skew",   f"{s.skew():.3f}")
        c5.metric("Kurt",   f"{s.kurt():.3f}")
        c6.metric("IQR",    f"{s.quantile(.75)-s.quantile(.25):.4g}")

        plot_type = st.radio("Plot type", ["Histogram + KDE", "Box + Violin", "ECDF", "Q-Q Plot"], horizontal=True)
        bins = st.slider("Bins (Histogram)", 10, 100, 30) if plot_type == "Histogram + KDE" else 30

        fig, axes = plt.subplots(1, 1 if plot_type not in ["Box + Violin"] else 2,
                                 figsize=(10, 4))
        if plot_type == "Histogram + KDE":
            ax = axes if not isinstance(axes, np.ndarray) else axes[0]
            sns.histplot(s, bins=bins, kde=True, ax=ax, color="#6366f1", edgecolor="white")
            ax.axvline(s.mean(),   color="#ef4444", ls="--", lw=1.5, label=f"Mean {s.mean():.2f}")
            ax.axvline(s.median(), color="#10b981", ls=":",  lw=1.5, label=f"Median {s.median():.2f}")
            ax.legend(fontsize=9)
            ax.set_title(f"Distribution of {sel_num}")

        elif plot_type == "Box + Violin":
            sns.violinplot(y=s, ax=axes[0], color="#818cf8", inner=None)
            axes[0].set_title("Violin")
            sns.boxplot(y=s, ax=axes[1], color="#f0abfc",
                        flierprops=dict(marker="o", markersize=3, alpha=.5))
            axes[1].set_title("Box")

        elif plot_type == "ECDF":
            ax = axes
            x_sorted = np.sort(s)
            y_ecdf   = np.arange(1, len(x_sorted)+1) / len(x_sorted)
            ax.step(x_sorted, y_ecdf, color="#6366f1", lw=2)
            ax.set_xlabel(sel_num); ax.set_ylabel("Cumulative probability")
            ax.set_title(f"ECDF — {sel_num}")

        else:  # Q-Q
            ax = axes
            (osm, osr), (slope, intercept, r) = stats.probplot(s, dist="norm")
            ax.scatter(osm, osr, s=15, color="#6366f1", alpha=.7)
            ax.plot(osm, slope * np.array(osm) + intercept, "r--", lw=1.5)
            ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Sample quantiles")
            ax.set_title(f"Q-Q Plot — {sel_num}  (r={r:.4f})")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Outlier summary
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        n_out = ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()
        st.markdown(f'<div class="info-box">📌 IQR outliers: <b>{n_out}</b> ({n_out/len(s)*100:.1f}%)'
                    f'  |  Lower fence: {q1-1.5*iqr:.3g}  |  Upper fence: {q3+1.5*iqr:.3g}</div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # ── Categorical
    if cat_cols:
        st.subheader("Categorical columns")
        sel_cat = st.selectbox("Select categorical column", cat_cols, key="uni_cat")
        vc = df[sel_cat].value_counts().reset_index()
        vc.columns = ["Value", "Count"]
        vc["Pct"] = (vc["Count"] / len(df) * 100).round(2)
        top_n = st.slider("Show top N categories", 5, min(50, len(vc)), min(20, len(vc)))
        vc_top = vc.head(top_n)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(vc_top, x="Value", y="Count", text="Pct",
                         color="Count", color_continuous_scale="Purples",
                         title=f"Top {top_n} — {sel_cat}")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.pie(vc_top, names="Value", values="Count",
                         title=f"Proportion — {sel_cat}", hole=0.4,
                         color_discrete_sequence=PALETTE)
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(vc, use_container_width=True)

    # ── All numeric distributions at once
    if num_cols and st.checkbox("Show all numeric distributions grid"):
        n_cols_plot = 3
        n_rows_plot = (len(num_cols) + n_cols_plot - 1) // n_cols_plot
        fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(14, 3.5 * n_rows_plot))
        axes = axes.flatten()
        for i, col in enumerate(num_cols):
            sns.histplot(df[col].dropna(), bins=25, kde=True, ax=axes[i],
                         color="#6366f1", edgecolor="white")
            axes[i].set_title(col, fontsize=10)
            axes[i].set_xlabel("")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle("All Numeric Distributions", y=1.01, fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════
# TAB 3 ── BIVARIATE
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<p class="section-title">Bivariate Analysis</p>', unsafe_allow_html=True)

    bv_type = st.radio(
        "Analysis type",
        ["Num × Num", "Num × Cat", "Cat × Cat", "Correlation Heatmap"],
        horizontal=True,
    )

    # ── Num × Num
    if bv_type == "Num × Num":
        if len(num_cols) < 2:
            st.info("Need at least 2 numeric columns.")
        else:
            c1, c2 = st.columns(2)
            x_col = c1.selectbox("X axis", num_cols, key="bv_x")
            y_col = c2.selectbox("Y axis", [c for c in num_cols if c != x_col], key="bv_y")
            color_col = st.selectbox("Color by (optional)", ["None"] + cat_cols, key="bv_col")
            plot_kind = st.radio("Plot", ["Scatter", "Hex-bin", "2D KDE"], horizontal=True)

            sub = df[[x_col, y_col] + ([color_col] if color_col != "None" else [])].dropna()
            r, p = pearsonr(sub[x_col], sub[y_col])
            rho, p2 = spearmanr(sub[x_col], sub[y_col])
            c1, c2 = st.columns(2)
            c1.metric("Pearson r",  f"{r:.4f}",  f"p={p:.2e}")
            c2.metric("Spearman ρ", f"{rho:.4f}", f"p={p2:.2e}")

            if plot_kind == "Scatter":
                fig = px.scatter(
                    sub, x=x_col, y=y_col,
                    color=color_col if color_col != "None" else None,
                    trendline="ols",
                    opacity=0.65,
                    color_discrete_sequence=PALETTE,
                )
            elif plot_kind == "Hex-bin":
                fig, ax = plt.subplots(figsize=(7, 5))
                hb = ax.hexbin(sub[x_col], sub[y_col], gridsize=30, cmap="Purples", mincnt=1)
                plt.colorbar(hb, ax=ax, label="Count")
                ax.set_xlabel(x_col); ax.set_ylabel(y_col)
                st.pyplot(fig); plt.close(fig); fig = None
            else:
                fig = px.density_contour(sub, x=x_col, y=y_col,
                                         color=color_col if color_col != "None" else None,
                                         color_discrete_sequence=PALETTE)
                fig.update_traces(contours_coloring="fill", contours_showlabels=True)

            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

    # ── Num × Cat
    elif bv_type == "Num × Cat":
        if not num_cols or not cat_cols:
            st.info("Need at least 1 numeric and 1 categorical column.")
        else:
            n_col = st.selectbox("Numeric column", num_cols, key="bv_nc")
            c_col = st.selectbox("Categorical column", cat_cols, key="bv_cc")
            kind  = st.radio("Plot", ["Box", "Violin", "Strip + Box", "Bar (mean ± CI)"], horizontal=True)

            sub = df[[n_col, c_col]].dropna()
            cats = sub[c_col].value_counts().index[:20].tolist()
            sub  = sub[sub[c_col].isin(cats)]

            if kind == "Box":
                fig = px.box(sub, x=c_col, y=n_col, color=c_col,
                             color_discrete_sequence=PALETTE, points="outliers")
            elif kind == "Violin":
                fig = px.violin(sub, x=c_col, y=n_col, color=c_col,
                                color_discrete_sequence=PALETTE, box=True)
            elif kind == "Strip + Box":
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.boxplot(data=sub, x=c_col, y=n_col, ax=ax,
                            palette="Set2", order=cats, fliersize=0)
                sns.stripplot(data=sub, x=c_col, y=n_col, ax=ax,
                              palette="Set2", order=cats, size=3, alpha=0.4, jitter=True)
                plt.xticks(rotation=30, ha="right")
                st.pyplot(fig); plt.close(fig); fig = None
            else:
                grp = sub.groupby(c_col)[n_col].agg(["mean","sem"]).reset_index()
                grp.columns = [c_col, "mean","sem"]
                fig = px.bar(grp, x=c_col, y="mean", error_y="sem",
                             color=c_col, color_discrete_sequence=PALETTE,
                             title=f"Mean ± SEM of {n_col} by {c_col}")

            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

    # ── Cat × Cat
    elif bv_type == "Cat × Cat":
        if len(cat_cols) < 2:
            st.info("Need at least 2 categorical columns.")
        else:
            c1, c2 = st.columns(2)
            ca = c1.selectbox("Column A", cat_cols, key="bv_ca")
            cb = c2.selectbox("Column B", [c for c in cat_cols if c != ca], key="bv_cb")
            sub = df[[ca, cb]].dropna()
            ct  = pd.crosstab(sub[ca], sub[cb])

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            sns.heatmap(ct,  ax=axes[0], annot=True, fmt="d", cmap="Purples",
                        linewidths=.5, cbar=False)
            axes[0].set_title("Contingency table (counts)")
            ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
            sns.heatmap(ct_pct, ax=axes[1], annot=True, fmt=".1f", cmap="Blues",
                        linewidths=.5, cbar=False)
            axes[1].set_title("Row percentages")
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)

            chi2, p, dof, _ = chi2_contingency(ct)
            st.markdown(f'<div class="info-box">χ² = <b>{chi2:.3f}</b> | df = {dof} | p = <b>{p:.4e}</b>'
                        f' — {"Significant" if p < 0.05 else "Not significant"} at α=0.05</div>',
                        unsafe_allow_html=True)

    # ── Correlation heatmap
    else:
        if len(num_cols) < 2:
            st.info("Need at least 2 numeric columns.")
        else:
            method = st.radio("Correlation method", ["pearson", "spearman", "kendall"], horizontal=True)
            corr = df[num_cols].corr(method=method)
            mask = np.triu(np.ones_like(corr, dtype=bool))
            fig, ax = plt.subplots(figsize=(max(8, len(num_cols)*.7), max(6, len(num_cols)*.6)))
            cmap = sns.diverging_palette(250, 10, as_cmap=True)
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap,
                        vmin=-1, vmax=1, center=0, square=True, ax=ax,
                        linewidths=.4, cbar_kws={"shrink": .7}, annot_kws={"size": 8})
            ax.set_title(f"{method.capitalize()} Correlation Heatmap", fontsize=13, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)

            st.markdown('<p class="section-title">Top Correlated Pairs</p>', unsafe_allow_html=True)
            corr_pairs = (corr.where(~mask)
                          .stack()
                          .reset_index()
                          .rename(columns={"level_0": "Col A", "level_1": "Col B", 0: "Correlation"}))
            corr_pairs["Abs"] = corr_pairs["Correlation"].abs()
            st.dataframe(corr_pairs.sort_values("Abs", ascending=False)
                         .drop("Abs", axis=1)
                         .reset_index(drop=True)
                         .head(30),
                         use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 ── MULTIVARIATE
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<p class="section-title">Multivariate Analysis</p>', unsafe_allow_html=True)

    mv_type = st.radio(
        "Method",
        ["Pair Plot", "Parallel Coordinates", "PCA", "Feature Importance", "Scatter Matrix 3D"],
        horizontal=True,
    )

    # ── Pair Plot
    if mv_type == "Pair Plot":
        max_cols = st.slider("Max numeric columns to include", 2, min(10, len(num_cols)), min(5, len(num_cols)))
        sel_pp = st.multiselect("Select columns", num_cols, default=num_cols[:max_cols])
        hue_pp = st.selectbox("Hue (color)", ["None"] + cat_cols, key="pp_hue")
        if len(sel_pp) < 2:
            st.info("Select at least 2 columns.")
        else:
            sub = df[sel_pp + ([hue_pp] if hue_pp != "None" else [])].dropna()
            fig = sns.pairplot(sub,
                               hue=hue_pp if hue_pp != "None" else None,
                               diag_kind="kde", plot_kws={"alpha": 0.5, "s": 20},
                               palette="Set2")
            fig.fig.suptitle("Pair Plot", y=1.01, fontsize=13, fontweight="bold")
            st.pyplot(fig.fig); plt.close("all")

    # ── Parallel Coordinates
    elif mv_type == "Parallel Coordinates":
        sel_pc = st.multiselect("Numeric columns", num_cols, default=num_cols[:6])
        hue_pc = st.selectbox("Color by", ["None"] + cat_cols + num_cols, key="pc_hue")
        if len(sel_pc) < 2:
            st.info("Select at least 2 columns.")
        else:
            sub = df[list(set(sel_pc + ([hue_pc] if hue_pc != "None" else [])))].dropna()
            if hue_pc != "None" and hue_pc in cat_cols:
                sub["_code"] = sub[hue_pc].astype("category").cat.codes
                color_col_pc = "_code"
            elif hue_pc != "None":
                color_col_pc = hue_pc
            else:
                color_col_pc = sel_pc[0]
            fig = px.parallel_coordinates(sub, dimensions=sel_pc, color=color_col_pc,
                                          color_continuous_scale=px.colors.sequential.Viridis)
            st.plotly_chart(fig, use_container_width=True)

    # ── PCA
    elif mv_type == "PCA":
        if len(num_cols) < 3:
            st.info("Need at least 3 numeric columns.")
        else:
            sel_pca = st.multiselect("Columns for PCA", num_cols, default=num_cols)
            hue_pca = st.selectbox("Color by", ["None"] + cat_cols, key="pca_hue")
            n_comp  = st.slider("Number of components", 2, min(len(sel_pca), 10), min(5, len(sel_pca)))
            if len(sel_pca) < 2:
                st.info("Select at least 2 columns.")
            else:
                sub = df[sel_pca + ([hue_pca] if hue_pca != "None" else [])].dropna()
                Xs  = StandardScaler().fit_transform(sub[sel_pca])
                pca = PCA(n_components=n_comp)
                comps = pca.fit_transform(Xs)
                ev = pca.explained_variance_ratio_ * 100

                pca_df = pd.DataFrame(comps, columns=[f"PC{i+1}" for i in range(n_comp)])
                if hue_pca != "None":
                    pca_df[hue_pca] = sub[hue_pca].values

                c1, c2 = st.columns(2)
                with c1:
                    fig = px.scatter(pca_df, x="PC1", y="PC2",
                                     color=hue_pca if hue_pca != "None" else None,
                                     color_discrete_sequence=PALETTE,
                                     title=f"PCA — PC1 ({ev[0]:.1f}%) vs PC2 ({ev[1]:.1f}%)",
                                     opacity=0.7)
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig2 = px.bar(x=[f"PC{i+1}" for i in range(n_comp)],
                                  y=ev, text=[f"{v:.1f}%" for v in ev],
                                  title="Explained Variance per PC",
                                  labels={"x": "Component", "y": "Variance %"})
                    fig2.update_traces(marker_color="#6366f1", textposition="outside")
                    fig2.update_layout(showlegend=False)
                    st.plotly_chart(fig2, use_container_width=True)

                # Loadings
                loadings = pd.DataFrame(
                    pca.components_.T,
                    index=sel_pca,
                    columns=[f"PC{i+1}" for i in range(n_comp)],
                )
                fig3, ax = plt.subplots(figsize=(8, max(4, len(sel_pca) * .35)))
                sns.heatmap(loadings, annot=True, fmt=".2f", cmap="RdBu_r",
                            vmin=-1, vmax=1, center=0, ax=ax, linewidths=.4,
                            annot_kws={"size": 8})
                ax.set_title("PCA Loadings")
                plt.tight_layout()
                st.pyplot(fig3); plt.close(fig3)

    # ── Feature Importance
    elif mv_type == "Feature Importance":
        if not num_cols:
            st.info("Need numeric columns.")
        else:
            target = st.selectbox("Target column", df.columns.tolist(), key="fi_target")
            feat_cols = [c for c in num_cols if c != target]
            if not feat_cols:
                st.info("No feature columns available.")
            else:
                sub = df[[target] + feat_cols].dropna()
                X   = sub[feat_cols].values
                y   = sub[target].values
                is_clf = sub[target].nunique() <= 20 and sub[target].dtype == object
                if is_clf:
                    le = LabelEncoder()
                    y  = le.fit_transform(y)
                    mdl = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    try:
                        y = y.astype(float)
                        mdl = RandomForestRegressor(n_estimators=100, random_state=42)
                    except Exception:
                        st.error("Target must be numeric or have ≤20 categories.")
                        st.stop()

                with st.spinner("Training Random Forest…"):
                    mdl.fit(X, y)

                imp_df = pd.DataFrame({
                    "Feature":   feat_cols,
                    "Importance": mdl.feature_importances_,
                }).sort_values("Importance", ascending=True)

                fig = px.bar(imp_df, y="Feature", x="Importance", orientation="h",
                             color="Importance", color_continuous_scale="Purples",
                             title=f"Random Forest Feature Importance → {target}")
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

    # ── 3D Scatter
    else:
        if len(num_cols) < 3:
            st.info("Need at least 3 numeric columns.")
        else:
            c1, c2, c3 = st.columns(3)
            x3 = c1.selectbox("X", num_cols, key="3d_x")
            y3 = c2.selectbox("Y", [c for c in num_cols if c != x3], key="3d_y")
            z3 = c3.selectbox("Z", [c for c in num_cols if c not in [x3, y3]], key="3d_z")
            hue3 = st.selectbox("Color by", ["None"] + cat_cols + num_cols, key="3d_hue")
            sub = df[[x3, y3, z3] + ([hue3] if hue3 != "None" else [])].dropna()
            fig = px.scatter_3d(sub, x=x3, y=y3, z=z3,
                                color=hue3 if hue3 != "None" else None,
                                opacity=0.65, color_discrete_sequence=PALETTE)
            fig.update_traces(marker_size=3)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 5 ── FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<p class="section-title">Feature Engineering</p>', unsafe_allow_html=True)

    if "fe_df" not in st.session_state:
        st.session_state.fe_df = df.copy()

    fe_df = st.session_state.fe_df
    fe_num_cols, fe_cat_cols, _, _ = classify_cols(fe_df)

    fe_type = st.radio(
        "Operation",
        ["Encoding", "Scaling / Normalization", "Missing Value Imputation",
         "Outlier Treatment", "Transformations", "New Feature"],
        horizontal=True,
    )

    # ── Encoding
    if fe_type == "Encoding":
        sel_enc = st.multiselect("Categorical columns to encode", fe_cat_cols)
        enc_method = st.radio("Method", ["Label Encoding", "One-Hot Encoding", "Frequency Encoding",
                                         "Target Encoding (requires target)"], horizontal=True)
        target_enc = None
        if enc_method == "Target Encoding (requires target)":
            target_enc = st.selectbox("Target column (numeric)", fe_num_cols)
        if sel_enc and st.button("Apply Encoding"):
            for col in sel_enc:
                if enc_method == "Label Encoding":
                    fe_df[f"{col}_le"] = LabelEncoder().fit_transform(fe_df[col].astype(str))
                elif enc_method == "One-Hot Encoding":
                    dummies = pd.get_dummies(fe_df[col], prefix=col)
                    fe_df   = pd.concat([fe_df, dummies], axis=1)
                elif enc_method == "Frequency Encoding":
                    freq    = fe_df[col].value_counts(normalize=True)
                    fe_df[f"{col}_freq"] = fe_df[col].map(freq)
                else:
                    if target_enc:
                        mean_target = fe_df.groupby(col)[target_enc].mean()
                        fe_df[f"{col}_te"] = fe_df[col].map(mean_target)
            st.session_state.fe_df = fe_df
            st.success("Encoding applied ✓")

    # ── Scaling
    elif fe_type == "Scaling / Normalization":
        sel_sc  = st.multiselect("Numeric columns to scale", fe_num_cols)
        sc_meth = st.radio("Method",
                           ["StandardScaler (z-score)", "MinMaxScaler (0–1)",
                            "RobustScaler (IQR)", "Log1p", "Sqrt"],
                           horizontal=True)
        if sel_sc and st.button("Apply Scaling"):
            if sc_meth in ["StandardScaler (z-score)", "MinMaxScaler (0–1)", "RobustScaler (IQR)"]:
                scaler = {"StandardScaler (z-score)": StandardScaler(),
                          "MinMaxScaler (0–1)": MinMaxScaler(),
                          "RobustScaler (IQR)": RobustScaler()}[sc_meth]
                scaled = scaler.fit_transform(fe_df[sel_sc])
                suffix = {"StandardScaler (z-score)": "_zscore",
                          "MinMaxScaler (0–1)": "_minmax",
                          "RobustScaler (IQR)": "_robust"}[sc_meth]
                fe_df[[f"{c}{suffix}" for c in sel_sc]] = scaled
            elif sc_meth == "Log1p":
                for c in sel_sc:
                    fe_df[f"{c}_log1p"] = np.log1p(fe_df[c].clip(lower=0))
            else:
                for c in sel_sc:
                    fe_df[f"{c}_sqrt"] = np.sqrt(fe_df[c].clip(lower=0))
            st.session_state.fe_df = fe_df
            st.success("Scaling applied ✓")

    # ── Imputation
    elif fe_type == "Missing Value Imputation":
        miss_cols = [c for c in fe_df.columns if fe_df[c].isnull().any()]
        if not miss_cols:
            st.markdown('<div class="info-box">✅ No missing values in the current dataframe.</div>',
                        unsafe_allow_html=True)
        else:
            sel_imp  = st.multiselect("Columns to impute", miss_cols, default=miss_cols)
            imp_meth = st.radio("Method",
                                ["Mean", "Median", "Mode", "Constant", "Forward fill", "Backward fill"],
                                horizontal=True)
            const_val = st.text_input("Constant value", "0") if imp_meth == "Constant" else None
            if sel_imp and st.button("Apply Imputation"):
                for c in sel_imp:
                    if   imp_meth == "Mean":           fe_df[c].fillna(fe_df[c].mean(),            inplace=True)
                    elif imp_meth == "Median":         fe_df[c].fillna(fe_df[c].median(),          inplace=True)
                    elif imp_meth == "Mode":           fe_df[c].fillna(fe_df[c].mode()[0],         inplace=True)
                    elif imp_meth == "Constant":       fe_df[c].fillna(const_val,                  inplace=True)
                    elif imp_meth == "Forward fill":   fe_df[c].fillna(method="ffill",             inplace=True)
                    else:                              fe_df[c].fillna(method="bfill",             inplace=True)
                st.session_state.fe_df = fe_df
                st.success("Imputation applied ✓")

    # ── Outlier treatment
    elif fe_type == "Outlier Treatment":
        sel_out  = st.multiselect("Numeric columns", fe_num_cols)
        out_meth = st.radio("Method", ["IQR Capping (Winsorize)", "Z-score Removal (|z|>3)", "Log Dampen"],
                            horizontal=True)
        if sel_out and st.button("Apply Outlier Treatment"):
            for c in sel_out:
                if out_meth == "IQR Capping (Winsorize)":
                    q1, q3 = fe_df[c].quantile([.25, .75])
                    iqr = q3 - q1
                    fe_df[c] = fe_df[c].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)
                elif out_meth == "Z-score Removal (|z|>3)":
                    z = (fe_df[c] - fe_df[c].mean()) / fe_df[c].std()
                    fe_df = fe_df[z.abs() <= 3]
                else:
                    fe_df[f"{c}_logdamp"] = np.log1p(np.abs(fe_df[c])) * np.sign(fe_df[c])
            st.session_state.fe_df = fe_df
            st.success("Outlier treatment applied ✓")

    # ── Transformations
    elif fe_type == "Transformations":
        sel_tr  = st.multiselect("Numeric columns", fe_num_cols)
        tr_meth = st.radio("Transform",
                           ["Box-Cox", "Yeo-Johnson", "Binning (quantile)", "Binning (uniform)"],
                           horizontal=True)
        n_bins  = st.slider("Bins", 2, 20, 5) if "Binning" in tr_meth else None
        if sel_tr and st.button("Apply Transformation"):
            for c in sel_tr:
                if tr_meth in ["Box-Cox", "Yeo-Johnson"]:
                    method_str = "box-cox" if tr_meth == "Box-Cox" else "yeo-johnson"
                    pt = PowerTransformer(method=method_str)
                    try:
                        fe_df[f"{c}_{method_str.replace('-','_')}"] = pt.fit_transform(fe_df[[c]].dropna())
                    except Exception as e:
                        st.warning(f"{c}: {e}")
                elif tr_meth == "Binning (quantile)":
                    fe_df[f"{c}_qbin"] = pd.qcut(fe_df[c], n_bins, labels=False, duplicates="drop")
                else:
                    fe_df[f"{c}_ubin"] = pd.cut(fe_df[c], n_bins, labels=False)
            st.session_state.fe_df = fe_df
            st.success("Transformation applied ✓")

    # ── New Feature
    else:
        st.markdown("Create a new column using a Python expression. Use `df` to reference the dataframe.")
        new_name = st.text_input("New column name", "new_feature")
        expr     = st.text_area("Expression (e.g. `df['col_a'] * df['col_b']`)",
                                height=80, value="df['" + (num_cols[0] if num_cols else "col") + "'] ** 2")
        if st.button("Create Feature"):
            try:
                fe_df[new_name] = eval(expr, {"df": fe_df, "np": np, "pd": pd})
                st.session_state.fe_df = fe_df
                st.success(f"Column '{new_name}' created ✓")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown(f"**Current engineered dataframe** — shape: `{st.session_state.fe_df.shape}`")
    st.dataframe(st.session_state.fe_df.head(50), use_container_width=True, height=250)

    if st.button("↩ Reset to original data"):
        st.session_state.fe_df = df.copy()
        st.experimental_rerun()


# ══════════════════════════════════════════════════════════════
# TAB 6 ── HYPOTHESIS TESTING
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<p class="section-title">Hypothesis Testing</p>', unsafe_allow_html=True)

    alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01)

    test_category = st.selectbox(
        "Test category",
        ["Normality Tests", "One-Sample t-test", "Two-Sample Tests",
         "Paired Tests", "One-Way ANOVA / Kruskal-Wallis",
         "Chi-Square Test", "Correlation Test", "Variance Tests"],
    )

    def show_result(stat_name, stat_val, p_val, alpha, extra=""):
        reject = p_val < alpha
        color  = "#f87171" if reject else "#4ade80"   # light red / light green — readable on both themes
        icon   = "❌ Reject H₀" if reject else "✅ Fail to reject H₀"
        st.markdown(f"""
        <div class="result-box">
            <p style="margin:0 0 6px; font-size:0.85rem; opacity:0.7;">Test statistic &amp; p-value</p>
            <p style="margin:0; font-size:1.3rem; font-weight:700;">
                {stat_name} = {stat_val:.5f} &nbsp;|&nbsp; p = {p_val:.5e}
            </p>
            <p style="margin:8px 0 0; font-size:1rem; font-weight:600; color:{color};">
                {icon} &nbsp;(α = {alpha})
            </p>
            {f'<p style="margin:6px 0 0; font-size:0.82rem; opacity:0.75;">{extra}</p>' if extra else ''}
        </div>
        """, unsafe_allow_html=True)

    # ── Normality
    if test_category == "Normality Tests":
        col = st.selectbox("Column", num_cols)
        s   = df[col].dropna()
        test_choice = st.radio("Test", ["Shapiro-Wilk (n≤5000)", "D'Agostino-Pearson", "Both"], horizontal=True)

        if st.button("Run Normality Test"):
            st.markdown(f"**H₀:** `{col}` is normally distributed")
            if test_choice in ["Shapiro-Wilk (n≤5000)", "Both"]:
                s_sw = s.sample(min(5000, len(s)), random_state=42)
                sw, p_sw = shapiro(s_sw)
                show_result("W (Shapiro-Wilk)", sw, p_sw, alpha)
            if test_choice in ["D'Agostino-Pearson", "Both"]:
                k2, p_dp = normaltest(s)
                show_result("K² (D'Agostino)", k2, p_dp, alpha)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            sns.histplot(s, bins=30, kde=True, ax=axes[0], color="#6366f1")
            axes[0].set_title(f"Distribution — {col}")
            (osm, osr), (slope, intercept, r) = stats.probplot(s, dist="norm")
            axes[1].scatter(osm, osr, s=15, color="#6366f1", alpha=.7)
            axes[1].plot(osm, slope * np.array(osm) + intercept, "r--", lw=1.5)
            axes[1].set_title(f"Q-Q Plot (r={r:.4f})")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── One-Sample t-test
    elif test_category == "One-Sample t-test":
        col = st.selectbox("Column", num_cols)
        mu0 = st.number_input("Hypothesised mean (μ₀)", value=float(df[col].mean()))
        alt = st.radio("Alternative", ["two-sided", "less", "greater"], horizontal=True)
        if st.button("Run Test"):
            s   = df[col].dropna()
            t, p = ttest_1samp(s, popmean=mu0, alternative=alt)
            alt_symbol = {'two-sided': '≠', 'less': '<', 'greater': '>'}[alt]
            st.markdown(f"**H₀:** μ = {mu0}  |  **H₁:** μ {alt_symbol} {mu0}")
            show_result("t", t, p, alpha,
                        extra=f"n={len(s)}  |  x̄={s.mean():.4f}  |  s={s.std():.4f}")

    # ── Two-Sample Tests
    elif test_category == "Two-Sample Tests":
        col   = st.selectbox("Numeric column", num_cols)
        grp   = st.selectbox("Group column", cat_cols)
        cats2 = df[grp].dropna().unique().tolist()
        g1    = st.selectbox("Group 1", cats2, index=0)
        g2    = st.selectbox("Group 2", [c for c in cats2 if c != g1], index=0)
        test2 = st.radio("Test", ["Welch t-test (normal)", "Mann-Whitney U (non-param)"], horizontal=True)
        alt2  = st.radio("Alternative", ["two-sided", "less", "greater"], horizontal=True)

        if st.button("Run Test"):
            a = df[df[grp] == g1][col].dropna()
            b = df[df[grp] == g2][col].dropna()
            st.markdown(f"**H₀:** {g1} and {g2} have equal {col}")

            if test2 == "Welch t-test (normal)":
                t, p = ttest_ind(a, b, equal_var=False, alternative=alt2)
                show_result("t (Welch)", t, p, alpha,
                            extra=f"n₁={len(a)} (x̄={a.mean():.3f})  |  n₂={len(b)} (x̄={b.mean():.3f})")
            else:
                u, p = mannwhitneyu(a, b, alternative=alt2)
                show_result("U (Mann-Whitney)", u, p, alpha,
                            extra=f"n₁={len(a)}  |  n₂={len(b)}")

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(a, bins=25, alpha=0.6, label=str(g1), color="#6366f1", edgecolor="white")
            ax.hist(b, bins=25, alpha=0.6, label=str(g2), color="#f59e0b", edgecolor="white")
            ax.legend(); ax.set_title(f"{col} by {grp}")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── Paired Tests
    elif test_category == "Paired Tests":
        col1 = st.selectbox("Column 1 (before)", num_cols, key="pt_c1")
        col2 = st.selectbox("Column 2 (after)",  [c for c in num_cols if c != col1], key="pt_c2")
        test_p = st.radio("Test", ["Paired t-test (normal)", "Wilcoxon signed-rank (non-param)"],
                          horizontal=True)
        if st.button("Run Paired Test"):
            sub  = df[[col1, col2]].dropna()
            diff = sub[col1] - sub[col2]
            st.markdown(f"**H₀:** mean({col1} − {col2}) = 0  |  n = {len(sub)}")
            if test_p == "Paired t-test (normal)":
                t, p = ttest_rel(sub[col1], sub[col2])
                show_result("t (Paired)", t, p, alpha,
                            extra=f"Mean diff = {diff.mean():.4f}  |  SD diff = {diff.std():.4f}")
            else:
                w, p = wilcoxon(sub[col1], sub[col2])
                show_result("W (Wilcoxon)", w, p, alpha)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(diff, bins=25, color="#6366f1", edgecolor="white", alpha=0.8)
            ax.axvline(0, color="red", lw=1.5, ls="--")
            ax.set_title(f"Paired difference — {col1} − {col2}")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── ANOVA / Kruskal-Wallis
    elif test_category == "One-Way ANOVA / Kruskal-Wallis":
        col_a  = st.selectbox("Numeric column", num_cols, key="anova_num")
        grp_a  = st.selectbox("Group column (≥ 2 groups)", cat_cols, key="anova_cat")
        test_a = st.radio("Test", ["One-Way ANOVA (normal)", "Kruskal-Wallis (non-param)"], horizontal=True)

        if st.button("Run Test"):
            groups = [g.values for _, g in df.groupby(grp_a)[col_a] if g.dropna().shape[0] > 1]
            st.markdown(f"**H₀:** All group means of `{col_a}` are equal across `{grp_a}` ({len(groups)} groups)")
            if test_a == "One-Way ANOVA (normal)":
                F, p = f_oneway(*groups)
                show_result("F (ANOVA)", F, p, alpha)
            else:
                H, p = kruskal(*groups)
                show_result("H (Kruskal-Wallis)", H, p, alpha)

            fig = px.box(df[[col_a, grp_a]].dropna(), x=grp_a, y=col_a, color=grp_a,
                         color_discrete_sequence=PALETTE)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── Chi-Square
    elif test_category == "Chi-Square Test":
        if len(cat_cols) < 2:
            st.info("Need at least 2 categorical columns.")
        else:
            ca = st.selectbox("Column A", cat_cols, key="chi_a")
            cb = st.selectbox("Column B", [c for c in cat_cols if c != ca], key="chi_b")
            if st.button("Run Chi-Square Test"):
                ct = pd.crosstab(df[ca].dropna(), df[cb].dropna())
                chi2, p, dof, exp = chi2_contingency(ct)
                st.markdown(f"**H₀:** `{ca}` and `{cb}` are independent")
                show_result("χ²", chi2, p, alpha, extra=f"df = {dof}  |  Cramér's V = {np.sqrt(chi2 / (len(df) * (min(ct.shape)-1))):.4f}")

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                sns.heatmap(ct,  annot=True, fmt="d", cmap="Purples", ax=axes[0],
                            linewidths=.5, cbar=False)
                axes[0].set_title("Observed counts")
                sns.heatmap(pd.DataFrame(exp, index=ct.index, columns=ct.columns),
                            annot=True, fmt=".1f", cmap="Blues", ax=axes[1],
                            linewidths=.5, cbar=False)
                axes[1].set_title("Expected counts")
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── Correlation Test
    elif test_category == "Correlation Test":
        if len(num_cols) < 2:
            st.info("Need at least 2 numeric columns.")
        else:
            x_c  = st.selectbox("Column X", num_cols, key="cor_x")
            y_c  = st.selectbox("Column Y", [c for c in num_cols if c != x_c], key="cor_y")
            meth = st.radio("Method", ["Pearson", "Spearman"], horizontal=True)
            if st.button("Run Correlation Test"):
                sub = df[[x_c, y_c]].dropna()
                if meth == "Pearson":
                    r, p = pearsonr(sub[x_c], sub[y_c])
                    show_result("r (Pearson)", r, p, alpha, extra=f"n={len(sub)}")
                else:
                    rho, p = spearmanr(sub[x_c], sub[y_c])
                    show_result("ρ (Spearman)", rho, p, alpha, extra=f"n={len(sub)}")

    # ── Variance Tests
    else:
        col_v  = st.selectbox("Numeric column", num_cols, key="var_col")
        grp_v  = st.selectbox("Group column", cat_cols, key="var_grp")
        test_v = st.radio("Test", ["Levene (robust)", "Bartlett (assumes normality)"], horizontal=True)
        if st.button("Run Variance Test"):
            groups = [g.dropna().values for _, g in df.groupby(grp_v)[col_v] if g.dropna().shape[0] > 1]
            st.markdown(f"**H₀:** All groups have equal variance in `{col_v}`")
            if test_v == "Levene (robust)":
                stat, p = levene(*groups)
                show_result("W (Levene)", stat, p, alpha)
            else:
                stat, p = bartlett(*groups)
                show_result("T (Bartlett)", stat, p, alpha)


# ══════════════════════════════════════════════════════════════
# TAB 7 ── SWEETVIZ
# ══════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<p class="section-title">SweetViz — Automated EDA Report</p>', unsafe_allow_html=True)

    if not SWEETVIZ_OK:
        st.markdown("""
        <div class="warn-box">
        ⚠️ <b>SweetViz not installed.</b><br>
        Run: <code>pip install sweetviz</code> then restart the app.
        </div>
        """, unsafe_allow_html=True)
    else:
        sv_col1, sv_col2 = st.columns(2)
        with sv_col1:
            sv_target = st.selectbox(
                "Target column (optional — leave 'None' for unsupervised report)",
                ["None"] + df.columns.tolist(),
                key="sv_target",
            )
        with sv_col2:
            sv_compare = st.checkbox("Compare train / test split", value=False, key="sv_compare")
            if sv_compare:
                test_frac = st.slider("Test fraction", 0.1, 0.5, 0.2, 0.05, key="sv_frac")

        sv_pairwise = st.checkbox("Show pairwise associations (slower for large datasets)", value=True)
        sv_mode = st.radio(
            "Report scope",
            ["Full dataset", "Engineered dataset"],
            horizontal=True,
            key="sv_mode",
        )

        st.markdown("""
        <div class="info-box">
        ℹ️ SweetViz generates a self-contained HTML report.
        It will be embedded below <b>and</b> available for download.
        Large datasets may take 10–30 seconds.
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Generate SweetViz Report", key="sv_run"):
            source_df = st.session_state.get("fe_df", df) if sv_mode == "Engineered dataset" else df
            target_arg = sv_target if sv_target != "None" else None

            with st.spinner("Generating SweetViz report…"):
                try:
                    sv.config_parser.read_string("[General]\nn_target_values_tail_cap = 10")

                    if sv_compare:
                        from sklearn.model_selection import train_test_split
                        train_sv, test_sv = train_test_split(
                            source_df, test_size=test_frac, random_state=42
                        )
                        report = sv.compare(
                            [train_sv, "Train"],
                            [test_sv,  "Test"],
                            target_feat=target_arg,
                            pairwise_analysis="on" if sv_pairwise else "off",
                        )
                    else:
                        report = sv.analyze(
                            source_df,
                            target_feat=target_arg,
                            pairwise_analysis="on" if sv_pairwise else "off",
                        )

                    # Save to a temp file and read back as HTML string
                    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                        tmp_path = tmp.name
                    report.show_html(filepath=tmp_path, open_browser=False, layout="widescreen", scale=1.0)

                    with open(tmp_path, "r", encoding="utf-8") as f:
                        sv_html = f.read()
                    os.unlink(tmp_path)

                    st.session_state["sv_html"] = sv_html
                    st.success("SweetViz report generated ✓")
                except Exception as e:
                    st.error(f"SweetViz error: {e}")

        if "sv_html" in st.session_state:
            sv_html = st.session_state["sv_html"]
            # Embed via components
            import streamlit.components.v1 as components
            st.markdown("#### Embedded Report")
            components.html(sv_html, height=900, scrolling=True)

            st.download_button(
                "⬇ Download SweetViz HTML",
                data=sv_html.encode("utf-8"),
                file_name="sweetviz_report.html",
                mime="text/html",
            )


# ══════════════════════════════════════════════════════════════
# TAB 8 ── YDATA PROFILING
# ══════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown('<p class="section-title">YData Profiling — Deep-Dive EDA Report</p>', unsafe_allow_html=True)

    if not YDATA_OK:
        st.markdown("""
        <div class="warn-box">
        ⚠️ <b>YData Profiling not installed.</b><br>
        Run: <code>pip install ydata-profiling streamlit-pandas-profiling</code> then restart.
        <br><br>
        If you have the legacy package, run: <code>pip install pandas-profiling streamlit-pandas-profiling</code>
        </div>
        """, unsafe_allow_html=True)
    else:
        yp_col1, yp_col2 = st.columns(2)
        with yp_col1:
            yp_mode = st.radio(
                "Report mode",
                ["minimal (fast)", "complete (detailed)", "explorative"],
                horizontal=True,
                key="yp_mode",
            )
        with yp_col2:
            yp_scope = st.radio(
                "Dataset scope",
                ["Full dataset", "Engineered dataset"],
                horizontal=True,
                key="yp_scope",
            )

        yp_sample = st.slider(
            "Row sample limit (0 = all rows — careful with large datasets)",
            0, min(50_000, len(df)), min(5_000, len(df)),
            step=500,
            key="yp_sample",
        )

        yp_correlations = st.checkbox("Compute correlations (slower)", value=True, key="yp_corr")
        yp_interactions = st.checkbox("Compute interactions / scatter matrix (slower)", value=False, key="yp_inter")
        yp_missing      = st.checkbox("Missing value analysis", value=True, key="yp_miss")
        yp_duplicates   = st.checkbox("Duplicate row analysis", value=True, key="yp_dup")

        st.markdown("""
        <div class="info-box">
        ℹ️ YData Profiling creates a comprehensive report with distribution plots, correlations,
        missing-value heatmaps, interactions, and more. <b>Complete</b> mode on large datasets
        can take several minutes — use <b>minimal</b> for a quick pass.
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Generate YData Profile", key="yp_run"):
            source_df = st.session_state.get("fe_df", df) if yp_scope == "Engineered dataset" else df

            if yp_sample > 0 and yp_sample < len(source_df):
                profile_df = source_df.sample(yp_sample, random_state=42).reset_index(drop=True)
                st.info(f"Profiling a {yp_sample:,}-row sample (full dataset has {len(source_df):,} rows).")
            else:
                profile_df = source_df

            mode_map = {
                "minimal (fast)":   True,
                "complete (detailed)": False,
                "explorative":      False,
            }
            explorative_flag = yp_mode == "explorative"

            with st.spinner("Generating YData profile… this may take a while ⏳"):
                try:
                    profile = ProfileReport(
                        profile_df,
                        title="YData Profiling Report — Auto EDA Studio",
                        minimal=mode_map[yp_mode],
                        explorative=explorative_flag,
                        correlations={
                            "pearson":  {"calculate": yp_correlations},
                            "spearman": {"calculate": yp_correlations},
                            "kendall":  {"calculate": False},
                            "phi_k":    {"calculate": False},
                            "cramers":  {"calculate": yp_correlations},
                        } if not mode_map[yp_mode] else None,
                        missing_diagrams={
                            "bar":    yp_missing,
                            "matrix": yp_missing,
                            "heatmap": yp_missing,
                        } if not mode_map[yp_mode] else None,
                        interactions={"continuous": yp_interactions} if not mode_map[yp_mode] else None,
                        duplicates={"head": 10} if yp_duplicates else None,
                        progress_bar=False,
                    )
                    st.session_state["ydata_profile"] = profile

                    # Also pre-render HTML for download
                    yp_html = profile.to_html()
                    st.session_state["ydata_html"] = yp_html
                    st.success("YData profile generated ✓")
                except Exception as e:
                    st.error(f"YData Profiling error: {e}")
                    st.info("Try switching to **minimal** mode or reducing the sample size.")

        if "ydata_profile" in st.session_state:
            profile = st.session_state["ydata_profile"]

            if SPP_OK:
                st.markdown("#### Embedded Report")
                st_profile_report(profile)
            else:
                st.markdown("""
                <div class="warn-box">
                ⚠️ <code>streamlit-pandas-profiling</code> is not installed — cannot embed report inline.<br>
                Run: <code>pip install streamlit-pandas-profiling</code><br>
                The report is still available for download below.
                </div>
                """, unsafe_allow_html=True)

                # Fallback: show key sections manually
                st.markdown("#### Quick Summary (fallback — install streamlit-pandas-profiling for full embed)")
                desc = profile.get_description()
                overview = desc.get("table", {})
                if overview:
                    ov_col1, ov_col2, ov_col3 = st.columns(3)
                    ov_col1.metric("Rows",          f"{overview.get('n', '?'):,}")
                    ov_col2.metric("Variables",     overview.get("n_var", "?"))
                    ov_col3.metric("Missing cells", f"{overview.get('n_cells_missing', 0):,}")
                    ov_col1.metric("Distinct rows", f"{overview.get('n_distinct', '?'):,}")
                    ov_col2.metric("Duplicates",    f"{overview.get('n_duplicates', 0):,}")
                    ov_col3.metric("Memory usage",  f"{overview.get('memory_size', 0)/1024:.1f} KB")

            if "ydata_html" in st.session_state:
                st.download_button(
                    "⬇ Download YData Profile HTML",
                    data=st.session_state["ydata_html"].encode("utf-8"),
                    file_name="ydata_profile_report.html",
                    mime="text/html",
                )

        # ── Side-by-side comparison
        st.markdown("---")
        st.markdown('<p class="section-title">Compare Two Datasets / Splits</p>', unsafe_allow_html=True)
        st.markdown("Generate separate profiles for two slices and download both.")

        cmp_col1, cmp_col2 = st.columns(2)
        with cmp_col1:
            cmp_col = st.selectbox("Split/filter column", ["None"] + cat_cols, key="cmp_col")
        with cmp_col2:
            if cmp_col != "None":
                unique_vals = df[cmp_col].dropna().unique().tolist()
                cmp_val1 = st.selectbox("Group A value", unique_vals, key="cmp_v1")
                cmp_val2 = st.selectbox("Group B value",
                                        [v for v in unique_vals if v != cmp_val1],
                                        key="cmp_v2")
            else:
                cmp_frac = st.slider("Train/test split fraction", 0.1, 0.5, 0.3, 0.05, key="cmp_frac")

        if cmp_col != "None" and st.button("Compare Groups", key="cmp_run"):
            df_a = df[df[cmp_col] == cmp_val1]
            df_b = df[df[cmp_col] == cmp_val2]
            with st.spinner("Profiling both groups…"):
                try:
                    pa = ProfileReport(df_a, title=f"Group: {cmp_val1}", minimal=True, progress_bar=False)
                    pb = ProfileReport(df_b, title=f"Group: {cmp_val2}", minimal=True, progress_bar=False)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.download_button(
                            f"⬇ Download profile — {cmp_val1}",
                            data=pa.to_html().encode("utf-8"),
                            file_name=f"profile_{cmp_val1}.html",
                            mime="text/html",
                        )
                    with c2:
                        st.download_button(
                            f"⬇ Download profile — {cmp_val2}",
                            data=pb.to_html().encode("utf-8"),
                            file_name=f"profile_{cmp_val2}.html",
                            mime="text/html",
                        )
                    st.success("Both profiles generated ✓  Download and open in browser to compare.")
                except Exception as e:
                    st.error(f"Comparison error: {e}")
        elif cmp_col == "None" and st.button("Compare Train/Test Split", key="cmp_split_run"):
            from sklearn.model_selection import train_test_split
            tr, te = train_test_split(df, test_size=cmp_frac, random_state=42)
            with st.spinner("Profiling train/test splits…"):
                try:
                    pa = ProfileReport(tr, title="Train set", minimal=True, progress_bar=False)
                    pb = ProfileReport(te, title="Test set",  minimal=True, progress_bar=False)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.download_button(
                            "⬇ Download Train profile",
                            data=pa.to_html().encode("utf-8"),
                            file_name="profile_train.html",
                            mime="text/html",
                        )
                    with c2:
                        st.download_button(
                            "⬇ Download Test profile",
                            data=pb.to_html().encode("utf-8"),
                            file_name="profile_test.html",
                            mime="text/html",
                        )
                    st.success("Train/Test profiles generated ✓")
                except Exception as e:
                    st.error(f"Split profile error: {e}")


# ══════════════════════════════════════════════════════════════
# TAB 9 ── EXPORT
# ══════════════════════════════════════════════════════════════
with tabs[8]:
    st.markdown('<p class="section-title">Export</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Original dataset**")
        st.download_button(
            "⬇ Download original CSV",
            data=df_to_csv(df),
            file_name="original_data.csv",
            mime="text/csv",
        )

    with c2:
        st.markdown("**Engineered dataset**")
        fe_df_export = st.session_state.get("fe_df", df)
        st.download_button(
            "⬇ Download engineered CSV",
            data=df_to_csv(fe_df_export),
            file_name="engineered_data.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.markdown("**Summary report (Markdown)**")
    report_lines = [
        "# Auto EDA Report\n",
        f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns\n",
        f"**Numeric columns ({len(num_cols)}):** {', '.join(num_cols)}\n",
        f"**Categorical columns ({len(cat_cols)}):** {', '.join(cat_cols)}\n",
        f"**Total missing values:** {df.isnull().sum().sum()} ({df.isnull().sum().sum()/df.size*100:.2f}%)\n",
        f"**Duplicate rows:** {df.duplicated().sum()}\n\n",
        "## Descriptive Statistics\n\n",
        df.describe(include="all").to_markdown(),
        "\n\n## Missing Value Summary\n\n",
        df.isnull().sum().reset_index().rename(columns={"index":"Column",0:"Missing"}).to_markdown(index=False),
    ]
    report_md = "\n".join(str(x) for x in report_lines)
    st.download_button(
        "⬇ Download Markdown report",
        data=report_md.encode(),
        file_name="eda_report.md",
        mime="text/markdown",
    )

    st.markdown("---")
    st.markdown("**Engineered dataframe info**")
    fe_df_final = st.session_state.get("fe_df", df)
    st.markdown(f"Shape: `{fe_df_final.shape[0]:,} rows × {fe_df_final.shape[1]} columns`  "
                f"|  New columns: `{fe_df_final.shape[1] - df.shape[1]}`")
    st.dataframe(fe_df_final.head(20), use_container_width=True)