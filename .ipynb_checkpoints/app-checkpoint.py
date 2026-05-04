import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import glob, os, warnings, joblib
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score, f1_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart API Protection · UNSW-NB15",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global style ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
code, pre, .mono { font-family: 'Space Mono', monospace; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0a0f1e;
    border-right: 1px solid #1e2d4a;
}
section[data-testid="stSidebar"] * { color: #c8d8f0 !important; }
section[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }

/* Main background */
.main { background: #06090f; color: #dce8ff; }

/* Cards */
.card {
    background: #0d1527;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-accent {
    border-left: 3px solid #3b82f6;
}

/* Metric boxes */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
.metric-box {
    background: #111d35;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    flex: 1; min-width: 130px;
    text-align: center;
}
.metric-box .val {
    font-family: 'Space Mono', monospace;
    font-size: 1.7rem; font-weight: 700;
    color: #60a5fa;
}
.metric-box .lbl {
    font-size: 0.75rem; color: #6b8ab0;
    text-transform: uppercase; letter-spacing: .08em;
    margin-top: .3rem;
}

/* Section headers */
.sec-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem; font-weight: 800;
    color: #93c5fd; margin: 1.5rem 0 .8rem;
    display: flex; align-items: center; gap: .5rem;
}

/* Status badge */
.badge {
    display: inline-block; border-radius: 6px;
    padding: .18rem .7rem; font-size: .75rem;
    font-family: 'Space Mono', monospace; font-weight: 700;
}
.badge-blue  { background: #1e3a5f; color: #60a5fa; }
.badge-green { background: #14312a; color: #34d399; }
.badge-red   { background: #3b1515; color: #f87171; }
.badge-amber { background: #33260a; color: #fbbf24; }

/* Divider */
hr.dim { border: none; border-top: 1px solid #1a2a42; margin: 1.5rem 0; }

/* Streamlit overrides */
.stButton > button {
    background: #1d4ed8; color: #fff;
    border: none; border-radius: 8px;
    font-family: 'Syne', sans-serif; font-weight: 600;
    padding: .55rem 1.4rem;
    transition: background .2s;
}
.stButton > button:hover { background: #2563eb; }
div[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    color: #60a5fa !important;
}
</style>
""", unsafe_allow_html=True)

# ── UNSW-NB15 column definitions ─────────────────────────────────────────────
UNSW_COLUMNS = [
    'srcip','sport','dstip','dsport','proto',
    'state','dur','sbytes','dbytes','sttl',
    'dttl','sloss','dloss','service','sload',
    'dload','spkts','dpkts','swin','dwin',
    'stcpb','dtcpb','smeansz','dmeansz','trans_depth',
    'res_bdy_len','sjit','djit','stime','ltime',
    'sintpkt','dintpkt','tcprtt','synack','ackdat',
    'is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd',
    'ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ltm','ct_src_dport_ltm',
    'ct_dst_sport_ltm','ct_dst_src_ltm','attack_cat','label'
]

NUMERIC_FEATURES = [
    'dur','sbytes','dbytes','sttl','dttl','sloss','dloss',
    'sload','dload','spkts','dpkts','swin','dwin',
    'smeansz','dmeansz','trans_depth','res_bdy_len',
    'sjit','djit','sintpkt','dintpkt','tcprtt','synack','ackdat',
    'is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd',
    'ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ltm',
    'ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm',
]

CAT_COLS = ['proto','state','service']

ATTACK_CATS = ['Normal','Fuzzers','Analysis','Backdoors','DoS',
               'Exploits','Generic','Reconnaissance','Shellcode','Worms']

PALETTE = {
    'bg':     '#06090f',
    'card':   '#0d1527',
    'border': '#1e2d4a',
    'blue':   '#3b82f6',
    'lblue':  '#60a5fa',
    'green':  '#34d399',
    'red':    '#f87171',
    'amber':  '#fbbf24',
    'text':   '#dce8ff',
    'muted':  '#6b8ab0',
}

# ── Session state keys ────────────────────────────────────────────────────────
for k in ['df','df_feat','X_train','X_test','y_train','y_test',
          'scaler','top_features','xgb','rf','iso',
          'y_pred_xgb','y_prob_xgb','y_pred_rf','y_prob_rf',
          'iso_labels','iso_acc','iso_f1']:
    if k not in st.session_state:
        st.session_state[k] = None

# ── Helpers ───────────────────────────────────────────────────────────────────
def dark_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(PALETTE['card'])
    ax.set_facecolor(PALETTE['card'])
    ax.tick_params(colors=PALETTE['muted'])
    ax.xaxis.label.set_color(PALETTE['muted'])
    ax.yaxis.label.set_color(PALETTE['muted'])
    ax.title.set_color(PALETTE['text'])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE['border'])
    return fig, ax

def dark_fig_multi(rows, cols, w=14, h=8):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h))
    fig.patch.set_facecolor(PALETTE['card'])
    for ax in np.array(axes).flatten():
        ax.set_facecolor(PALETTE['card'])
        ax.tick_params(colors=PALETTE['muted'], labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE['border'])
    return fig, axes

def badge(label, kind='blue'):
    return f'<span class="badge badge-{kind}">{label}</span>'

def section(icon, title):
    st.markdown(f'<div class="sec-header">{icon} {title}</div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Smart API Protection")
    st.markdown('<span style="color:#3b6ea8;font-size:.8rem;">UNSW-NB15 Dataset</span>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", [
        "📊 EDA",
        "🔧 Feature Engineering",
        "🤖 Model Training",
        "🔍 Predict",
    ])
    st.markdown("---")
    st.markdown('<span style="color:#3b6ea8;font-size:.75rem;">Pipeline Status</span>', unsafe_allow_html=True)
    st.markdown(f"Data loaded: {badge('Yes','green') if st.session_state.df is not None else badge('No','red')}", unsafe_allow_html=True)
    st.markdown(f"Features ready: {badge('Yes','green') if st.session_state.df_feat is not None else badge('No','red')}", unsafe_allow_html=True)
    st.markdown(f"Models trained: {badge('Yes','green') if st.session_state.xgb is not None else badge('No','red')}", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 EDA":
    st.markdown("# 📊 Exploratory Data Analysis")
    st.markdown('<hr class="dim">', unsafe_allow_html=True)

    # ── Load data ──
    section("📂", "Load UNSW-NB15 Dataset")

    col1, col2 = st.columns([2, 1])
    with col1:
        data_folder = st.text_input("Data folder path (containing UNSW-NB15_*.csv)", value="data/raw/")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        load_btn = st.button("⚡ Load Full Dataset")

    if load_btn:
        all_files = sorted(glob.glob(os.path.join(data_folder, "UNSW-NB15_*.csv")))
        if not all_files:
            st.error(f"No UNSW-NB15_*.csv files found in `{data_folder}`")
        else:
            dfs = []
            prog = st.progress(0, text="Loading files…")
            for i, f in enumerate(all_files):
                prog.progress((i) / len(all_files), text=f"Loading {os.path.basename(f)}…")
                try:
                    tmp = pd.read_csv(f, header=None, names=UNSW_COLUMNS,
                                      encoding='utf-8', low_memory=False)
                except UnicodeDecodeError:
                    tmp = pd.read_csv(f, header=None, names=UNSW_COLUMNS,
                                      encoding='latin-1', low_memory=False)
                tmp['attack_cat'] = tmp['attack_cat'].astype(str).str.strip()
                tmp['attack_cat'] = tmp['attack_cat'].replace({'nan': 'Normal', '': 'Normal'})
                tmp['label'] = pd.to_numeric(tmp['label'], errors='coerce').fillna(0).astype(int)
                dfs.append(tmp)
            prog.progress(1.0, text="Concatenating…")
            df = pd.concat(dfs, ignore_index=True)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            df['binary_label'] = df['label'].astype(int)
            st.session_state.df = df
            prog.empty()
            st.success(f"✅ Loaded {len(df):,} rows from {len(all_files)} files")

    # ── Display EDA if data loaded ──
    df = st.session_state.df
    if df is not None:
        st.markdown('<hr class="dim">', unsafe_allow_html=True)

        # KPI row
        attack_rate = df['binary_label'].mean()
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-box"><div class="val">{len(df):,}</div><div class="lbl">Total Records</div></div>
          <div class="metric-box"><div class="val">{df.shape[1]}</div><div class="lbl">Columns</div></div>
          <div class="metric-box"><div class="val">{df['binary_label'].sum():,}</div><div class="lbl">Attack Records</div></div>
          <div class="metric-box"><div class="val">{attack_rate:.1%}</div><div class="lbl">Attack Rate</div></div>
          <div class="metric-box"><div class="val">{df['attack_cat'].nunique()}</div><div class="lbl">Attack Categories</div></div>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "🎯 Label Distribution", "📈 Feature Correlations", "📊 Distributions"])

        with tab1:
            section("🔍", "Data Preview")
            st.dataframe(df.head(10).style.set_properties(**{'background-color': '#0d1527', 'color': '#dce8ff'}), use_container_width=True)
            st.markdown('<hr class="dim">', unsafe_allow_html=True)
            section("🔍", "Data Types & Missing Values")
            info_df = pd.DataFrame({
                'dtype': df.dtypes,
                'null_count': df.isnull().sum(),
                'null_%': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(info_df.style.set_properties(**{'background-color': '#0d1527', 'color': '#dce8ff'}), use_container_width=True)

        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                section("🎯", "Attack Category Counts")
                cat_counts = df['attack_cat'].value_counts()
                fig, ax = dark_fig(7, 4)
                colors = [PALETTE['blue'] if x != 'Normal' else PALETTE['green'] for x in cat_counts.index]
                cat_counts.plot(kind='barh', ax=ax, color=colors[::-1], edgecolor='none')
                ax.set_xlabel("Count", color=PALETTE['muted'])
                ax.set_title("Attack Categories — UNSW-NB15", color=PALETTE['text'])
                plt.tight_layout()
                st.pyplot(fig)

            with c2:
                section("🎯", "Binary Label Split")
                fig, ax = dark_fig(5, 4)
                vals = df['binary_label'].value_counts()
                ax.pie(vals, labels=['Normal', 'Attack'],
                       colors=[PALETTE['green'], PALETTE['red']],
                       autopct='%1.1f%%', startangle=140,
                       textprops={'color': PALETTE['text']},
                       wedgeprops={'edgecolor': PALETTE['card'], 'linewidth': 2})
                ax.set_title("Normal vs Attack", color=PALETTE['text'])
                plt.tight_layout()
                st.pyplot(fig)

        with tab3:
            section("📊", "Top 15 Features Correlated with Attack Label")
            drop_cols = ['srcip','dstip','sport','dsport','proto','state',
                         'service','attack_cat','label','stime','ltime','stcpb','dtcpb']
            num_cols = df.drop(columns=[c for c in drop_cols if c in df.columns]) \
                         .select_dtypes(include=[np.number]).columns
            corr = df[num_cols].corrwith(df['binary_label']).abs().sort_values(ascending=False).head(15)

            fig, ax = dark_fig(10, 5)
            bars = ax.barh(corr.index[::-1], corr.values[::-1],
                           color=PALETTE['blue'], edgecolor='none')
            # gradient color
            for i, bar in enumerate(bars):
                bar.set_alpha(0.5 + 0.5 * (i / len(bars)))
            ax.set_xlabel("Absolute Correlation", color=PALETTE['muted'])
            ax.set_title("Feature Correlation with Attack Label", color=PALETTE['text'])
            plt.tight_layout()
            st.pyplot(fig)

            st.dataframe(
                corr.reset_index().rename(columns={'index':'Feature', 0:'Correlation'})
                    .style.background_gradient(cmap='Blues'),
                use_container_width=True
            )

        with tab4:
            section("📈", "Feature Distributions — Normal vs Attack")
            drop_cols2 = ['srcip','dstip','sport','dsport','proto','state',
                          'service','attack_cat','label','stime','ltime','stcpb','dtcpb','binary_label']
            num_cols2 = df.drop(columns=[c for c in drop_cols2 if c in df.columns]) \
                          .select_dtypes(include=[np.number]).columns.tolist()
            feat_sel = st.selectbox("Select feature to plot", num_cols2[:20])
            if feat_sel:
                fig, ax = dark_fig(10, 4)
                normal = df[df['binary_label'] == 0][feat_sel]
                attack = df[df['binary_label'] == 1][feat_sel]
                ax.hist(normal.clip(upper=normal.quantile(0.99)), bins=50,
                        alpha=0.6, label='Normal', color=PALETTE['green'])
                ax.hist(attack.clip(upper=attack.quantile(0.99)), bins=50,
                        alpha=0.6, label='Attack', color=PALETTE['red'])
                ax.set_title(f"{feat_sel} — Normal vs Attack", color=PALETTE['text'])
                ax.legend(labelcolor=PALETTE['text'], facecolor=PALETTE['card'])
                plt.tight_layout()
                st.pyplot(fig)

        st.markdown('<hr class="dim">', unsafe_allow_html=True)
        section("💾", "Save Cleaned Data")
        save_path = st.text_input("Save path", value="data/processed/cleaned_data.csv")
        if st.button("💾 Save CSV"):
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            df.to_csv(save_path, index=False)
            st.success(f"✅ Saved {len(df):,} rows → `{save_path}`")

    else:
        st.info("👆 Enter your data folder path and click **Load Full Dataset** to begin.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔧 Feature Engineering":
    st.markdown("# 🔧 Feature Engineering")
    st.markdown('<hr class="dim">', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Please load the dataset on the **📊 EDA** page first.")
        st.stop()

    df = st.session_state.df.copy()

    # ── Step 1: Encode categoricals ──
    section("🔤", "Step 1 — Encode Categorical Features")
    st.markdown("`proto`, `state`, `service` are label-encoded.")

    if st.button("▶ Run Encoding"):
        le = LabelEncoder()
        encoded = []
        for col in CAT_COLS:
            if col in df.columns:
                df[col + '_enc'] = le.fit_transform(df[col].astype(str).str.strip().str.lower())
                encoded.append(col)
        st.success(f"✅ Encoded: {encoded}")
        st.session_state._enc_df = df

    if hasattr(st.session_state, '_enc_df'):
        df = st.session_state._enc_df

    # ── Step 2: Engineer features ──
    st.markdown('<hr class="dim">', unsafe_allow_html=True)
    section("🔧", "Step 2 — Engineer New Features")

    engineered = {
        'pkt_ratio':      'spkts / (dpkts + 1)',
        'byte_ratio':     'sbytes / (dbytes + 1)',
        'high_src_load':  'sload > 90th percentile → binary',
        'fast_synack':    'synack < 5th percentile → binary',
        'short_flow':     'dur < 10th percentile → binary',
        'total_jit':      'sjit + djit',
        'win_diff':       '|swin - dwin|',
    }
    st.table(pd.DataFrame(list(engineered.items()), columns=['Feature', 'Formula']))

    if st.button("▶ Engineer Features"):
        if 'spkts' in df.columns and 'dpkts' in df.columns:
            df['pkt_ratio'] = df['spkts'] / (df['dpkts'] + 1)
        if 'sbytes' in df.columns and 'dbytes' in df.columns:
            df['byte_ratio'] = df['sbytes'] / (df['dbytes'] + 1)
        if 'sload' in df.columns:
            df['high_src_load'] = (df['sload'] > df['sload'].quantile(0.90)).astype(int)
        if 'synack' in df.columns:
            df['fast_synack'] = (df['synack'] < df['synack'].quantile(0.05)).astype(int)
        if 'dur' in df.columns:
            df['short_flow'] = (df['dur'] < df['dur'].quantile(0.10)).astype(int)
        if 'sjit' in df.columns and 'djit' in df.columns:
            df['total_jit'] = df['sjit'] + df['djit']
        if 'swin' in df.columns and 'dwin' in df.columns:
            df['win_diff'] = (df['swin'] - df['dwin']).abs()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.median(numeric_only=True), inplace=True)
        st.session_state._eng_df = df
        st.success(f"✅ Feature engineering done. Shape: {df.shape}")

    if hasattr(st.session_state, '_eng_df'):
        df = st.session_state._eng_df

    # ── Step 3: SelectKBest ──
    st.markdown('<hr class="dim">', unsafe_allow_html=True)
    section("📊", "Step 3 — Feature Selection (SelectKBest · F-Score)")

    top_n = st.slider("Number of top features to keep", 10, 40, 20)

    if st.button("▶ Run Feature Selection"):
        drop = ['srcip','dstip','sport','dsport','proto','state','service',
                'attack_cat','label','stime','ltime','stcpb','dtcpb','binary_label']
        X_all = df.drop(columns=[c for c in drop if c in df.columns]).select_dtypes(include=[np.number])
        y_all = df['binary_label']
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X_all, y_all)
        feat_scores = pd.DataFrame({'feature': X_all.columns, 'score': selector.scores_}) \
                        .sort_values('score', ascending=False)
        top_features = feat_scores.head(top_n)['feature'].tolist()

        fig, ax = dark_fig(10, 5)
        top20 = feat_scores.head(20)
        ax.barh(top20['feature'][::-1], top20['score'][::-1], color=PALETTE['blue'], edgecolor='none')
        ax.set_xlabel("F-Score", color=PALETTE['muted'])
        ax.set_title(f"Top 20 Features by F-Score", color=PALETTE['text'])
        plt.tight_layout()
        st.pyplot(fig)

        st.session_state._top_features = top_features
        st.session_state._feat_scores  = feat_scores
        st.success(f"✅ Top {top_n} features selected: {top_features}")

    if hasattr(st.session_state, '_top_features'):
        top_features = st.session_state._top_features

        # ── Step 4: Scale + Split ──
        st.markdown('<hr class="dim">', unsafe_allow_html=True)
        section("⚖️", "Step 4 — Scale & Split")

        test_size = st.slider("Test set size", 0.1, 0.4, 0.2, step=0.05)

        if st.button("▶ Scale & Split"):
            X = df[top_features]
            y = df['binary_label']
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=top_features)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42, stratify=y)

            st.session_state.df_feat     = df
            st.session_state.top_features = top_features
            st.session_state.scaler       = scaler
            st.session_state.X_train      = X_train
            st.session_state.X_test       = X_test
            st.session_state.y_train      = y_train
            st.session_state.y_test       = y_test

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("X_train rows", f"{X_train.shape[0]:,}")
            c2.metric("X_test rows",  f"{X_test.shape[0]:,}")
            c3.metric("Train attack%", f"{y_train.mean():.1%}")
            c4.metric("Test attack%",  f"{y_test.mean():.1%}")
            st.success("✅ Ready for model training!")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Training":
    st.markdown("# 🤖 Model Training")
    st.markdown('<hr class="dim">', unsafe_allow_html=True)

    if st.session_state.X_train is None:
        st.warning("⚠️ Please complete **Feature Engineering** first.")
        st.stop()

    X_train = st.session_state.X_train
    X_test  = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test  = st.session_state.y_test

    # ── SMOTE ──
    section("⚖️", "Class Balancing — SMOTE")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="card card-accent">
        <b>Before SMOTE</b><br>
        Normal : {(y_train==0).sum():,}<br>
        Attack : {(y_train==1).sum():,}
        </div>
        """, unsafe_allow_html=True)

    use_smote = st.checkbox("Apply SMOTE to training set", value=True)

    if st.button("▶ Apply SMOTE & Prepare"):
        if use_smote:
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_train)
        else:
            X_res, y_res = X_train, y_train
        st.session_state._X_res = X_res
        st.session_state._y_res = y_res
        with col2:
            st.markdown(f"""
            <div class="card card-accent">
            <b>After SMOTE</b><br>
            Normal : {(y_res==0).sum():,}<br>
            Attack : {(y_res==1).sum():,}
            </div>
            """, unsafe_allow_html=True)
        st.success("✅ Training data ready!")

    st.markdown('<hr class="dim">', unsafe_allow_html=True)

    # ── XGBoost params ──
    section("🤖", "XGBoost Hyperparameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        n_est  = st.number_input("n_estimators", 50, 500, 200, step=50)
        max_d  = st.number_input("max_depth", 2, 12, 6)
    with c2:
        lr     = st.number_input("learning_rate", 0.01, 0.5, 0.1, step=0.01, format="%.2f")
        subsam = st.slider("subsample", 0.5, 1.0, 0.8, step=0.05)
    with c3:
        col_bt = st.slider("colsample_bytree", 0.5, 1.0, 0.8, step=0.05)

    if st.button("🚀 Train All Models"):
        if not hasattr(st.session_state, '_X_res') or st.session_state._X_res is None:
            st.error("Run SMOTE first!")
            st.stop()

        X_res = st.session_state._X_res
        y_res = st.session_state._y_res

        with st.spinner("Training XGBoost…"):
            xgb = XGBClassifier(
                n_estimators=int(n_est), max_depth=int(max_d),
                learning_rate=lr, subsample=subsam,
                colsample_bytree=col_bt,
                use_label_encoder=False, eval_metric='logloss',
                random_state=42, n_jobs=-1)
            xgb.fit(X_res, y_res)

        with st.spinner("Training Random Forest…"):
            rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                        random_state=42, n_jobs=-1)
            rf.fit(X_res, y_res)

        with st.spinner("Training Isolation Forest…"):
            iso = IsolationForest(n_estimators=100, contamination=0.05,
                                  random_state=42, n_jobs=-1)
            iso.fit(X_train[y_train == 0])

        # Evaluate
        y_pred_xgb  = xgb.predict(X_test)
        y_prob_xgb  = xgb.predict_proba(X_test)[:, 1]
        y_pred_rf   = rf.predict(X_test)
        y_prob_rf   = rf.predict_proba(X_test)[:, 1]
        iso_preds   = iso.predict(X_test)
        iso_labels  = (iso_preds == -1).astype(int)

        st.session_state.xgb        = xgb
        st.session_state.rf         = rf
        st.session_state.iso        = iso
        st.session_state.y_pred_xgb = y_pred_xgb
        st.session_state.y_prob_xgb = y_prob_xgb
        st.session_state.y_pred_rf  = y_pred_rf
        st.session_state.y_prob_rf  = y_prob_rf
        st.session_state.iso_labels = iso_labels
        st.session_state.iso_acc    = accuracy_score(y_test, iso_labels)
        st.session_state.iso_f1     = f1_score(y_test, iso_labels)
        st.success("✅ All models trained!")

    # ── Results ──
    if st.session_state.xgb is not None:
        y_pred_xgb = st.session_state.y_pred_xgb
        y_prob_xgb = st.session_state.y_prob_xgb
        y_pred_rf  = st.session_state.y_pred_rf
        y_prob_rf  = st.session_state.y_prob_rf
        iso_labels = st.session_state.iso_labels

        acc_x = accuracy_score(y_test, y_pred_xgb)
        f1_x  = f1_score(y_test, y_pred_xgb)
        auc_x = roc_auc_score(y_test, y_prob_xgb)
        acc_r = accuracy_score(y_test, y_pred_rf)
        f1_r  = f1_score(y_test, y_pred_rf)
        auc_r = roc_auc_score(y_test, y_prob_rf)

        st.markdown('<hr class="dim">', unsafe_allow_html=True)
        section("📊", "Model Comparison")

        comp = pd.DataFrame({
            'Model':    ['XGBoost', 'Random Forest', 'Isolation Forest'],
            'Accuracy': [f"{acc_x:.4f}", f"{acc_r:.4f}", f"{st.session_state.iso_acc:.4f}"],
            'F1 Score': [f"{f1_x:.4f}",  f"{f1_r:.4f}",  f"{st.session_state.iso_f1:.4f}"],
            'AUC-ROC':  [f"{auc_x:.4f}", f"{auc_r:.4f}", "N/A (unsupervised)"],
        })
        st.dataframe(comp.set_index('Model'), use_container_width=True)

        st.markdown('<hr class="dim">', unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["Confusion Matrix", "ROC Curve", "Feature Importance", "Classification Report"])

        with tab1:
            c1, c2 = st.columns(2)
            for col, (name, preds) in zip([c1,c2],[('XGBoost',y_pred_xgb),('Random Forest',y_pred_rf)]):
                with col:
                    cm = confusion_matrix(y_test, preds)
                    fig, ax = dark_fig(5, 4)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                xticklabels=['Normal','Attack'],
                                yticklabels=['Normal','Attack'],
                                linewidths=1, linecolor=PALETTE['card'])
                    ax.set_title(f"Confusion Matrix — {name}", color=PALETTE['text'])
                    ax.set_ylabel("Actual", color=PALETTE['muted'])
                    ax.set_xlabel("Predicted", color=PALETTE['muted'])
                    plt.tight_layout()
                    st.pyplot(fig)

        with tab2:
            fig, ax = dark_fig(9, 5)
            for name, prob, color in [('XGBoost', y_prob_xgb, PALETTE['blue']),
                                       ('Random Forest', y_prob_rf, PALETTE['green'])]:
                fpr, tpr, _ = roc_curve(y_test, prob)
                auc = roc_auc_score(y_test, prob)
                ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={auc:.4f})")
            ax.plot([0,1],[0,1],'--', color=PALETTE['muted'], lw=1)
            ax.set_xlabel("False Positive Rate", color=PALETTE['muted'])
            ax.set_ylabel("True Positive Rate", color=PALETTE['muted'])
            ax.set_title("ROC Curves — UNSW-NB15", color=PALETTE['text'])
            ax.legend(facecolor=PALETTE['card'], labelcolor=PALETTE['text'])
            plt.tight_layout()
            st.pyplot(fig)

        with tab3:
            xgb_model = st.session_state.xgb
            feat_imp = pd.Series(xgb_model.feature_importances_,
                                 index=X_train.columns).sort_values(ascending=False).head(15)
            fig, ax = dark_fig(10, 5)
            feat_imp[::-1].plot(kind='barh', ax=ax, color=PALETTE['blue'], edgecolor='none')
            ax.set_title("Top 15 Feature Importances — XGBoost", color=PALETTE['text'])
            ax.set_xlabel("Importance Score", color=PALETTE['muted'])
            plt.tight_layout()
            st.pyplot(fig)

        with tab4:
            report = classification_report(y_test, y_pred_xgb,
                                           target_names=['Normal','Attack'],
                                           output_dict=True)
            st.dataframe(pd.DataFrame(report).T.style.format("{:.3f}"), use_container_width=True)

        st.markdown('<hr class="dim">', unsafe_allow_html=True)
        section("💾", "Save Models")
        save_dir = st.text_input("Models directory", value="models/")
        if st.button("💾 Save All Models"):
            os.makedirs(save_dir, exist_ok=True)
            joblib.dump(st.session_state.xgb,    os.path.join(save_dir, 'classifier.pkl'))
            joblib.dump(st.session_state.rf,     os.path.join(save_dir, 'random_forest.pkl'))
            joblib.dump(st.session_state.iso,    os.path.join(save_dir, 'anomaly_detector.pkl'))
            joblib.dump(st.session_state.scaler, os.path.join(save_dir, 'scaler.pkl'))
            joblib.dump(st.session_state.top_features, os.path.join(save_dir, 'feature_names.pkl'))
            st.success(f"✅ Saved classifier, random_forest, anomaly_detector, scaler, feature_names → `{save_dir}`")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Predict":
    st.markdown("# 🔍 Live Threat Prediction")
    st.markdown('<hr class="dim">', unsafe_allow_html=True)

    section("📂", "Load Models")
    col1, col2 = st.columns(2)
    with col1:
        model_dir = st.text_input("Models directory", value="models/")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📂 Load Saved Models"):
            try:
                st.session_state.xgb          = joblib.load(os.path.join(model_dir, 'classifier.pkl'))
                st.session_state.iso          = joblib.load(os.path.join(model_dir, 'anomaly_detector.pkl'))
                st.session_state.scaler       = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
                st.session_state.top_features = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
                st.success("✅ Models loaded from disk!")
            except Exception as e:
                st.error(f"Failed: {e}")

    model_ready = (st.session_state.xgb is not None and
                   st.session_state.scaler is not None and
                   st.session_state.top_features is not None)

    if not model_ready:
        st.info("Train models in **🤖 Model Training** or load saved models above.")
        st.stop()

    st.markdown('<hr class="dim">', unsafe_allow_html=True)
    section("🧪", "Test on Held-Out Data")

    if st.session_state.X_test is not None:
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        n = len(X_test)
        idx = st.slider("Sample index", 0, n - 1, 0)

        sample = X_test.iloc[[idx]]
        actual = y_test.iloc[idx]

        threat_score = st.session_state.xgb.predict_proba(sample)[0][1]
        anomaly_flag = st.session_state.iso.predict(sample)[0] == -1

        if threat_score > 0.85 or anomaly_flag:
            decision = "BLOCK"; badge_kind = "red"
        elif threat_score > 0.5:
            decision = "FLAG"; badge_kind = "amber"
        else:
            decision = "ALLOW"; badge_kind = "green"

        st.markdown(f"""
        <div class="card card-accent">
          <div style="display:flex;gap:2rem;align-items:center;flex-wrap:wrap;">
            <div>
              <div class="lbl">Actual Label</div>
              <div style="font-size:1.3rem;font-weight:700;color:{'#f87171' if actual==1 else '#34d399'}">
                {'⚔️ Attack' if actual==1 else '✅ Normal'}
              </div>
            </div>
            <div>
              <div class="lbl">Threat Score</div>
              <div style="font-family:'Space Mono';font-size:1.6rem;color:#60a5fa">{threat_score:.4f}</div>
            </div>
            <div>
              <div class="lbl">Anomaly Flag</div>
              <div style="font-size:1.3rem">{'🚨 Yes' if anomaly_flag else '✅ No'}</div>
            </div>
            <div>
              <div class="lbl">Decision</div>
              <span class="badge badge-{badge_kind}" style="font-size:1.1rem;padding:.4rem 1rem">{decision}</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Threat gauge
        fig, ax = plt.subplots(figsize=(8, 1.2))
        fig.patch.set_facecolor(PALETTE['card'])
        ax.set_facecolor(PALETTE['card'])
        ax.barh(0, 1, color=PALETTE['border'], height=0.4)
        bar_color = PALETTE['red'] if threat_score > 0.85 else (PALETTE['amber'] if threat_score > 0.5 else PALETTE['green'])
        ax.barh(0, threat_score, color=bar_color, height=0.4)
        ax.axvline(0.5,  color=PALETTE['amber'], lw=1.5, linestyle='--')
        ax.axvline(0.85, color=PALETTE['red'],   lw=1.5, linestyle='--')
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Threat Score", color=PALETTE['muted'])
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.tick_params(colors=PALETTE['muted'])
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown('<hr class="dim">', unsafe_allow_html=True)
    section("✏️", "Manual Input Prediction")

    top_features = st.session_state.top_features
    scaler       = st.session_state.scaler

    with st.expander("Enter feature values manually"):
        input_vals = {}
        cols = st.columns(4)
        for i, feat in enumerate(top_features):
            with cols[i % 4]:
                input_vals[feat] = st.number_input(feat, value=0.0, key=f"inp_{feat}", format="%.4f")

        if st.button("🔍 Predict"):
            raw = pd.DataFrame([input_vals])
            scaled = pd.DataFrame(scaler.transform(raw), columns=top_features)
            threat_score = st.session_state.xgb.predict_proba(scaled)[0][1]
            anomaly_flag = st.session_state.iso.predict(scaled)[0] == -1

            if threat_score > 0.85 or anomaly_flag:
                decision = "BLOCK"; badge_kind = "red"
            elif threat_score > 0.5:
                decision = "FLAG";  badge_kind = "amber"
            else:
                decision = "ALLOW"; badge_kind = "green"

            st.markdown(f"""
            <div class="card">
              Threat Score: <span style="font-family:Space Mono;color:#60a5fa">{threat_score:.4f}</span> &nbsp;|&nbsp;
              Anomaly: {'🚨 Yes' if anomaly_flag else '✅ No'} &nbsp;|&nbsp;
              Decision: <span class="badge badge-{badge_kind}">{decision}</span>
            </div>
            """, unsafe_allow_html=True)
