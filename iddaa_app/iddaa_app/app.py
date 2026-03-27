import re
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# SAYFA AYARLARI
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="İddaa Tahmin Modeli",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { color: #e2e8f0; }
    .metric-card {
        background: #1a1d26;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 32px; font-weight: 700; margin: 4px 0; }
    .metric-note  { font-size: 12px; color: #94a3b8; }
    .green { color: #22c55e; }
    .amber { color: #f59e0b; }
    .blue  { color: #3b82f6; }
    .red   { color: #ef4444; }
    div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/football2.png", width=40)
    st.title("İddaa Tahmin")
    st.caption("Haftalık dosya yükle → Tahmin al")

    st.divider()
    uploaded = st.file_uploader(
        "📂 Veri Dosyası (.txt)",
        type=["txt"],
        help="iddaa_100hafta_merged türünde txt dosyası"
    )

    st.divider()
    st.subheader("⚙️ Model Ayarları")

    selected_models = st.multiselect(
        "Modeller",
        ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "Gradient Boosting"],
        default=["Random Forest", "XGBoost", "LightGBM"],
    )

    cv_folds = st.slider("Cross-Validation Katlama", 3, 10, 5)

    confidence_threshold = st.slider(
        "Güven Eşiği (%)",
        min_value=50, max_value=75, value=60,
        help="Bu eşiğin üstündeki tahminler 'güçlü' sayılır"
    )

    st.divider()
    st.subheader("🔍 Filtreler")
    filter_league = st.text_input("Lig Filtresi", placeholder="ör: TÜR S")
    filter_pred   = st.selectbox("Tahmin Filtresi", ["Tümü", "1", "X", "2"])
    filter_ou     = st.selectbox("A/Ü 2.5", ["Tümü", "Üstü", "Altı"])

    st.divider()
    st.caption("Her hafta yeni dosya yükleyebilirsiniz.\nModel sıfırdan eğitilir.")


# ─────────────────────────────────────────────
# PARSE FONKSİYONU
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def parse_iddaa(content: bytes) -> pd.DataFrame:
    lines = content.decode("utf-8", errors="ignore").splitlines()

    def safe_float(lst, idx):
        try:
            v = lst[idx].strip()
            return float(v) if v and v not in ("-", "") else np.nan
        except:
            return np.nan

    def extract_detail(s):
        out = {}
        defs = [
            ("iy",   r"İlk Yarı Sonucu \d+ 1([\d.]+) 0([\d.]+) 2([\d.]+)"),
            ("sy",   r"İkinci Yarı Sonucu \d+ 1([\d.]+) 0([\d.]+) 2([\d.]+)"),
            ("ou15", r"1\.5 Altı/Üstü \d+ Alt([\d.]+) Üst([\d.]+)"),
            ("ou35", r"3\.5 Altı/Üstü \d+ Alt([\d.]+) Üst([\d.]+)"),
            ("tc",   r"Tek / Çift \d+ Tek([\d.]+) Çift([\d.]+)"),
            ("tg",   r"Toplam Gol \d+ 0-1 ([\d.]+) 2-3 ([\d.]+)"),
            ("dc",   r"Çifte Şans \d+ 1/X([\d.]+) 1/2([\d.]+) 0/2([\d.]+)"),
        ]
        for key, pat in defs:
            m = re.search(pat, s)
            if not m:
                continue
            g = [float(x) for x in m.groups()]
            if key == "iy":   out.update(iy1=g[0], iy0=g[1], iy2=g[2])
            elif key == "sy": out.update(sy1=g[0], sy0=g[1], sy2=g[2])
            elif key == "ou15": out.update(ou15u=g[0], ou15o=g[1])
            elif key == "ou35": out.update(ou35u=g[0], ou35o=g[1])
            elif key == "tc": out.update(odd_o=g[0], even_o=g[1])
            elif key == "tg": out.update(tg01=g[0], tg23=g[1])
            elif key == "dc": out.update(dc_1x=g[0], dc_12=g[1], dc_x2=g[2])
        return out

    records = []
    for i, line in enumerate(lines):
        parts = line.strip().split("\t")
        if len(parts) < 10:
            continue
        if not re.match(r"^\d{2}:\d{2}$", parts[1].strip()):
            continue

        score_raw = parts[6].strip() if len(parts) > 6 else ""
        ht_raw    = parts[8].strip() if len(parts) > 8 else ""
        played    = bool(re.match(r"^\d+-\d+$", score_raw))

        detail_str = ""
        if i + 1 < len(lines):
            dp = lines[i + 1].strip().split("\t")
            if len(dp) > 1:
                detail_str = dp[1]

        rec = {
            "league":   parts[3].strip() if len(parts) > 3 else "",
            "home":     parts[5].strip() if len(parts) > 5 else "",
            "away":     parts[7].strip() if len(parts) > 7 else "",
            "score":    score_raw if played else None,
            "ht_score": ht_raw    if played else None,
            "played":   played,
            "o1":  safe_float(parts, 9),
            "ox":  safe_float(parts, 10),
            "o2":  safe_float(parts, 11),
            "ou25u": safe_float(parts, 16),
            "ou25o": safe_float(parts, 17),
        }
        rec.update(extract_detail(detail_str))
        records.append(rec)

    df = pd.DataFrame(records)
   if not records:
        st.error("❌ Dosyadan hiç maç okunamadı. Dosya formatını kontrol edin.")
        st.stop()

    df = pd.DataFrame(records)

    # Kolon yoksa oluştur
    for col in ["o1", "ox", "o2"]:
        if col not in df.columns:
            df[col] = np.nan

    df = df[df["o1"].notna() & df["ox"].notna() & df["o2"].notna()].reset_index(drop=True)

    if len(df) == 0:
        st.error("❌ Geçerli oran verisi bulunamadı. Dosya formatı farklı olabilir.")
        st.stop()
    return df


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Normalize 1X2
    d["r1"] = 1 / d["o1"];  d["rx"] = 1 / d["ox"];  d["r2"] = 1 / d["o2"]
    d["margin"] = d["r1"] + d["rx"] + d["r2"]
    d["p1"] = d["r1"] / d["margin"]
    d["px"] = d["rx"] / d["margin"]
    d["p2"] = d["r2"] / d["margin"]

    d["fav_prob"]    = d[["p1","px","p2"]].max(axis=1)
    d["second_prob"] = d[["p1","px","p2"]].apply(lambda r: sorted(r)[-2], axis=1)
    d["fav_gap"]     = d["fav_prob"] - d["second_prob"]
    d["home_away_ratio"] = d["p1"] / (d["p2"] + 1e-9)
    d["draw_pull"]   = d["px"] / (d["p1"] + d["p2"] + 1e-9)
    d["entropy"]     = -(d["p1"]*np.log(d["p1"]+1e-9) +
                         d["px"]*np.log(d["px"]+1e-9) +
                         d["p2"]*np.log(d["p2"]+1e-9))
    d["log_o1"] = np.log(d["o1"])
    d["log_ox"] = np.log(d["ox"])
    d["log_o2"] = np.log(d["o2"])
    d["o1_o2_spread"] = d["o1"] - d["o2"]
    d["min_odds"]     = d[["o1","o2"]].min(axis=1)

    # OU 2.5
    m = d["ou25u"].notna() & d["ou25o"].notna()
    d.loc[m,"ou25_tot"]    = 1/d.loc[m,"ou25u"] + 1/d.loc[m,"ou25o"]
    d.loc[m,"p_over25"]    = (1/d.loc[m,"ou25o"]) / d.loc[m,"ou25_tot"]
    d.loc[m,"p_under25"]   = (1/d.loc[m,"ou25u"]) / d.loc[m,"ou25_tot"]
    d.loc[m,"ou25_margin"] = d.loc[m,"ou25_tot"] - 1

    # OU 1.5 / 3.5
    for uc, oc in [("ou15u","ou15o"),("ou35u","ou35o")]:
        if uc in d.columns and oc in d.columns:
            mk = d[uc].notna() & d[oc].notna()
            name = "p_over" + uc[2:4]
            d.loc[mk, name] = (1/d.loc[mk,oc]) / (1/d.loc[mk,uc] + 1/d.loc[mk,oc])

    # IY
    m = d["iy1"].notna() & d["iy0"].notna() & d["iy2"].notna()
    d.loc[m,"iy_tot"] = 1/d.loc[m,"iy1"] + 1/d.loc[m,"iy0"] + 1/d.loc[m,"iy2"]
    d.loc[m,"iy_p1"]  = (1/d.loc[m,"iy1"]) / d.loc[m,"iy_tot"]
    d.loc[m,"iy_px"]  = (1/d.loc[m,"iy0"]) / d.loc[m,"iy_tot"]
    d.loc[m,"iy_p2"]  = (1/d.loc[m,"iy2"]) / d.loc[m,"iy_tot"]
    d["iy_1_delta"] = d.get("iy_p1", pd.Series(np.nan, index=d.index)) - d["p1"]
    d["iy_2_delta"] = d.get("iy_p2", pd.Series(np.nan, index=d.index)) - d["p2"]
    d["iy_x_delta"] = d.get("iy_px", pd.Series(np.nan, index=d.index)) - d["px"]

    # 2Y
    m = d["sy1"].notna() & d["sy0"].notna() & d["sy2"].notna()
    d.loc[m,"sy_tot"] = 1/d.loc[m,"sy1"] + 1/d.loc[m,"sy0"] + 1/d.loc[m,"sy2"]
    d.loc[m,"sy_p1"]  = (1/d.loc[m,"sy1"]) / d.loc[m,"sy_tot"]
    d.loc[m,"sy_p2"]  = (1/d.loc[m,"sy2"]) / d.loc[m,"sy_tot"]

    # DC
    m = d["dc_1x"].notna() & d["dc_x2"].notna() & d["dc_12"].notna()
    d.loc[m,"dc_p1"] = 1/d.loc[m,"dc_1x"] - 1/d.loc[m,"dc_x2"] + 1/d.loc[m,"dc_12"]
    d.loc[m,"dc_p2"] = 1/d.loc[m,"dc_x2"] - 1/d.loc[m,"dc_1x"] + 1/d.loc[m,"dc_12"]

    # Tek/Çift
    m = d["odd_o"].notna() & d["even_o"].notna()
    d.loc[m,"p_odd"] = (1/d.loc[m,"odd_o"]) / (1/d.loc[m,"odd_o"] + 1/d.loc[m,"even_o"])

    # Toplam gol
    m = d["tg01"].notna() & d["tg23"].notna()
    d.loc[m,"tg_tot"]  = 1/d.loc[m,"tg01"] + 1/d.loc[m,"tg23"]
    d.loc[m,"p_low_g"] = (1/d.loc[m,"tg01"]) / d.loc[m,"tg_tot"]

    # Hedef
    def result(score):
        try:
            h, a = map(int, score.split("-"))
            return "1" if h > a else ("X" if h == a else "2")
        except:
            return None

    d["result"] = d["score"].apply(result)

    def total_goals(score):
        try:
            h, a = map(int, score.split("-"))
            return h + a
        except:
            return np.nan

    d["total_goals"] = d["score"].apply(total_goals)
    d["over25"] = (d["total_goals"] > 2.5).astype(float)

    return d


# ─────────────────────────────────────────────
# MODEL EĞİTİM
# ─────────────────────────────────────────────
FEATURES = [
    "p1","px","p2",
    "fav_prob","second_prob","fav_gap",
    "home_away_ratio","draw_pull","entropy",
    "log_o1","log_ox","log_o2","o1_o2_spread","min_odds","margin",
    "p_over25","p_under25","ou25_margin",
    "p_over15","p_over35",
    "iy_p1","iy_px","iy_p2","iy_1_delta","iy_2_delta","iy_x_delta",
    "sy_p1","sy_p2",
    "dc_p1","dc_p2",
    "p_odd","p_low_g",
]

def get_model(name):
    if name == "Logistic Regression":
        return Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42))])
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)
    if name == "XGBoost":
        return xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0)
    if name == "LightGBM":
        return lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)


def train_models(X, y_enc, le, selected, cv_folds, progress_bar):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}
    oof_preds = {}
    total = len(selected)

    for idx, name in enumerate(selected):
        progress_bar.progress((idx) / total, text=f"Eğitiliyor: {name}...")
        model = get_model(name)
        oof_prob = np.zeros((len(X), 3))
        acc_list, ll_list = [], []

        for tr_idx, val_idx in skf.split(X, y_enc):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y_enc[tr_idx], y_enc[val_idx]
            model.fit(X_tr, y_tr)
            pred  = model.predict(X_val)
            prob  = model.predict_proba(X_val)
            acc_list.append(accuracy_score(y_val, pred))
            ll_list.append(log_loss(y_val, prob))
            oof_prob[val_idx] = prob

        # Son tam eğitim
        model.fit(X, y_enc)
        results[name]   = {"acc": np.mean(acc_list), "std": np.std(acc_list), "ll": np.mean(ll_list), "model": model}
        oof_preds[name] = oof_prob

    progress_bar.progress(1.0, text="✅ Eğitim tamamlandı!")

    # Ensemble
    weights = {n: results[n]["acc"] for n in selected}
    ens_prob = sum(weights[n] * oof_preds[n] for n in selected) / sum(weights.values())
    ens_pred = ens_prob.argmax(axis=1)
    results["Ensemble"] = {
        "acc": accuracy_score(y_enc, ens_pred),
        "std": 0.0,
        "ll":  log_loss(y_enc, ens_prob),
        "model": None
    }
    oof_preds["Ensemble"] = ens_prob

    best = max(results, key=lambda k: results[k]["acc"])
    return results, oof_preds, best


def predict_upcoming(upcoming_df, imputer, results, oof_preds, selected, le):
    feats = [f for f in FEATURES if f in upcoming_df.columns]
    X_up  = upcoming_df[feats].reindex(columns=FEATURES)
    X_imp = pd.DataFrame(imputer.transform(X_up), columns=FEATURES)

    weights = {n: results[n]["acc"] for n in selected}
    probs   = sum(weights[n] * results[n]["model"].predict_proba(X_imp)
                  for n in selected if results[n]["model"] is not None)
    probs  /= sum(w for n, w in weights.items() if results[n]["model"] is not None)

    preds    = le.inverse_transform(probs.argmax(axis=1))
    top_prob = probs.max(axis=1)
    return probs, preds, top_prob


# ─────────────────────────────────────────────
# RENK PALETİ
# ─────────────────────────────────────────────
C = dict(bg="#0f1117", card="#1a1d26", text="#e2e8f0", muted="#64748b",
         green="#22c55e", amber="#f59e0b", blue="#3b82f6", red="#ef4444",
         c1="#3b82f6", cx="#f59e0b", c2="#ef4444")

PLOTLY_LAYOUT = dict(
    paper_bgcolor=C["bg"], plot_bgcolor=C["card"],
    font=dict(color=C["text"], size=12),
    margin=dict(t=50, b=40, l=40, r=20),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
)


# ─────────────────────────────────────────────
# UYGULAMA ANA AKIŞ
# ─────────────────────────────────────────────
if uploaded is None:
    st.title("⚽ İddaa Tahmin Modeli")
    st.info("Sol panelden **veri dosyasını yükle** ve tahminler otomatik hesaplanır.")
    st.markdown("""
    #### Nasıl Çalışır?
    1. Sol panelden `.txt` dosyasını yükle
    2. Model ayarlarını seç (varsayılan iyi çalışır)
    3. Eğitim otomatik başlar (~30 sn)
    4. 4 sekme: **Dashboard · Tahminler · Geçmiş Maçlar · Model Analizi**

    #### Desteklenen Veri
    - `iddaa_100hafta_merged` formatı
    - Her hafta aynı format, farklı veriler — sorunsuz çalışır
    - Oynanan + yaklaşan maçları otomatik ayırt eder
    """)
    st.stop()

# ── Veriyi yükle ve parse et ──────────────────
with st.spinner("Veri okunuyor..."):
    content  = uploaded.read()
    df_raw   = parse_iddaa(content)
    df       = build_features(df_raw)

played_df   = df[df["played"] & df["result"].notna()].reset_index(drop=True)
upcoming_df = df[~df["played"]].reset_index(drop=True)

if len(played_df) < 20:
    st.error(f"Yeterli oynanan maç bulunamadı ({len(played_df)}). Daha kapsamlı bir dosya yükleyin.")
    st.stop()

if not selected_models:
    st.warning("Sol panelden en az 1 model seçin.")
    st.stop()

# ── Feature matrix ────────────────────────────
feats_avail = [f for f in FEATURES if f in played_df.columns]
X_all  = played_df[feats_avail].reindex(columns=FEATURES)
y_all  = played_df["result"]
le     = LabelEncoder()
y_enc  = le.fit_transform(y_all)
imputer = SimpleImputer(strategy="mean")
X_imp  = pd.DataFrame(imputer.fit_transform(X_all), columns=FEATURES)

# ── Model eğitimi ─────────────────────────────
progress = st.progress(0, text="Modeller eğitiliyor...")
results, oof_preds, best_name = train_models(X_imp, y_enc, le, selected_models, cv_folds, progress)
progress.empty()

# ── Upcoming tahminleri ───────────────────────
if len(upcoming_df) > 0:
    probs_up, preds_up, top_probs_up = predict_upcoming(
        upcoming_df, imputer, results, oof_preds, selected_models, le
    )
else:
    probs_up = preds_up = top_probs_up = None

# ── OOF ve tahmin sonuçları ───────────────────
best_oof   = oof_preds[best_name]
best_preds = le.inverse_transform(best_oof.argmax(axis=1))
best_conf  = best_oof.max(axis=1)

# ─────────────────────────────────────────────
# SEKMELER
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "⚽ Tahminler", "✅ Geçmiş Maçlar", "🔬 Model Analizi"])


# ══════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════
with tab1:
    st.markdown(f"### 📂 {uploaded.name}  —  {len(df)} maç")

    # Metrik kartları
    acc = accuracy_score(y_all, best_preds)
    n_high = (best_conf >= confidence_threshold/100).sum()
    avg_margin = (played_df["margin"].mean() - 1) * 100

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    metrics = [
        (col1, len(df),         "Toplam Maç",          f"{len(played_df)} sonuç", "blue"),
        (col2, f"{acc*100:.1f}%", "Model Doğruluğu",   f"{best_name}",             "green" if acc>=0.55 else "amber"),
        (col3, n_high,          f"Güven ≥{confidence_threshold}%", f"{len(upcoming_df)} bekliyor", "green"),
        (col4, f"{avg_margin:.1f}%", "Bookmaker Marjı", "Ort. aşım",              "amber"),
        (col5, f"{(y_all=='1').sum()/len(y_all)*100:.0f}%", "Ev Sahibi %", f"{(y_all=='1').sum()} maç", "blue"),
        (col6, f"{(y_all=='X').sum()/len(y_all)*100:.0f}%", "Beraberlik %", f"{(y_all=='X').sum()} maç", "amber"),
    ]
    for col, val, lbl, note, cls in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{lbl}</div>
              <div class="metric-value {cls}">{val}</div>
              <div class="metric-note">{note}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        # Sonuç dağılımı
        rc = y_all.value_counts()
        fig = go.Figure(go.Pie(
            labels=rc.index.tolist(), values=rc.values.tolist(),
            marker_colors=[C["c1"], C["cx"], C["c2"]],
            hole=0.42, textinfo="label+percent+value",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Sonuç Dağılımı (Oynanan)", height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Oran aralığına göre doğruluk
        bins   = [(1.0,1.4),(1.4,1.7),(1.7,2.1),(2.1,2.6),(2.6,9)]
        b_lbls = ["1.0–1.4","1.4–1.7","1.7–2.1","2.1–2.6","2.6+"]
        b_accs, b_ns = [], []
        for lo, hi in bins:
            mask = (played_df["min_odds"] >= lo) & (played_df["min_odds"] < hi)
            sy = y_all.values[mask.values]; sp = best_preds[mask.values]
            b_accs.append(accuracy_score(sy, sp)*100 if len(sy) else 0)
            b_ns.append(len(sy))
        fig = go.Figure(go.Bar(
            x=[f"{l}<br>n={n}" for l,n in zip(b_lbls,b_ns)], y=b_accs,
            marker_color=[C["green"] if a>=60 else C["amber"] if a>=50 else C["red"] for a in b_accs],
            text=[f"{a:.0f}%" for a in b_accs], textposition="outside",
        ))
        fig.add_hline(y=55, line_dash="dash", line_color="gray", annotation_text="55% baseline")
        fig.update_layout(**PLOTLY_LAYOUT, title="Favori Oranına Göre Doğruluk",
                          height=320, yaxis=dict(range=[0,100], gridcolor="rgba(255,255,255,0.06)"))
        st.plotly_chart(fig, use_container_width=True)

    # Model karşılaştırma
    all_names = list(results.keys())
    all_accs  = [results[n]["acc"]*100 for n in all_names]
    all_lls   = [results[n]["ll"] for n in all_names]

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        fig = go.Figure(go.Bar(
            x=all_names, y=all_accs,
            marker_color=[C["green"] if a>=57 else C["amber"] if a>=53 else C["red"] for a in all_accs],
            text=[f"{a:.1f}%" for a in all_accs], textposition="outside",
        ))
        fig.add_hline(y=55, line_dash="dash", line_color="gray")
        fig.update_layout(**PLOTLY_LAYOUT, title="Model Doğruluk Karşılaştırması",
                          height=300, yaxis=dict(range=[40,75], gridcolor="rgba(255,255,255,0.06)"))
        st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        fig = go.Figure(go.Bar(
            x=all_names, y=all_lls,
            marker_color=C["blue"],
            text=[f"{l:.3f}" for l in all_lls], textposition="outside",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Log-Loss Karşılaştırması (düşük = iyi)",
                          height=300, yaxis=dict(gridcolor="rgba(255,255,255,0.06)"))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 — TAHMİNLER
# ══════════════════════════════════════════════
with tab2:
    if upcoming_df is None or len(upcoming_df) == 0:
        st.info("Bu dosyada henüz oynanmamış maç bulunmuyor.")
    else:
        # Tahmin tablosu oluştur
        pred_rows = []
        for i, row in upcoming_df.iterrows():
            prob = probs_up[i] if i < len(probs_up) else None
            if prob is None:
                continue
            pred_cls = preds_up[i]
            top_p    = top_probs_up[i]
            conf_lbl = "🔴 Düşük" if top_p < 0.50 else ("🟡 Orta" if top_p < confidence_threshold/100 else "🟢 Yüksek")

            # OU tahmini
            ou_pred = "-"
            if "p_over25" in row and pd.notna(row.get("p_over25")):
                ou_pred = "Üstü" if row["p_over25"] > 0.5 else "Altı"

            pred_rows.append({
                "Lig":        row["league"],
                "Ev Sahibi":  row["home"],
                "Deplasman":  row["away"],
                "1 Oranı":    row["o1"],
                "X Oranı":    row["ox"],
                "2 Oranı":    row["o2"],
                "P(1)%":      f"{prob[list(le.classes_).index('1')]*100:.1f}" if '1' in le.classes_ else "-",
                "P(X)%":      f"{prob[list(le.classes_).index('X')]*100:.1f}" if 'X' in le.classes_ else "-",
                "P(2)%":      f"{prob[list(le.classes_).index('2')]*100:.1f}" if '2' in le.classes_ else "-",
                "Tahmin":     pred_cls,
                "Güven%":     round(top_p * 100, 1),
                "Güven":      conf_lbl,
                "A/Ü 2.5":    ou_pred,
            })

        pred_df = pd.DataFrame(pred_rows)

        # Filtreler uygula
        if filter_league:
            pred_df = pred_df[pred_df["Lig"].str.contains(filter_league, case=False, na=False)]
        if filter_pred != "Tümü":
            pred_df = pred_df[pred_df["Tahmin"] == filter_pred]
        if filter_ou != "Tümü":
            pred_df = pred_df[pred_df["A/Ü 2.5"] == filter_ou]

        pred_df = pred_df.sort_values("Güven%", ascending=False)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Toplam Tahmin", len(pred_df))
        col_b.metric(f"Güven ≥{confidence_threshold}%", (pred_df["Güven%"] >= confidence_threshold).sum())
        col_c.metric("A/Ü Üstü", (pred_df["A/Ü 2.5"] == "Üstü").sum())

        # Renk haritası için stil
        def color_pred(val):
            if val == "1":   return "background-color: #1e3a5f; color: #60a5fa"
            if val == "X":   return "background-color: #3b2f00; color: #fbbf24"
            if val == "2":   return "background-color: #3b1515; color: #f87171"
            return ""

        styled = pred_df.style.applymap(color_pred, subset=["Tahmin"])
        st.dataframe(styled, use_container_width=True, height=600)

        # CSV indir
        csv = pred_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("⬇️ CSV İndir", csv, "tahminler.csv", "text/csv")

        # Scatter: güven vs oran
        fig = go.Figure()
        for cls, clr in [("1", C["c1"]), ("X", C["cx"]), ("2", C["c2"])]:
            sub = pred_df[pred_df["Tahmin"] == cls]
            if len(sub) == 0: continue
            odds_col = {"1":"1 Oranı","X":"X Oranı","2":"2 Oranı"}[cls]
            fig.add_trace(go.Scatter(
                x=sub[odds_col], y=sub["Güven%"],
                mode="markers", name=f"Tahmin {cls}",
                marker=dict(color=clr, size=8, opacity=0.7),
                text=sub["Ev Sahibi"] + " vs " + sub["Deplasman"],
                hoverinfo="text+x+y"
            ))
        fig.add_hline(y=confidence_threshold, line_dash="dash", line_color=C["green"],
                      annotation_text=f"%{confidence_threshold} eşiği")
        fig.update_layout(**PLOTLY_LAYOUT, title="Güven Skoru vs Oran",
                          xaxis_title="Tahmin Edilen Sonucun Oranı",
                          yaxis_title="Güven %", height=380)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — GEÇMİŞ MAÇLAR
# ══════════════════════════════════════════════
with tab3:
    hist_rows = []
    for i, row in played_df.iterrows():
        prob = best_oof[i]
        pred = best_preds[i]
        true = row["result"]
        hit  = pred == true
        hist_rows.append({
            "Lig":        row["league"],
            "Ev Sahibi":  row["home"],
            "Deplasman":  row["away"],
            "Skor":       row["score"],
            "Gerçek":     true,
            "Tahmin":     pred,
            "Güven%":     round(prob.max()*100, 1),
            "Sonuç":      "✅ Doğru" if hit else "❌ Yanlış",
        })

    hist_df = pd.DataFrame(hist_rows)

    # Filtre
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        lg_f = st.text_input("Lig", placeholder="filtrele...", key="hist_lg")
    with col_f2:
        hit_f = st.selectbox("Sonuç", ["Tümü","✅ Doğru","❌ Yanlış"], key="hist_hit")
    with col_f3:
        pred_f = st.selectbox("Tahmin", ["Tümü","1","X","2"], key="hist_pred")

    if lg_f:
        hist_df = hist_df[hist_df["Lig"].str.contains(lg_f, case=False, na=False)]
    if hit_f != "Tümü":
        hist_df = hist_df[hist_df["Sonuç"] == hit_f]
    if pred_f != "Tümü":
        hist_df = hist_df[hist_df["Tahmin"] == pred_f]

    col_a, col_b, col_c = st.columns(3)
    n_hit = (hist_df["Sonuç"] == "✅ Doğru").sum()
    col_a.metric("Gösterilen Maç", len(hist_df))
    col_b.metric("Doğru", n_hit)
    col_c.metric("Doğruluk", f"{n_hit/max(len(hist_df),1)*100:.1f}%")

    st.dataframe(hist_df, use_container_width=True, height=550)

    # ROI
    roi_cum, roi_list = 0, []
    for _, r in hist_df.iterrows():
        o = played_df.loc[played_df["home"]==r["Ev Sahibi"]].head(1)
        if len(o) == 0: continue
        odds = o["o1"].values[0] if r["Tahmin"]=="1" else (o["ox"].values[0] if r["Tahmin"]=="X" else o["o2"].values[0])
        roi_cum += (odds-1) if r["Sonuç"]=="✅ Doğru" else -1
        roi_list.append(roi_cum)

    if roi_list:
        fig = go.Figure(go.Scatter(
            y=roi_list, mode="lines",
            line=dict(color=C["green"] if roi_list[-1]>=0 else C["red"], width=2),
            fill="tozeroy",
            fillcolor="rgba(34,197,94,0.08)" if roi_list[-1]>=0 else "rgba(239,68,68,0.08)"
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(**PLOTLY_LAYOUT, title="Kümülatif ROI (1 birim/maç)",
                          xaxis_title="Maç", yaxis_title="Birim", height=300)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 4 — MODEL ANALİZİ
# ══════════════════════════════════════════════
with tab4:
    st.subheader("Olasılık Kalibrasyonu")
    from sklearn.calibration import calibration_curve

    cols_cal = st.columns(3)
    for idx, cls in enumerate(le.classes_):
        y_bin    = (y_all == cls).astype(int).values
        prob_pos = best_oof[:, idx]
        try:
            frac, mean_pred = calibration_curve(y_bin, prob_pos, n_bins=7, strategy="quantile")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mean_pred, y=frac, mode="lines+markers",
                                     line=dict(color=C["blue"], width=2), name="Model"))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                     line=dict(color="gray", dash="dash"), name="Mükemmel"))
            fig.update_layout(**PLOTLY_LAYOUT, title=f"Sınıf: {cls}", height=280,
                              xaxis_title="Tahmini Olasılık", yaxis_title="Gerçek Oran")
            cols_cal[idx].plotly_chart(fig, use_container_width=True)
        except:
            cols_cal[idx].warning(f"Sınıf {cls} için kalibrasyon hesaplanamadı.")

    st.divider()
    st.subheader("Feature Önem Sıralaması")

    # Tree-based modelden feature importance al
    fi_data = None
    for name in ["Random Forest", "XGBoost", "LightGBM", "Gradient Boosting"]:
        if name in results and results[name]["model"] is not None:
            m = results[name]["model"]
            fi = m.feature_importances_ if hasattr(m, "feature_importances_") else None
            if fi is not None:
                fi_data = pd.DataFrame({"Feature": FEATURES, "Önem": fi})
                fi_data = fi_data.sort_values("Önem", ascending=True).tail(20)
                st.caption(f"Kaynak: {name}")
                break

    if fi_data is not None:
        fig = go.Figure(go.Bar(
            y=fi_data["Feature"], x=fi_data["Önem"], orientation="h",
            marker_color=C["blue"], opacity=0.8,
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Top 20 Feature Importance",
                          xaxis_title="Önem Skoru", height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Lig Bazlı Performans")
    league_rows = []
    for lg in played_df["league"].unique():
        mask = played_df["league"] == lg
        if mask.sum() < 3: continue
        sy = y_all.values[mask.values]
        sp = best_preds[mask.values]
        league_rows.append({"Lig": lg, "Maç": mask.sum(),
                             "Doğru": (sy==sp).sum(),
                             "Doğruluk%": round((sy==sp).mean()*100,1)})

    lg_df = pd.DataFrame(league_rows).sort_values("Doğruluk%", ascending=False)
    fig = go.Figure(go.Bar(
        y=lg_df["Lig"], x=lg_df["Doğruluk%"], orientation="h",
        marker_color=[C["green"] if a>=60 else C["amber"] if a>=50 else C["red"] for a in lg_df["Doğruluk%"]],
        text=lg_df.apply(lambda r: f"{r['Doğruluk%']}% ({r['Maç']} maç)", axis=1),
        textposition="outside",
    ))
    fig.add_vline(x=55, line_dash="dash", line_color="gray")
    fig.update_layout(**PLOTLY_LAYOUT, title="Lig Bazlı Doğruluk (≥3 maç)",
                      height=max(300, len(lg_df)*25), xaxis=dict(range=[0,100]))
    st.plotly_chart(fig, use_container_width=True)
