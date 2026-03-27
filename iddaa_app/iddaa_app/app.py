import re
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

st.set_page_config(page_title="İddaa Tahmin Modeli", page_icon="⚽",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .stApp{background-color:#0f1117}
  .block-container{padding-top:1.5rem}
  h1,h2,h3{color:#e2e8f0}
  .mc{background:#1a1d26;border:1px solid rgba(255,255,255,0.08);
      border-radius:10px;padding:16px 20px;text-align:center}
  .ml{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1px}
  .mv{font-size:32px;font-weight:700;margin:4px 0}
  .mn{font-size:12px;color:#94a3b8}
  .green{color:#22c55e}.amber{color:#f59e0b}.blue{color:#3b82f6}.red{color:#ef4444}
</style>""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────
with st.sidebar:
    st.title("⚽ İddaa Tahmin")
    st.caption("Haftalık dosya yükle → Tahmin al")
    st.divider()
    uploaded = st.file_uploader("📂 Veri Dosyası (.txt)", type=["txt"])
    st.divider()
    st.subheader("⚙️ Model Ayarları")
    selected_models = st.multiselect(
        "Modeller",
        ["Logistic Regression","Random Forest","XGBoost","LightGBM","Gradient Boosting"],
        default=["Random Forest","XGBoost"],
    )
    cv_folds   = st.slider("CV Katlama", 3, 10, 3)
    conf_thr   = st.slider("Güven Eşiği (%)", 50, 75, 60)
    min_league = st.slider("Lig Min Maç (Encoding)", 10, 100, 30,
                           help="Bu sayıdan az maçı olan ligler 'OTHER' grubuna alınır")
    st.divider()
    st.subheader("🔍 Filtreler")
    filter_league = st.text_input("Lig", placeholder="ör: TÜR S")
    filter_pred   = st.selectbox("Tahmin", ["Tümü","1","X","2"])
    filter_ou     = st.selectbox("A/Ü 2.5", ["Tümü","Üstü","Altı"])

# ── PARSE ─────────────────────────────────────
def safe_float(lst, idx):
    try:
        v = lst[idx].strip()
        return float(v) if v and v not in ("-","") else np.nan
    except:
        return np.nan

def extract_detail(s):
    out = {}
    for key, pat in [
        ("iy",   r"İlk Yarı Sonucu \d+ 1([\d.]+) 0([\d.]+) 2([\d.]+)"),
        ("sy",   r"İkinci Yarı Sonucu \d+ 1([\d.]+) 0([\d.]+) 2([\d.]+)"),
        ("ou15", r"1\.5 Altı/Üstü \d+ Alt([\d.]+) Üst([\d.]+)"),
        ("ou35", r"3\.5 Altı/Üstü \d+ Alt([\d.]+) Üst([\d.]+)"),
        ("tc",   r"Tek / Çift \d+ Tek([\d.]+) Çift([\d.]+)"),
        ("tg",   r"Toplam Gol \d+ 0-1 ([\d.]+) 2-3 ([\d.]+)"),
        ("dc",   r"Çifte Şans \d+ 1/X([\d.]+) 1/2([\d.]+) 0/2([\d.]+)"),
    ]:
        m = re.search(pat, s)
        if not m: continue
        g = [float(x) for x in m.groups()]
        if   key=="iy":   out.update(iy1=g[0],iy0=g[1],iy2=g[2])
        elif key=="sy":   out.update(sy1=g[0],sy0=g[1],sy2=g[2])
        elif key=="ou15": out.update(ou15u=g[0],ou15o=g[1])
        elif key=="ou35": out.update(ou35u=g[0],ou35o=g[1])
        elif key=="tc":   out.update(odd_o=g[0],even_o=g[1])
        elif key=="tg":   out.update(tg01=g[0],tg23=g[1])
        elif key=="dc":   out.update(dc_1x=g[0],dc_12=g[1],dc_x2=g[2])
    return out

@st.cache_data(show_spinner=False)
def parse_iddaa(content: bytes) -> pd.DataFrame:
    lines   = content.decode("utf-8", errors="ignore").splitlines()
    records = []

    for i, line in enumerate(lines):
        parts = line.strip().split("\t")
        time_str = league = home = away = score_raw = ht_raw = None
        o1_idx   = None

        # FORMAT A: tarih \t saat \t boş \t lig \t num \t ev \t skor \t dep \t iy \t o1...
        if (len(parts) >= 12
                and re.match(r"\d{2}\.\d{2}\.\d{4}", parts[0])
                and re.match(r"^\d{2}:\d{2}$", parts[1].strip())):
            time_str  = parts[1].strip()
            league    = parts[3].strip() if len(parts) > 3 else ""
            home      = parts[5].strip() if len(parts) > 5 else ""
            score_raw = parts[6].strip() if len(parts) > 6 else ""
            away      = parts[7].strip() if len(parts) > 7 else ""
            ht_raw    = parts[8].strip() if len(parts) > 8 else ""
            o1_idx    = 9

        # FORMAT B: saat \t boş \t lig \t num \t ev \t skor \t dep \t iy \t o1...
        elif (len(parts) >= 10
              and re.match(r"^\d{2}:\d{2}$", parts[0].strip())):
            time_str  = parts[0].strip()
            league    = parts[2].strip() if len(parts) > 2 else ""
            home      = parts[4].strip() if len(parts) > 4 else ""
            score_raw = parts[5].strip() if len(parts) > 5 else ""
            away      = parts[6].strip() if len(parts) > 6 else ""
            ht_raw    = parts[7].strip() if len(parts) > 7 else ""
            o1_idx    = 8
        else:
            continue

        played = bool(re.match(r"^\d+-\d+$", score_raw or ""))
        o1 = safe_float(parts, o1_idx)
        ox = safe_float(parts, o1_idx + 1)
        o2 = safe_float(parts, o1_idx + 2)
        if np.isnan(o1) or np.isnan(ox) or np.isnan(o2):
            continue

        detail_str = ""
        if i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if "Çifte Şans" in nxt or "İlk Yarı" in nxt:
                detail_str = nxt

        rec = {
            "league":   league,
            "home":     home,
            "away":     away,
            "score":    score_raw if played else None,
            "ht_score": ht_raw    if played else None,
            "played":   played,
            "o1": o1, "ox": ox, "o2": o2,
            "ou25u": safe_float(parts, o1_idx + 7),
            "ou25o": safe_float(parts, o1_idx + 8),
        }
        rec.update(extract_detail(detail_str))
        records.append(rec)

    if not records:
        st.error("❌ Dosyadan hiç maç okunamadı.")
        st.stop()

    df = pd.DataFrame(records)
    for col in ["o1","ox","o2","ou25u","ou25o","iy1","iy0","iy2",
                "sy1","sy0","sy2","ou15u","ou15o","ou35u","ou35o",
                "odd_o","even_o","tg01","tg23","dc_1x","dc_12","dc_x2"]:
        if col not in df.columns:
            df[col] = np.nan

    df = df[df["o1"].notna() & df["ox"].notna() & df["o2"].notna()].reset_index(drop=True)
    if len(df) == 0:
        st.error("❌ Geçerli oran verisi bulunamadı.")
        st.stop()
    return df

# ── FEATURE ENGINEERING ───────────────────────
def build_features(df):
    d = df.copy()
    d["r1"]=1/d["o1"]; d["rx"]=1/d["ox"]; d["r2"]=1/d["o2"]
    d["margin"]=d["r1"]+d["rx"]+d["r2"]
    d["p1"]=d["r1"]/d["margin"]; d["px"]=d["rx"]/d["margin"]; d["p2"]=d["r2"]/d["margin"]
    d["fav_prob"]   =d[["p1","px","p2"]].max(axis=1)
    d["second_prob"]=d[["p1","px","p2"]].apply(lambda r:sorted(r)[-2],axis=1)
    d["fav_gap"]    =d["fav_prob"]-d["second_prob"]
    d["home_away_ratio"]=d["p1"]/(d["p2"]+1e-9)
    d["draw_pull"]  =d["px"]/(d["p1"]+d["p2"]+1e-9)
    d["entropy"]    =-(d["p1"]*np.log(d["p1"]+1e-9)+d["px"]*np.log(d["px"]+1e-9)+d["p2"]*np.log(d["p2"]+1e-9))
    d["log_o1"]=np.log(d["o1"]); d["log_ox"]=np.log(d["ox"]); d["log_o2"]=np.log(d["o2"])
    d["o1_o2_spread"]=d["o1"]-d["o2"]; d["min_odds"]=d[["o1","o2"]].min(axis=1)

    m=d["ou25u"].notna()&d["ou25o"].notna()
    d.loc[m,"ou25_tot"]=1/d.loc[m,"ou25u"]+1/d.loc[m,"ou25o"]
    d.loc[m,"p_over25"]=(1/d.loc[m,"ou25o"])/d.loc[m,"ou25_tot"]
    d.loc[m,"p_under25"]=(1/d.loc[m,"ou25u"])/d.loc[m,"ou25_tot"]
    d.loc[m,"ou25_margin"]=d.loc[m,"ou25_tot"]-1

    for uc,oc,nm in [("ou15u","ou15o","p_over15"),("ou35u","ou35o","p_over35")]:
        mk=d[uc].notna()&d[oc].notna()
        d.loc[mk,nm]=(1/d.loc[mk,oc])/(1/d.loc[mk,uc]+1/d.loc[mk,oc])

    m=d["iy1"].notna()&d["iy0"].notna()&d["iy2"].notna()
    d.loc[m,"iy_tot"]=1/d.loc[m,"iy1"]+1/d.loc[m,"iy0"]+1/d.loc[m,"iy2"]
    d.loc[m,"iy_p1"]=(1/d.loc[m,"iy1"])/d.loc[m,"iy_tot"]
    d.loc[m,"iy_px"]=(1/d.loc[m,"iy0"])/d.loc[m,"iy_tot"]
    d.loc[m,"iy_p2"]=(1/d.loc[m,"iy2"])/d.loc[m,"iy_tot"]
    d["iy_1_delta"]=d["iy_p1"]-d["p1"]
    d["iy_2_delta"]=d["iy_p2"]-d["p2"]
    d["iy_x_delta"]=d["iy_px"]-d["px"]

    m=d["sy1"].notna()&d["sy0"].notna()&d["sy2"].notna()
    d.loc[m,"sy_tot"]=1/d.loc[m,"sy1"]+1/d.loc[m,"sy0"]+1/d.loc[m,"sy2"]
    d.loc[m,"sy_p1"]=(1/d.loc[m,"sy1"])/d.loc[m,"sy_tot"]
    d.loc[m,"sy_p2"]=(1/d.loc[m,"sy2"])/d.loc[m,"sy_tot"]

    m=d["dc_1x"].notna()&d["dc_x2"].notna()&d["dc_12"].notna()
    d.loc[m,"dc_p1"]=1/d.loc[m,"dc_1x"]-1/d.loc[m,"dc_x2"]+1/d.loc[m,"dc_12"]
    d.loc[m,"dc_p2"]=1/d.loc[m,"dc_x2"]-1/d.loc[m,"dc_1x"]+1/d.loc[m,"dc_12"]

    m=d["odd_o"].notna()&d["even_o"].notna()
    d.loc[m,"p_odd"]=(1/d.loc[m,"odd_o"])/(1/d.loc[m,"odd_o"]+1/d.loc[m,"even_o"])

    m=d["tg01"].notna()&d["tg23"].notna()
    d.loc[m,"tg_tot"]=1/d.loc[m,"tg01"]+1/d.loc[m,"tg23"]
    d.loc[m,"p_low_g"]=(1/d.loc[m,"tg01"])/d.loc[m,"tg_tot"]

    def res(s):
        try: h,a=map(int,s.split("-")); return "1" if h>a else ("X" if h==a else "2")
        except: return None
    def goals(s):
        try: h,a=map(int,s.split("-")); return h+a
        except: return np.nan

    d["result"]     =d["score"].apply(res)
    d["total_goals"]=d["score"].apply(goals)
    d["over25"]     =(d["total_goals"]>2.5).astype(float)
    return d

BASE_FEATURES=[
    "p1","px","p2","fav_prob","second_prob","fav_gap",
    "home_away_ratio","draw_pull","entropy",
    "log_o1","log_ox","log_o2","o1_o2_spread","min_odds","margin",
    "p_over25","p_under25","ou25_margin","p_over15","p_over35",
    "iy_p1","iy_px","iy_p2","iy_1_delta","iy_2_delta","iy_x_delta",
    "sy_p1","sy_p2","dc_p1","dc_p2","p_odd","p_low_g",
]

def get_model(name):
    if name=="Logistic Regression":
        return Pipeline([("sc",StandardScaler()),("clf",LogisticRegression(C=1.0,max_iter=1000,random_state=42))])
    if name=="Random Forest":
        return RandomForestClassifier(n_estimators=100,max_depth=8,min_samples_leaf=5,class_weight="balanced",random_state=42,n_jobs=-1)
    if name=="XGBoost":
        return xgb.XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,eval_metric="mlogloss",random_state=42,n_jobs=-1,verbosity=0)
    if name=="LightGBM":
        return lgb.LGBMClassifier(n_estimators=100,max_depth=5,learning_rate=0.05,num_leaves=31,subsample=0.8,colsample_bytree=0.8,class_weight="balanced",random_state=42,n_jobs=-1,verbose=-1)
    if name=="Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=100,max_depth=4,learning_rate=0.05,subsample=0.8,random_state=42)

def train_models(X,y_enc,le,selected,cv_folds,prog):
    skf=StratifiedKFold(n_splits=cv_folds,shuffle=True,random_state=42)
    results={}; oof_preds={}
    for idx,name in enumerate(selected):
        prog.progress(idx/len(selected), text=f"Eğitiliyor: {name}...")
        model=get_model(name); oof=np.zeros((len(X),3)); accs=[]; lls=[]
        for tr,val in skf.split(X,y_enc):
            model.fit(X.iloc[tr],y_enc[tr])
            p=model.predict(X.iloc[val]); pr=model.predict_proba(X.iloc[val])
            accs.append(accuracy_score(y_enc[val],p)); lls.append(log_loss(y_enc[val],pr))
            oof[val]=pr
        model.fit(X,y_enc)
        results[name]={"acc":np.mean(accs),"std":np.std(accs),"ll":np.mean(lls),"model":model}
        oof_preds[name]=oof
    prog.progress(1.0,text="✅ Tamamlandı!")
    w={n:results[n]["acc"] for n in selected}; tw=sum(w.values())
    ep=sum(w[n]*oof_preds[n] for n in selected)/tw
    results["Ensemble"]={"acc":accuracy_score(y_enc,ep.argmax(axis=1)),"std":0,"ll":log_loss(y_enc,ep),"model":None}
    oof_preds["Ensemble"]=ep
    best=max(results,key=lambda k:results[k]["acc"])
    return results,oof_preds,best

def predict_up(up_df, imp, results, selected, le, le_lig, league_counts, min_league, feature_cols):
    # Lig encoding — eğitimle aynı mantık
    up_encoded = up_df['league'].map(
        lambda x: x if league_counts.get(x, 0) >= min_league else 'OTHER'
    )
    up_encoded = up_encoded.map(
        lambda x: x if x in le_lig.classes_ else 'OTHER'
    )
    up_lig = le_lig.transform(up_encoded)

    X_up = up_df[BASE_FEATURES].reindex(columns=BASE_FEATURES).copy()
    X_up['league_code'] = up_lig
    X_up = X_up[feature_cols]  # Eğitimdekiyle aynı kolon sırası
    Xi = pd.DataFrame(imp.transform(X_up), columns=feature_cols)

    w={n:results[n]["acc"] for n in selected if results[n]["model"] is not None}
    p=sum(w[n]*results[n]["model"].predict_proba(Xi) for n in w)/sum(w.values())
    return p, le.inverse_transform(p.argmax(axis=1)), p.max(axis=1)

# ── PLOTLY YARDIMCI ───────────────────────────
BG=dict(paper_bgcolor="#0f1117",plot_bgcolor="#1a1d26",font=dict(color="#e2e8f0",size=12),margin=dict(t=50,b=40,l=40,r=20))
GR=dict(gridcolor="rgba(255,255,255,0.06)",zerolinecolor="rgba(255,255,255,0.1)")
C=dict(green="#22c55e",amber="#f59e0b",blue="#3b82f6",red="#ef4444",c1="#3b82f6",cx="#f59e0b",c2="#ef4444")

def lay(fig,title="",h=350,xa=None,ya=None):
    fig.update_layout(**BG,title=title,height=h,
                      xaxis={**GR,**(xa or {})},yaxis={**GR,**(ya or {})})

def bc(v,hi=60,lo=50):
    return C["green"] if v>=hi else (C["amber"] if v>=lo else C["red"])

# ── ANA AKIŞ ──────────────────────────────────
if uploaded is None:
    st.title("⚽ İddaa Tahmin Modeli")
    st.info("Sol panelden **veri dosyasını yükle** → tahminler otomatik hesaplanır.")
    st.markdown("""
    **Desteklenen formatlar:**
    - Eski format (tarih sütunu başta)
    - Yeni format (saat sütunu başta)
    - Her ikisi de otomatik algılanır.
    
    **Yeni özellik:** Lig bazlı encoding — model her ligin dinamiğini ayrı öğrenir.
    """)
    st.stop()

with st.spinner("Veri okunuyor..."):
    df_raw = parse_iddaa(uploaded.read())
    df     = build_features(df_raw)

played_df  = df[df["played"]&df["result"].notna()].reset_index(drop=True)
upcoming_df= df[~df["played"]].reset_index(drop=True)

if len(played_df)<10:
    st.error(f"Yeterli oynanan maç yok ({len(played_df)}). Daha büyük dosya yükleyin.")
    st.stop()
if not selected_models:
    st.warning("Sol panelden en az 1 model seçin.")
    st.stop()

# ── Lig Encoding ──────────────────────────────
league_counts = played_df['league'].value_counts().to_dict()

played_df['league_encoded'] = played_df['league'].map(
    lambda x: x if league_counts.get(x, 0) >= min_league else 'OTHER'
)
le_lig = LabelEncoder()
lig_feature = le_lig.fit_transform(played_df['league_encoded'])

# ── Feature Matrix ────────────────────────────
X_base = played_df[BASE_FEATURES].reindex(columns=BASE_FEATURES).copy()
X_base['league_code'] = lig_feature
FEATURES = BASE_FEATURES + ['league_code']

y_all  = played_df["result"]
le     = LabelEncoder()
y_enc  = le.fit_transform(y_all)
imp    = SimpleImputer(strategy="mean")
X_imp  = pd.DataFrame(imp.fit_transform(X_base), columns=FEATURES)

# ── Eğitim ───────────────────────────────────
prog  = st.progress(0, text="Eğitiliyor...")
res, oof, best = train_models(X_imp, y_enc, le, selected_models, cv_folds, prog)
prog.empty()

# ── Upcoming Tahminleri ───────────────────────
probs_up = preds_up = tops_up = None
if len(upcoming_df) > 0:
    try:
        probs_up, preds_up, tops_up = predict_up(
            upcoming_df, imp, res, selected_models, le,
            le_lig, league_counts, min_league, FEATURES
        )
    except Exception as e:
        st.warning(f"Tahmin sırasında hata: {e}")

best_oof   = oof[best]
best_preds = le.inverse_transform(best_oof.argmax(axis=1))
best_conf  = best_oof.max(axis=1)

# ── SEKMELER ──────────────────────────────────
t1,t2,t3,t4 = st.tabs(["📊 Dashboard","⚽ Tahminler","✅ Geçmiş Maçlar","🔬 Model Analizi"])

# ── TAB 1 ─────────────────────────────────────
with t1:
    st.markdown(f"### 📂 {uploaded.name} — {len(df)} maç")
    acc   = accuracy_score(y_all, best_preds)
    n_hi  = int((best_conf >= conf_thr/100).sum())
    avg_m = (played_df["margin"].mean()-1)*100
    n_leagues = played_df['league'].nunique()

    cols=st.columns(6)
    for col,(val,lbl,note,cls) in zip(cols,[
        (len(df),"Toplam Maç",f"{len(played_df)} sonuç","blue"),
        (f"{acc*100:.1f}%","Model Doğruluğu",best,"green" if acc>=0.55 else "amber"),
        (n_hi,f"Güven ≥{conf_thr}%",f"{len(upcoming_df)} bekliyor","green"),
        (f"{avg_m:.1f}%","Bookmaker Marjı","Ort. aşım","amber"),
        (f"{(y_all=='1').mean()*100:.0f}%","Ev Sahibi %",f"{(y_all=='1').sum()} maç","blue"),
        (n_leagues,"Lig Sayısı",f"≥{min_league} maç encoding","amber"),
    ]):
        with col:
            st.markdown(f'<div class="mc"><div class="ml">{lbl}</div><div class="mv {cls}">{val}</div><div class="mn">{note}</div></div>',unsafe_allow_html=True)

    st.divider()
    cl,cr = st.columns(2)

    with cl:
        rc = y_all.value_counts()
        fig = go.Figure(go.Pie(labels=rc.index.tolist(),values=rc.values.tolist(),
                               marker_colors=[C["c1"],C["cx"],C["c2"]],hole=0.42,textinfo="label+percent+value"))
        lay(fig,"Sonuç Dağılımı",h=320); st.plotly_chart(fig,use_container_width=True)

    with cr:
        bins=[(1.0,1.4),(1.4,1.7),(1.7,2.1),(2.1,2.6),(2.6,9.0)]
        lbls=["1.0–1.4","1.4–1.7","1.7–2.1","2.1–2.6","2.6+"]
        accs2=[]; ns=[]
        for lo,hi in bins:
            mk=(played_df["min_odds"]>=lo)&(played_df["min_odds"]<hi)
            sy=y_all.values[mk.values]; sp=best_preds[mk.values]
            accs2.append(accuracy_score(sy,sp)*100 if len(sy) else 0); ns.append(len(sy))
        fig=go.Figure(go.Bar(x=[f"{l}<br>n={n}" for l,n in zip(lbls,ns)],y=accs2,
                             marker_color=[bc(a) for a in accs2],text=[f"{a:.0f}%" for a in accs2],textposition="outside"))
        fig.add_hline(y=55,line_dash="dash",line_color="gray",annotation_text="55% baseline")
        lay(fig,"Favori Oranına Göre Doğruluk",h=320,ya=dict(range=[0,105]))
        st.plotly_chart(fig,use_container_width=True)

    cl2,cr2=st.columns(2)
    names_r=list(res.keys()); accs_r=[res[n]["acc"]*100 for n in names_r]; lls_r=[res[n]["ll"] for n in names_r]

    with cl2:
        fig=go.Figure(go.Bar(x=names_r,y=accs_r,marker_color=[bc(a) for a in accs_r],
                             text=[f"{a:.1f}%" for a in accs_r],textposition="outside"))
        fig.add_hline(y=55,line_dash="dash",line_color="gray")
        lay(fig,"Model Doğruluk Karşılaştırması",h=300,ya=dict(range=[40,80]))
        st.plotly_chart(fig,use_container_width=True)

    with cr2:
        fig=go.Figure(go.Bar(x=names_r,y=lls_r,marker_color=C["blue"],
                             text=[f"{l:.3f}" for l in lls_r],textposition="outside"))
        lay(fig,"Log-Loss (düşük = iyi)",h=300); st.plotly_chart(fig,use_container_width=True)

# ── TAB 2 ─────────────────────────────────────
with t2:
    if probs_up is None:
        st.info("Oynanmamış maç bulunmuyor veya tahmin hesaplanamadı.")
    else:
        cls_list=list(le.classes_); rows=[]
        for i in range(len(upcoming_df)):
            row=upcoming_df.iloc[i]; prob=probs_up[i]
            pred=preds_up[i]; top=tops_up[i]
            cl_lbl="🟢 Yüksek" if top>=conf_thr/100 else ("🟡 Orta" if top>=0.50 else "🔴 Düşük")
            ou="-"
            if "p_over25" in row.index and pd.notna(row.get("p_over25")):
                ou="Üstü" if row["p_over25"]>0.5 else "Altı"
            rows.append({
                "Lig":row["league"],"Ev Sahibi":row["home"],"Deplasman":row["away"],
                "1 Oranı":row["o1"],"X Oranı":row["ox"],"2 Oranı":row["o2"],
                "P(1)%":round(prob[cls_list.index("1")]*100,1) if "1" in cls_list else 0,
                "P(X)%":round(prob[cls_list.index("X")]*100,1) if "X" in cls_list else 0,
                "P(2)%":round(prob[cls_list.index("2")]*100,1) if "2" in cls_list else 0,
                "Tahmin":pred,"Güven%":round(top*100,1),"Güven":cl_lbl,"A/Ü 2.5":ou,
            })

        pdf=pd.DataFrame(rows)
        if filter_league: pdf=pdf[pdf["Lig"].str.contains(filter_league,case=False,na=False)]
        if filter_pred!="Tümü": pdf=pdf[pdf["Tahmin"]==filter_pred]
        if filter_ou!="Tümü":   pdf=pdf[pdf["A/Ü 2.5"]==filter_ou]
        pdf=pdf.sort_values("Güven%",ascending=False)

        ca,cb,cc=st.columns(3)
        ca.metric("Toplam",len(pdf))
        cb.metric(f"Güven ≥{conf_thr}%",(pdf["Güven%"]>=conf_thr).sum())
        cc.metric("A/Ü Üstü",(pdf["A/Ü 2.5"]=="Üstü").sum())

        st.dataframe(pdf,use_container_width=True,height=520)
        st.download_button("⬇️ CSV İndir",pdf.to_csv(index=False,encoding="utf-8-sig"),"tahminler.csv","text/csv")

        fig=go.Figure()
        for cls,clr in [("1",C["c1"]),("X",C["cx"]),("2",C["c2"])]:
            sub=pdf[pdf["Tahmin"]==cls]
            if len(sub)==0: continue
            oc={"1":"1 Oranı","X":"X Oranı","2":"2 Oranı"}[cls]
            fig.add_trace(go.Scatter(x=sub[oc],y=sub["Güven%"],mode="markers",name=f"Tahmin {cls}",
                                     marker=dict(color=clr,size=8,opacity=0.7),
                                     text=sub["Ev Sahibi"]+" vs "+sub["Deplasman"],hoverinfo="text+x+y"))
        fig.add_hline(y=conf_thr,line_dash="dash",line_color=C["green"],annotation_text=f"%{conf_thr} eşiği")
        lay(fig,"Güven vs Oran",h=380,xa=dict(title="Oran"),ya=dict(title="Güven %"))
        st.plotly_chart(fig,use_container_width=True)

# ── TAB 3 ─────────────────────────────────────
with t3:
    hrows=[]
    for i,row in played_df.iterrows():
        prob=best_oof[i]; pred=best_preds[i]; true=row["result"]
        hrows.append({"Lig":row["league"],"Ev Sahibi":row["home"],"Deplasman":row["away"],
                      "Skor":row["score"],"Gerçek":true,"Tahmin":pred,
                      "Güven%":round(prob.max()*100,1),
                      "Sonuç":"✅ Doğru" if pred==true else "❌ Yanlış"})
    hdf=pd.DataFrame(hrows)

    hf1,hf2,hf3=st.columns(3)
    with hf1: lf=st.text_input("Lig filtre","",key="h_lg")
    with hf2: hf=st.selectbox("Sonuç",["Tümü","✅ Doğru","❌ Yanlış"],key="h_hit")
    with hf3: pf=st.selectbox("Tahmin",["Tümü","1","X","2"],key="h_p")

    if lf:         hdf=hdf[hdf["Lig"].str.contains(lf,case=False,na=False)]
    if hf!="Tümü": hdf=hdf[hdf["Sonuç"]==hf]
    if pf!="Tümü": hdf=hdf[hdf["Tahmin"]==pf]

    nh=(hdf["Sonuç"]=="✅ Doğru").sum()
    ha,hb,hc=st.columns(3)
    ha.metric("Maç",len(hdf)); hb.metric("Doğru",nh); hc.metric("Doğruluk",f"{nh/max(len(hdf),1)*100:.1f}%")
    st.dataframe(hdf,use_container_width=True,height=480)

    roi_c=0.0; rl=[]
    for _,r in hdf.iterrows():
        mk=(played_df["home"]==r["Ev Sahibi"])&(played_df["away"]==r["Deplasman"])
        sub=played_df[mk]
        if len(sub)==0: continue
        o=sub.iloc[0]
        odds=o["o1"] if r["Tahmin"]=="1" else (o["ox"] if r["Tahmin"]=="X" else o["o2"])
        roi_c+=(odds-1) if r["Sonuç"]=="✅ Doğru" else -1
        rl.append(round(roi_c,2))
    if rl:
        col=C["green"] if rl[-1]>=0 else C["red"]
        fig=go.Figure(go.Scatter(y=rl,mode="lines",line=dict(color=col,width=2),fill="tozeroy",
                                  fillcolor=f"rgba({'34,197,94' if rl[-1]>=0 else '239,68,68'},0.08)"))
        fig.add_hline(y=0,line_dash="dash",line_color="gray")
        lay(fig,"Kümülatif ROI (1 birim/maç)",h=280,xa=dict(title="Maç"),ya=dict(title="Birim"))
        st.plotly_chart(fig,use_container_width=True)

# ── TAB 4 ─────────────────────────────────────
with t4:
    st.subheader("Olasılık Kalibrasyonu")
    ccols=st.columns(3)
    for idx,cls in enumerate(le.classes_):
        yb=(y_all==cls).astype(int).values; pp=best_oof[:,idx]
        try:
            fr,mp=calibration_curve(yb,pp,n_bins=7,strategy="quantile")
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=mp,y=fr,mode="lines+markers",line=dict(color=C["blue"],width=2),name="Model"))
            fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",line=dict(color="gray",dash="dash"),name="Mükemmel"))
            lay(fig,f"Sınıf: {cls}",h=260,xa=dict(title="Tahmini Olasılık"),ya=dict(title="Gerçek Oran"))
            ccols[idx].plotly_chart(fig,use_container_width=True)
        except: ccols[idx].warning(f"{cls} için yeterli veri yok.")

    st.divider(); st.subheader("Feature Önem Sıralaması")
    fi_data=None
    for nm in ["Random Forest","XGBoost","LightGBM","Gradient Boosting"]:
        if nm in res and res[nm]["model"]:
            fi=getattr(res[nm]["model"],"feature_importances_",None)
            if fi is not None:
                fi_data=pd.DataFrame({"Feature":FEATURES,"Önem":fi}).sort_values("Önem",ascending=True).tail(20)
                st.caption(f"Kaynak: {nm}"); break
    if fi_data is not None:
        fig=go.Figure(go.Bar(y=fi_data["Feature"],x=fi_data["Önem"],orientation="h",marker_color=C["blue"],opacity=0.8))
        lay(fig,"Top 20 Feature Importance",h=500,xa=dict(title="Önem Skoru"))
        st.plotly_chart(fig,use_container_width=True)

  st.divider(); st.subheader("Lig Bazlı Performans")

    lgr=[]
    for lg in played_df["league"].unique():
        mk=played_df["league"]==lg
        if mk.sum()<3: continue
        sy=y_all.values[mk.values]; sp=best_preds[mk.values]
        lgr.append({
            "Lig":      lg,
            "Maç":      int(mk.sum()),
            "Doğru":    int((sy==sp).sum()),
            "Doğruluk%":round((sy==sp).mean()*100,1),
            "Ev%":      round((sy=="1").mean()*100,1),
            "Ber%":     round((sy=="X").mean()*100,1),
            "Dep%":     round((sy=="2").mean()*100,1),
        })

    if lgr:
        lgdf = pd.DataFrame(lgr).sort_values("Doğruluk%", ascending=False)

        # Filtre
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            min_mac = st.slider("Min Maç Sayısı", 3, 50, 10, key="lg_min")
        with col_f2:
            sort_by = st.selectbox("Sıralama", ["Doğruluk% ↓","Maç ↓","Lig A-Z"], key="lg_sort")

        lgdf = lgdf[lgdf["Maç"] >= min_mac]
        if sort_by == "Maç ↓":       lgdf = lgdf.sort_values("Maç", ascending=False)
        elif sort_by == "Lig A-Z":   lgdf = lgdf.sort_values("Lig")
        else:                         lgdf = lgdf.sort_values("Doğruluk%", ascending=False)

        # Özet metrikler
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Toplam Lig", len(lgdf))
        m2.metric("Başarılı Lig (≥60%)", (lgdf["Doğruluk%"]>=60).sum())
        m3.metric("Orta (50-60%)",        ((lgdf["Doğruluk%"]>=50)&(lgdf["Doğruluk%"]<60)).sum())
        m4.metric("Zayıf (<50%)",          (lgdf["Doğruluk%"]<50).sum())

        # Yatay bar grafik
        fig = go.Figure(go.Bar(
            y=lgdf["Lig"],
            x=lgdf["Doğruluk%"],
            orientation="h",
            marker_color=[bc(a) for a in lgdf["Doğruluk%"]],
            text=lgdf.apply(lambda r: f"{r['Doğruluk%']}% ({r['Maç']} maç)", axis=1),
            textposition="outside",
        ))
        fig.add_vline(x=55, line_dash="dash", line_color="gray", annotation_text="55%")
        fig.add_vline(x=50, line_dash="dot",  line_color="rgba(255,255,255,0.2)")
        lay(fig, "Lig Bazlı Model Doğruluğu",
            h=max(400, len(lgdf)*22),
            xa=dict(range=[0, 110], title="Doğruluk %"))
        st.plotly_chart(fig, use_container_width=True)

        # Detay tablo
        st.markdown("#### Detay Tablo")
        st.dataframe(
            lgdf.style.background_gradient(
                subset=["Doğruluk%"], cmap="RdYlGn", vmin=30, vmax=75
            ),
            use_container_width=True,
            height=400
        )

    # Lig encoding özeti
    st.divider(); st.subheader("Lig Encoding Özeti")
    enc_df = pd.DataFrame({
        "Lig": list(league_counts.keys()),
        "Maç Sayısı": list(league_counts.values())
    }).sort_values("Maç Sayısı", ascending=False)
    enc_df["Encoding"] = enc_df["Maç Sayısı"].apply(
        lambda x: "✅ Ayrı Grup" if x >= min_league else "⚪ OTHER"
    )
    st.dataframe(enc_df.head(50), use_container_width=True, height=400)
