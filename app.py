"""
California Housing Price Predictor — Streamlit App
====================================================
KNN Regression · GridSearchCV · scikit-learn Pipeline

How to run:
    pip install streamlit scikit-learn pandas numpy matplotlib seaborn
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(
    page_title="CA Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
    --bg:#0d1b2a; --surface:#132233; --surface2:#1a2e44; --border:#1f3a55;
    --teal:#00b4d8; --teal-dark:#0077b6; --amber:#f4a261; --amber-light:#ffd166;
    --text:#e0eaf5; --muted:#7a97b5; --radius:12px;
}
html,body,[class*="css"]{ font-family:'DM Sans',sans-serif; background-color:var(--bg)!important; color:var(--text)!important; }
.main .block-container{ padding:2rem 2.5rem 3rem; max-width:1300px; }
[data-testid="stSidebar"]{ background:var(--surface)!important; border-right:1px solid var(--border); }
[data-testid="stSidebar"] *{ color:var(--text)!important; }
h1,h2,h3{ font-family:'Space Mono',monospace!important; letter-spacing:-0.03em; }
h1{ color:var(--teal)!important; font-size:1.9rem!important; }
h2{ color:var(--amber)!important; font-size:1.3rem!important; }
h3{ color:var(--text)!important; font-size:1.05rem!important; }
[data-testid="stMetric"]{ background:var(--surface2); border:1px solid var(--border); border-radius:var(--radius); padding:1rem 1.2rem; }
[data-testid="stMetricLabel"]{ color:var(--muted)!important; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; }
[data-testid="stMetricValue"]{ color:var(--teal)!important; font-family:'Space Mono',monospace; font-size:1.6rem; }
[data-testid="stMetricDelta"]{ color:var(--amber)!important; }
.stButton>button{ background:linear-gradient(135deg,var(--teal-dark),var(--teal)); color:#fff!important; border:none; border-radius:var(--radius); padding:0.65rem 1.8rem; font-family:'Space Mono',monospace; font-size:0.85rem; transition:opacity 0.2s,transform 0.15s; width:100%; }
.stButton>button:hover{ opacity:0.88; transform:translateY(-1px); }
.stAlert{ border-radius:var(--radius)!important; }
.pred-card{ background:linear-gradient(135deg,#0e2a45,#0d3b5e); border:2px solid var(--teal); border-radius:16px; padding:2rem; text-align:center; margin:1.2rem 0; }
.pred-value{ font-family:'Space Mono',monospace; font-size:3.2rem; font-weight:700; color:var(--amber-light); line-height:1; }
.pred-label{ font-size:0.85rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.12em; margin-top:0.5rem; }
hr{ border-color:var(--border)!important; margin:1.5rem 0; }
[data-testid="stTabs"] button{ font-family:'Space Mono',monospace!important; font-size:0.82rem; color:var(--muted)!important; }
[data-testid="stTabs"] button[aria-selected="true"]{ color:var(--teal)!important; border-bottom:2px solid var(--teal)!important; }
</style>
""", unsafe_allow_html=True)

BG="0d1b2a"; SURFACE="#132233"; SURFACE2="#1a2e44"; TEAL="#00b4d8"; AMBER="#f4a261"; MUTED="#7a97b5"; TEXT="#e0eaf5"
BG="#0d1b2a"

def dark_fig(figsize=(8,4)):
    fig,ax=plt.subplots(figsize=figsize,facecolor=BG)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED,labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor(SURFACE2)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED); ax.title.set_color(TEXT)
    return fig,ax

MODEL_PATH="california_knn_pipeline.pkl"

@st.cache_resource(show_spinner=False)
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH,"rb") as f: model=pickle.load(f)
        return model,"loaded"
    X,y=fetch_california_housing(return_X_y=True,as_frame=True)
    X_train,_,y_train,_=train_test_split(X,y,test_size=0.2,random_state=42)
    nt=Pipeline([("imputer",SimpleImputer(strategy="mean")),("scaler",StandardScaler())])
    pre=ColumnTransformer([("num",nt,X.columns.tolist())])
    pipe=Pipeline([("preprocessor",pre),("knn",KNeighborsRegressor())])
    gs=GridSearchCV(pipe,{"knn__n_neighbors":[3,5,7,9],"knn__weights":["uniform","distance"],"knn__p":[1,2]},cv=5,scoring="r2",n_jobs=-1)
    gs.fit(X_train,y_train)
    with open(MODEL_PATH,"wb") as f: pickle.dump(gs.best_estimator_,f)
    return gs.best_estimator_,"trained"

@st.cache_data(show_spinner=False)
def load_dataset():
    X,y=fetch_california_housing(return_X_y=True,as_frame=True)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    return X,y,X_train,X_test,y_train,y_test

@st.cache_data(show_spinner=False)
def compute_metrics(_model,X_test,y_test):
    yp=_model.predict(X_test)
    r2=r2_score(y_test,yp); mse=mean_squared_error(y_test,yp); rmse=np.sqrt(mse)
    return yp,r2,mse,rmse

FEATURES={
    "MedInc":   {"label":"Median Income (×$10k)",    "min":0.5,  "max":15.0, "default":3.87,  "step":0.1,  "fmt":"%.2f"},
    "HouseAge": {"label":"Housing Median Age (yrs)", "min":1.0,  "max":52.0, "default":28.6,  "step":1.0,  "fmt":"%.0f"},
    "AveRooms": {"label":"Avg Rooms / Household",    "min":1.0,  "max":20.0, "default":5.43,  "step":0.1,  "fmt":"%.1f"},
    "AveBedrms":{"label":"Avg Bedrooms / Household", "min":0.5,  "max":6.0,  "default":1.10,  "step":0.05, "fmt":"%.2f"},
    "Population":{"label":"Block Population",        "min":5.0,  "max":3500.0,"default":1425.0,"step":10.0,"fmt":"%.0f"},
    "AveOccup": {"label":"Avg Household Size",       "min":1.0,  "max":10.0, "default":3.07,  "step":0.1,  "fmt":"%.2f"},
    "Latitude": {"label":"Latitude",                 "min":32.5, "max":42.0, "default":35.63, "step":0.1,  "fmt":"%.2f"},
    "Longitude":{"label":"Longitude",                "min":-124.3,"max":-114.3,"default":-119.57,"step":0.1,"fmt":"%.2f"},
}
FEATURE_DESCS={
    "MedInc":"Higher income areas typically command much higher house prices.",
    "HouseAge":"Older housing stock can mean established neighbourhoods or higher renovation costs.",
    "AveRooms":"More rooms per household generally correlates with larger, pricier homes.",
    "AveBedrms":"Bedrooms-to-rooms ratio indicates unit density within households.",
    "Population":"Denser block groups can push prices up (urban) or down (overcrowded).",
    "AveOccup":"Higher occupancy may signal rental vs owner-occupied neighbourhoods.",
    "Latitude":"California spans from tropical south to cooler north — location matters.",
    "Longitude":"Coastal vs inland position is a major price driver in California.",
}

st.markdown("# 🏠 California Housing Price Predictor")
st.markdown("<p style='color:#7a97b5;font-size:0.92rem;margin-top:-0.8rem;'>KNN Regression · GridSearchCV · scikit-learn Pipeline</p>",unsafe_allow_html=True)
st.markdown("---")

with st.spinner("Loading model and dataset…"):
    model,model_status=load_or_train_model()
    X,y,X_train,X_test,y_train,y_test=load_dataset()
    y_pred_all,r2,mse,rmse=compute_metrics(model,X_test,y_test)

if model_status=="loaded":
    st.success("✅ Pre-trained pipeline loaded from `california_knn_pipeline.pkl`")
else:
    st.info("ℹ️ No `.pkl` file found — model was trained fresh and saved automatically.")

knn_step=model.named_steps["knn"]
best_params={"n_neighbors":knn_step.n_neighbors,"weights":knn_step.weights,"p":knn_step.p}

with st.sidebar:
    st.markdown("## 🎛️ Input Features")
    st.markdown("<p style='color:#7a97b5;font-size:0.8rem;'>Adjust the sliders then click Predict.</p>",unsafe_allow_html=True)
    st.markdown("---")
    user_inputs={}
    for feat,meta in FEATURES.items():
        user_inputs[feat]=st.slider(label=meta["label"],min_value=float(meta["min"]),max_value=float(meta["max"]),value=float(meta["default"]),step=float(meta["step"]),format=meta["fmt"],help=FEATURE_DESCS[feat])
    st.markdown("---")
    predict_btn=st.button("🔍 Predict Price")

tab_pred,tab_perf,tab_data,tab_about=st.tabs(["🏡 Prediction","📊 Model Performance","🔬 Data Insights","ℹ️ About"])

with tab_pred:
    col_inp,col_res=st.columns([1,1],gap="large")
    with col_inp:
        st.markdown("### Current Input Values")
        input_df=pd.DataFrame([user_inputs])
        st.dataframe(input_df.T.rename(columns={0:"Value"}),use_container_width=True)
    with col_res:
        st.markdown("### Predicted Price")
        if predict_btn:
            prediction=model.predict(input_df)[0]; price_usd=prediction*100_000
            st.markdown(f'<div class="pred-card"><div class="pred-value">${price_usd:,.0f}</div><div class="pred-label">Estimated Median House Value</div><div style="color:#00b4d8;font-family:Space Mono,monospace;font-size:1.1rem;margin-top:0.8rem;">{prediction:.4f} × $100k units</div></div>',unsafe_allow_html=True)
            tier,col=("Budget","#2ec4b6") if prediction<1 else ("Affordable","#00b4d8") if prediction<2 else ("Mid-Range","#f4a261") if prediction<3.5 else ("Premium","#e76f51") if prediction<5 else ("Luxury","#e63946")
            st.markdown(f"<div style='text-align:center;color:{col};font-family:Space Mono,monospace;font-size:0.95rem;margin-top:-0.5rem;'>▸ Market Tier: <b>{tier}</b></div>",unsafe_allow_html=True)
            st.markdown("#### How your inputs compare to dataset range")
            ratios={f:(user_inputs[f]-X[f].min())/(X[f].max()-X[f].min()) for f in FEATURES}
            fig,ax=dark_fig(figsize=(7,3))
            ax.barh(list(ratios.keys()),list(ratios.values()),color=[TEAL if v>=0.5 else AMBER for v in ratios.values()],height=0.55,edgecolor="none")
            ax.set_xlim(0,1); ax.set_xlabel("Normalised value (0=min, 1=max)",fontsize=8)
            ax.axvline(0.5,color=MUTED,linestyle="--",linewidth=0.8,alpha=0.6)
            ax.set_title("Feature Percentile (relative to dataset range)",fontsize=9,color=TEXT); ax.invert_yaxis(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        else:
            st.info("👈 Adjust the sliders in the sidebar and click **Predict Price** to see a result.")

with tab_perf:
    st.markdown("### Model Evaluation Metrics")
    st.markdown("<p style='color:#7a97b5;font-size:0.85rem;'>Evaluated on 20% held-out test set (4,128 samples).</p>",unsafe_allow_html=True)
    m1,m2,m3,m4=st.columns(4)
    m1.metric("R² Score",f"{r2:.4f}",delta=f"{(r2-0.5)*100:.1f}% above baseline")
    m2.metric("RMSE",f"{rmse:.4f}",delta=f"≈ ${rmse*100_000:,.0f} avg error")
    m3.metric("MSE",f"{mse:.4f}"); m4.metric("Test Samples","4,128")
    st.markdown("---"); st.markdown("### Best Hyperparameters (from GridSearchCV)")
    pc1,pc2,pc3=st.columns(3)
    pc1.metric("n_neighbors",str(best_params["n_neighbors"])); pc2.metric("weights",str(best_params["weights"]))
    pc3.metric("p (distance)",f"{best_params['p']}  ({'Manhattan' if best_params['p']==1 else 'Euclidean'})")
    st.markdown("---"); col_a,col_b=st.columns(2)
    with col_a:
        st.markdown("#### Actual vs Predicted Values")
        fig,ax=dark_fig(figsize=(6,5))
        ax.scatter(y_test,y_pred_all,alpha=0.18,s=8,color=TEAL,linewidths=0)
        lims=[min(y_test.min(),y_pred_all.min())-0.1,max(y_test.max(),y_pred_all.max())+0.1]
        ax.plot(lims,lims,color=AMBER,linewidth=1.5,label="Perfect prediction")
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title(f"Actual vs Predicted  (R²={r2:.4f})",color=TEXT,fontsize=10)
        ax.legend(facecolor=SURFACE,edgecolor=MUTED,labelcolor=TEXT,fontsize=8); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with col_b:
        st.markdown("#### Residuals Distribution")
        residuals=y_test.values-y_pred_all
        fig,ax=dark_fig(figsize=(6,5))
        ax.hist(residuals,bins=60,color=TEAL,edgecolor="none",alpha=0.85)
        ax.axvline(0,color=AMBER,linewidth=1.5,linestyle="--",label="Zero error")
        ax.set_xlabel("Residual (Actual − Predicted)"); ax.set_ylabel("Count"); ax.set_title("Residuals Histogram",color=TEXT,fontsize=10)
        ax.legend(facecolor=SURFACE,edgecolor=MUTED,labelcolor=TEXT,fontsize=8)
        ax.annotate(f"Mean:{residuals.mean():.4f}\nStd:{residuals.std():.4f}",xy=(0.98,0.95),xycoords="axes fraction",ha="right",va="top",fontsize=8,color=MUTED,fontfamily="monospace")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

with tab_data:
    st.markdown("### Dataset Overview")
    di1,di2,di3,di4=st.columns(4)
    di1.metric("Total Samples","20,640"); di2.metric("Features","8"); di3.metric("Missing Values","0"); di4.metric("Target Range",f"${y.min()*100:.0f}k – ${y.max()*100:.0f}k")
    st.markdown("---"); col_c,col_d=st.columns(2)
    with col_c:
        st.markdown("#### Target Distribution")
        fig,ax=dark_fig(figsize=(6,4))
        ax.hist(y,bins=60,color=AMBER,edgecolor="none",alpha=0.85)
        ax.axvline(y.mean(),color=TEAL,linewidth=1.5,linestyle="--",label=f"Mean={y.mean():.2f}")
        ax.set_xlabel("Median House Value ($100k)"); ax.set_ylabel("Count"); ax.set_title("Distribution of Median House Value",color=TEXT,fontsize=10)
        ax.legend(facecolor=SURFACE,edgecolor=MUTED,labelcolor=TEXT,fontsize=8); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with col_d:
        st.markdown("#### Feature Correlation with Target")
        df_all=X.copy(); df_all["MedHouseVal"]=y
        corr_target=df_all.corr()["MedHouseVal"].drop("MedHouseVal").sort_values()
        fig,ax=dark_fig(figsize=(6,4))
        colors=[TEAL if v>0 else "#e63946" for v in corr_target.values]
        ax.barh(corr_target.index,corr_target.values,color=colors,height=0.6,edgecolor="none")
        ax.axvline(0,color=MUTED,linewidth=0.8); ax.set_xlabel("Pearson Correlation"); ax.set_title("Feature vs Target Correlation",color=TEXT,fontsize=10)
        ax.invert_yaxis(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.markdown("#### Full Feature Correlation Heatmap")
    fig,ax=plt.subplots(figsize=(10,6),facecolor=BG); ax.set_facecolor(BG)
    corr_matrix=df_all.corr(); mask=np.triu(np.ones_like(corr_matrix,dtype=bool))
    sns.heatmap(corr_matrix,mask=mask,cmap=sns.diverging_palette(200,20,as_cmap=True),vmin=-1,vmax=1,annot=True,fmt=".2f",annot_kws={"size":8,"color":TEXT},linewidths=0.5,linecolor=BG,ax=ax,cbar_kws={"shrink":0.8})
    ax.set_title("Correlation Matrix (lower triangle)",color=TEXT,fontsize=10,pad=12); ax.tick_params(colors=MUTED,labelsize=8)
    plt.setp(ax.get_xticklabels(),rotation=30,ha="right"); plt.setp(ax.get_yticklabels(),rotation=0); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.markdown("#### Sample Records from Dataset")
    st.dataframe(X.head(8).style.background_gradient(cmap="Blues",axis=0),use_container_width=True)

with tab_about:
    st.markdown("### About This App")
    a1,a2=st.columns(2)
    with a1:
        st.markdown("""
**🧰 Tech Stack**
- Python 3.10+
- scikit-learn (Pipeline, GridSearchCV, KNeighborsRegressor)
- Streamlit
- Pandas / NumPy
- Matplotlib / Seaborn
        """)
    with a2:
        st.markdown("""
**🔧 ML Pipeline**
1. `SimpleImputer(strategy='mean')` → handles missing values
2. `StandardScaler()` → normalises all features
3. `KNeighborsRegressor` → distance-based regression
4. `GridSearchCV` (5-fold CV) → tunes `n_neighbors`, `weights`, `p`

**📐 Evaluation Metric**
Primary: **R² Score** · Secondary: MSE, RMSE
        """)
    st.markdown("---"); st.markdown("### Hyperparameter Search Space")
    st.dataframe(pd.DataFrame({
        "Hyperparameter":["n_neighbors","weights","p"],
        "Values Tested":["[3,5,7,9]","['uniform','distance']","[1,2]"],
        "Description":["Number of nearest neighbours","Uniform=equal weight; Distance=closer neighbours weighted more","1=Manhattan, 2=Euclidean"],
    }),use_container_width=True,hide_index=True)
    st.markdown("---")
    st.markdown("<p style='color:#7a97b5;font-size:0.8rem;'>Dataset: California Housing · sklearn.datasets · Source: 1990 U.S. Census · 20,640 block groups</p>",unsafe_allow_html=True)
