# -*- coding: utf-8 -*-
from pathlib import Path
import streamlit as st
import joblib
import shap
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

# ========= 1) 基础设置 =========
matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

st.set_page_config(
    page_title="SLE患者TE风险预测",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========= 2) 加载模型 =========
try:
    BASE_DIR = Path(__file__).resolve().parent  # streamlit运行
except NameError:
    BASE_DIR = Path(r"C:\Users\chanly\Desktop\sle-vte-app")  # jupyter调试

MODEL_PATH = BASE_DIR / "models" / "rf_BSMOTE_stepwise_train_data_GBDT_final.joblib"

st.write(f"当前项目目录：{BASE_DIR}")
st.write(f"模型路径：{MODEL_PATH}")
st.write(f"模型是否存在：{MODEL_PATH.exists()}")

if not MODEL_PATH.exists():
    st.error(f"模型文件不存在：{MODEL_PATH}")
    st.stop()

# ========= 兼容补丁：解决 numpy/joblib 反序列化旧模型报错 =========
import sys

try:
    import numpy.core
    sys.modules["numpy._core"] = numpy.core
except Exception as e:
    st.warning(f"numpy._core 补丁加载失败：{e}")

try:
    import numpy.random._pickle as nrp

    _orig_ctor = nrp.__bit_generator_ctor

    def _patched_bit_generator_ctor(bit_generator_name):
        if isinstance(bit_generator_name, type):
            bit_generator_name = bit_generator_name.__name__
        return _orig_ctor(bit_generator_name)

    nrp.__bit_generator_ctor = _patched_bit_generator_ctor
except Exception as e:
    st.warning(f"MT19937 补丁加载失败：{e}")

# ========= 加载模型 =========
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"模型加载失败：{e}")
    st.stop()

try:
    explainer = shap.TreeExplainer(model)
except Exception as e:
    st.error(f"SHAP 解释器初始化失败：{e}")
    st.stop()

# ========= 3) 特征：网页显示中文 + 单位（用于输入） =========
DISPLAY_LABELS = {
    "Age at Onset": "发病年龄（岁）",
    "Disease Duration": "病程（年）",
    "Statins": "是否服用他汀类药物（0否/1是）",
    "Cardiac Involvement": "是否患有心脏受累（0否/1是）",
    "Arthritis": "是否患有关节炎（0否/1是）",
    "EF": "射血分数 EF（%）",
    "Aspirin": "是否服用阿司匹林（0否/1是）",
    "IVS": "心室间隔 IVS（mm）",
    "MMF": "是否服用吗替麦考酚酯（0否/1是）",
    "Cr": "肌酐（μmol/L）",
}

BINARY_FEATURES = {"Statins", "Cardiac Involvement", "Arthritis", "Aspirin", "MMF"}

CONT_RANGES = {
    "Age at Onset": (0.0, 100.0, 40.0),
    "Disease Duration": (0.0, 50.0, 5.0),
    "EF": (0.0, 100.0, 60.0),
    "IVS": (0.0, 40.0, 12.0),
    "Cr": (0.0, 2000.0, 80.0),
}

REFERENCE_RANGES = {
    "Cr": (44.0, 97.0),
    "IVS": (5.6, 10.6),
    "EF": (50.0, 70.0),
}

def get_status(value: float, ref_range):
    lo, hi = ref_range
    if value < lo:
        return "偏低"
    elif value > hi:
        return "偏高"
    else:
        return "正常"

def build_input_df(user_values: dict) -> pd.DataFrame:
    input_df = pd.DataFrame([user_values])
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        for c in expected:
            if c not in input_df.columns:
                input_df[c] = 0
        input_df = input_df[expected]
    return input_df

# ========= 4) 页面标题 =========
title = "👨‍⚕️系统性红斑狼疮患者血栓栓塞性疾病风险预测👨‍⚕️"
st.markdown(f"<h1 style='text-align:center'>{title}</h1>", unsafe_allow_html=True)
st.divider()
st.markdown(":orange[⚠️请注意输入指标单位；⚠️请将指标输入完整，否则无法计算风险；⚠️预测结果仅供参考，请咨询专业医师。]")

# ========= 5) 输入区 =========
if hasattr(model, "feature_names_in_"):
    FEATURES = list(model.feature_names_in_)
else:
    FEATURES = list(DISPLAY_LABELS.keys())

col1, col2 = st.columns(2, gap="large")
user_values = {}

with col1:
    left_feats = FEATURES[:5]
    for f in left_feats:
        label = DISPLAY_LABELS.get(f, f)
        if f in BINARY_FEATURES:
            user_values[f] = st.selectbox(label, [0, 1], index=0)
        else:
            mn, mx, dv = CONT_RANGES.get(f, (0.0, 9999.0, 0.0))
            user_values[f] = st.number_input(label, min_value=float(mn), max_value=float(mx), value=float(dv))

with col2:
    right_feats = FEATURES[5:]
    for f in right_feats:
        label = DISPLAY_LABELS.get(f, f)
        if f in BINARY_FEATURES:
            user_values[f] = st.selectbox(label, [0, 1], index=0)
        else:
            mn, mx, dv = CONT_RANGES.get(f, (0.0, 9999.0, 0.0))
            user_values[f] = st.number_input(label, min_value=float(mn), max_value=float(mx), value=float(dv))

predict_button = st.button("点击预测", type="primary")

# ========= 6) 预测 + 输出 =========
if predict_button:
    input_df = build_input_df(user_values)

    if hasattr(model, "predict_proba"):
        risk = float(model.predict_proba(input_df)[:, 1][0])
    else:
        risk = float(model.predict(input_df)[0])

    if risk < 0.4:
        risk_level = "低风险"
        color = "green"
    elif risk < 0.7:
        risk_level = "中风险"
        color = "orange"
    else:
        risk_level = "高风险"
        color = "red"

    st.markdown(
        f"<div style='text-align:center; font-size:28px; font-weight:bold;'>"
        f"您未来6个月内发生VTE的风险为："
        f"<span style='color:{color}; font-size:38px;'> {risk:.2%} - {risk_level}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.subheader("以下是个体化风险分析结果", divider="rainbow")

    cols_for_status = [c for c in input_df.columns if c in REFERENCE_RANGES]
    status_rows = []
    for c in cols_for_status:
        v = float(input_df[c].iloc[0])
        status_rows.append({
            "指标": DISPLAY_LABELS.get(c, c),
            "数值": v,
            "状态": get_status(v, REFERENCE_RANGES[c]),
            "参考范围": f"{REFERENCE_RANGES[c][0]}–{REFERENCE_RANGES[c][1]}"
        })
    status_df = pd.DataFrame(status_rows)

    try:
        shap_values = explainer.shap_values(input_df)
        shap_exp = explainer(input_df)
    except Exception as e:
        st.error(f"SHAP 计算失败：{e}")
        shap_values = None
        shap_exp = None

    c1, c2, c3 = st.columns([1.2, 2, 2], gap="medium")

    with c1:
        st.subheader("各指标状态")
        if len(status_df) == 0:
            st.info("当前仅对部分指标设置了参考范围。")
        else:
            st.dataframe(status_df, hide_index=True, use_container_width=True)

    with c2:
        st.subheader("个体化风险因素排名")
        if shap_values is not None:
            fig, ax = plt.subplots(figsize=(6, 6))
            if isinstance(shap_values, list) and len(shap_values) == 2:
                sv = shap_values[1]
            else:
                sv = shap_values
            shap.summary_plot(sv, features=input_df, plot_type="bar", show=False)
            plt.xlabel("影响程度")
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
        else:
            st.warning("暂无 SHAP 重要性图。")

    with c3:
        st.subheader("个体化风险构成")
        if shap_exp is not None:
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            try:
                shap.plots.waterfall(shap_exp[0], show=False)
                plt.tight_layout()
                st.pyplot(fig2, clear_figure=True)
                st.markdown(
                    "📌 :red[红色] 表示该指标增加VTE风险；"
                    " :blue[蓝色] 表示该指标降低/未增加VTE风险；"
                    " :green[数值] 表示影响大小。"
                )
            except Exception as e:
                st.error(f"瀑布图绘制失败：{e}")
        else:
            st.warning("暂无 SHAP 瀑布图。")