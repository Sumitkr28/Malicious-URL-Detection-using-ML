# app.py - SafeTrace Malicious URL Detection
# Final Year Project - Department of CSE - GNIOT Greater Noida
# Run using: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import shap
import matplotlib.pyplot as plt
from malicious_stack import extract_features_series

# page config
st.set_page_config(page_title="SafeTrace", layout="wide")

# light background custom styling
st.markdown("""
    <style>
        .main { background-color: #f9f9f9; }
        .block-container { padding: 2rem 3rem; }
        .result-box {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin-top: 10px;
        }
        .section-box {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin-top: 15px;
        }
        .header-bar {
            background-color: #1a3a5c;
            padding: 18px 30px;
            border-radius: 8px;
            color: white;
            margin-bottom: 20px;
        }
        .footer-bar {
            background-color: #1a3a5c;
            padding: 12px 30px;
            border-radius: 8px;
            color: white;
            text-align: center;
            margin-top: 30px;
            font-size: 13px;
        }
        .label-phishing { color: #c0392b; font-size: 22px; font-weight: bold; }
        .label-malware  { color: #e67e22; font-size: 22px; font-weight: bold; }
        .label-defacement { color: #f39c12; font-size: 22px; font-weight: bold; }
        .label-benign   { color: #27ae60; font-size: 22px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# header
st.markdown("""
    <div class="header-bar">
        <h2 style="margin:0; color:white;">SafeTrace - Malicious URL Detection</h2>
        <p style="margin:4px 0 0 0; font-size:14px; color:#ccd9e8;">
            Final Year Project &nbsp;|&nbsp; Department of CSE &nbsp;|&nbsp; GNIOT Greater Noida
        </p>
    </div>
""", unsafe_allow_html=True)

# about section
st.write("This tool checks whether a given URL is safe or malicious using a stacking ensemble model.")
st.write("The model is trained on 651,191 URLs across 4 categories: benign, phishing, malware, and defacement.")
st.write("SHAP values are used to explain which features contributed most to each prediction.")

st.write("---")

# load model
@st.cache_resource
def load_model(model_path='stacked_model.pkl'):
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data

# prediction function
def predict_and_explain(url, data):
    trained_bases = data['base_models']
    meta = data['meta_model']
    le = data.get('label_encoder')
    classes = data.get('classes')
    n_classes = data.get('n_classes')

    feats_df = extract_features_series(pd.Series([url]))
    if 'url' in feats_df.columns:
        feats_df = feats_df.drop(columns=['url'])
    Xf = feats_df.values.astype(float)

    probs_list = []
    for name in ['RandomForest', 'LightGBM', 'XGBoost', 'CatBoost']:
        probs = trained_bases[name].predict_proba(Xf)
        probs_list.append(probs)
    meta_features = np.hstack(probs_list)

    pred_code = int(meta.predict(meta_features)[0])
    if le is not None:
        try:
            label_name = le.inverse_transform([pred_code])[0]
        except:
            label_name = classes[pred_code]
    else:
        label_name = classes[pred_code]

    meta_proba = meta.predict_proba(meta_features)[0]
    confidence = float(meta_proba[pred_code]) * 100

    explainer = shap.TreeExplainer(trained_bases['RandomForest'])
    shap_values = explainer.shap_values(Xf)
    feat_names = feats_df.columns.tolist()

    if isinstance(shap_values, list):
        if pred_code < len(shap_values):
            class_shap = shap_values[pred_code][0]
        else:
            class_shap = shap_values[0][0]
    else:
        if shap_values.ndim == 3:
            class_shap = shap_values[0, :, pred_code]
        else:
            class_shap = shap_values[0]

    shap_pairs = list(zip(feat_names, class_shap))
    shap_pairs_sorted = sorted(shap_pairs, key=lambda x: abs(x[1]), reverse=True)[:8]

    return label_name, confidence, shap_pairs_sorted, pred_code

# threat info
threat_info = {
    'phishing': {
        'explanation': "Phishing is a cyberattack where criminals create fake websites that impersonate legitimate ones like banks, PayPal, or Google. The goal is to trick users into entering their credentials. These fake sites often look identical to real ones but have slightly altered URLs like paypa1.com instead of paypal.com.",
        'prevention': [
            "Always check the URL carefully before entering any login credentials",
            "Look for HTTPS but note that even phishing sites can have HTTPS",
            "Never click links from unknown emails - go directly to the official website",
            "Enable two factor authentication on all important accounts"
        ]
    },
    'malware': {
        'explanation': "Malware URLs are links that when visited, automatically download harmful software onto your device. This includes viruses, trojans, ransomware, and spyware. These URLs often disguise themselves as software downloads, crack files, or fake update pages.",
        'prevention': [
            "Never download software from unofficial or unknown websites",
            "Keep your operating system and antivirus software always up to date",
            "Do not click on pop-up download buttons or fake update alerts",
            "Use a reputable antivirus program that scans downloads in real time"
        ]
    },
    'defacement': {
        'explanation': "Defacement refers to URLs of websites that have been hacked and their content replaced by attackers, usually to display political messages or promote hacker groups. Defaced sites are often government or educational websites targeted for visibility.",
        'prevention': [
            "Avoid entering personal information on websites that look visually broken",
            "Report defaced websites to the organization that owns them immediately",
            "Do not download anything from a website that appears tampered with",
            "Check official social media to verify if a website is genuinely hacked"
        ]
    },
    'benign': {
        'explanation': "This URL appears to be safe and legitimate. Benign URLs are normal websites like news portals, educational sites, ecommerce platforms, and social media. No suspicious patterns were found in the URL structure or domain features.",
        'prevention': [
            "Even on safe websites always use strong unique passwords",
            "Regularly clear your browser cookies and cache for privacy",
            "Be cautious about what personal information you share online",
            "Keep your browser updated to ensure latest security patches"
        ]
    }
}

# color map for labels
label_colors = {
    'phishing': '#c0392b',
    'malware': '#e67e22',
    'defacement': '#f39c12',
    'benign': '#27ae60'
}

# load model
data = load_model()
if data is None:
    st.write("Model file not found. Please run training first.")
    st.stop()

st.write("Model is ready.")

# url input row
url_input = st.text_input("Enter the URL to check:", placeholder="e.g. http://example.com")
check_btn = st.button("Check URL")

if check_btn:
    if not url_input.strip():
        st.write("Please enter a URL first.")
    else:
        with st.spinner("Analyzing URL..."):
            try:
                label, confidence, shap_pairs, pred_code = predict_and_explain(url_input.strip(), data)
                label_color = label_colors.get(label.lower(), '#333333')

                st.write("---")

                # two column layout for result and shap
                col1, col2 = st.columns([1, 1])

                # left column - prediction result
                with col1:
                    st.markdown('<div class="section-box">', unsafe_allow_html=True)
                    st.subheader("Prediction Result")
                    st.markdown(f'<p style="font-size:26px; font-weight:bold; color:{label_color};">{label.upper()}</p>', unsafe_allow_html=True)
                    st.write(f"Confidence: {confidence:.2f}%")
                    st.progress(int(confidence))

                    st.write("")
                    st.write("URL analyzed:")
                    st.code(url_input.strip())

                    st.write("")
                    st.write("Top contributing features:")
                    for i, (feat, val) in enumerate(shap_pairs[:3], 1):
                        direction = "suspicious" if val > 0 else "normal"
                        st.write(f"{i}. {feat} - {direction}")
                    st.markdown('</div>', unsafe_allow_html=True)

                # right column - shap chart
                with col2:
                    st.markdown('<div class="section-box">', unsafe_allow_html=True)
                    st.subheader("Feature Analysis (SHAP)")
                    st.write("Red bars pushed prediction towards this category. Green bars pushed away.")

                    features = [p[0] for p in shap_pairs]
                    values = [p[1] for p in shap_pairs]
                    bar_colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in values]

                    fig, ax = plt.subplots(figsize=(6, 4))
                    fig.patch.set_facecolor('#ffffff')
                    ax.set_facecolor('#f9f9f9')
                    ax.barh(features[::-1], values[::-1], color=bar_colors[::-1])
                    ax.axvline(x=0, color='black', linewidth=0.8)
                    ax.set_xlabel("SHAP Value")
                    ax.set_title("Feature Contributions")
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.markdown('</div>', unsafe_allow_html=True)

                st.write("---")

                # threat info section - full width below columns
                if label.lower() in threat_info:
                    info_col1, info_col2 = st.columns([1, 1])

                    with info_col1:
                        st.markdown('<div class="section-box">', unsafe_allow_html=True)
                        st.subheader("About This Category")
                        st.write(threat_info[label.lower()]['explanation'])
                        st.markdown('</div>', unsafe_allow_html=True)

                    with info_col2:
                        st.markdown('<div class="section-box">', unsafe_allow_html=True)
                        st.subheader("How To Stay Safe")
                        for i, tip in enumerate(threat_info[label.lower()]['prevention'], 1):
                            st.write(f"{i}. {tip}")
                        st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.write(f"Something went wrong: {str(e)}")

# footer
st.markdown("""
    <div class="footer-bar">
        SafeTrace &nbsp;|&nbsp; Final Year Project &nbsp;|&nbsp; Department of CSE &nbsp;|&nbsp; GNIOT Greater Noida
    </div>
""", unsafe_allow_html=True)