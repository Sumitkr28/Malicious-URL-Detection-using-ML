#!/usr/bin/env python3
"""malicious_stack.py (readable output version)
Stacking ensemble pipeline for multiclass malicious-URL detection.

Improvements:
- When training, saves label encoder and class names into the stacked model file.
- When predicting, prints human-readable class names (e.g., PHISHING) instead of numeric codes.
- Shows top contributing base models and quick heuristic reasons.
- CLI: --train, --url, --file
"""

import os, sys, re, argparse, pickle, time
from urllib.parse import urlparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
import xgboost as xgb
from catboost import CatBoostClassifier

EXCEL_PATH = 'malicious_ds.xlsx'

# ----------------------------- Feature engineering helpers -----------------------------
def having_ip_address(url):
    return 1 if re.search(r'(([0-9]{1,3}\.){3}[0-9]{1,3})', str(url)) else 0

def abnormal_url(url):
    try:
        hostname = urlparse(str(url)).hostname
        if not hostname:
            return 0
        return 1 if re.search(re.escape(str(hostname)), str(url)) else 0
    except:
        return 0

def count_dot(url): return str(url).count('.')
def count_www(url): return str(url).count('www')
def count_atrate(url): return str(url).count('@')
def no_of_dir(url): return urlparse(str(url)).path.count('/')
def no_of_embed(url): return urlparse(str(url)).path.count('//')
def shortening_service(url): return 1 if re.search(r'bit\.ly|goo\.gl|tinyurl|t\.co|ow\.ly|lnkd\.in|tiny\.cc', str(url)) else 0
def count_https(url): return str(url).count('https')
def count_http(url): return str(url).count('http')
def count_per(url): return str(url).count('%')
def count_ques(url): return str(url).count('?')
def count_hyphen(url): return str(url).count('-')
def count_equal(url): return str(url).count('=')
def url_length(url): return len(str(url))
def hostname_length(url): return len(urlparse(str(url)).netloc)
def suspicious_words(url): return 1 if re.search(r'paypal|login|signin|bank|account|update|free|bonus|ebay|webscr', str(url), re.I) else 0
def digit_count(url): return sum(1 for ch in str(url) if ch.isdigit())
def letter_count(url): return sum(1 for ch in str(url) if ch.isalpha())
def fd_length(url):
    parts = urlparse(str(url)).path.split('/')
    return len(parts[1]) if len(parts) > 1 else 0
def tld_length_safe(url):
    try:
        from tld import get_tld
        tld = get_tld(str(url), fail_silently=True)
        return len(tld) if tld else -1
    except:
        return -1

def extract_features_series(url_series):
    df = pd.DataFrame({'url': url_series.astype(str)})
    df['use_of_ip'] = df['url'].apply(having_ip_address)
    df['abnormal_url'] = df['url'].apply(abnormal_url)
    df['count.'] = df['url'].apply(count_dot)
    df['count-www'] = df['url'].apply(count_www)
    df['count@'] = df['url'].apply(count_atrate)
    df['count_dir'] = df['url'].apply(no_of_dir)
    df['count_embed_domian'] = df['url'].apply(no_of_embed)
    df['short_url'] = df['url'].apply(shortening_service)
    df['count-https'] = df['url'].apply(count_https)
    df['count-http'] = df['url'].apply(count_http)
    df['count%'] = df['url'].apply(count_per)
    df['count?'] = df['url'].apply(count_ques)
    df['count-'] = df['url'].apply(count_hyphen)
    df['count='] = df['url'].apply(count_equal)
    df['url_length'] = df['url'].apply(url_length)
    df['hostname_length'] = df['url'].apply(hostname_length)
    df['sus_url'] = df['url'].apply(suspicious_words)
    df['count-digits'] = df['url'].apply(digit_count)
    df['count-letters'] = df['url'].apply(letter_count)
    df['fd_length'] = df['url'].apply(fd_length)
    df['tld_length'] = df['url'].apply(tld_length_safe)
    return df

# ----------------------------- Stacking training -----------------------------
def train_stacking(X, y, label_encoder, base_models, n_folds=3, random_state=42, model_path='stacked_model.pkl'):
    print(f"Creating out-of-fold predictions with {n_folds} folds...")
    n_samples = X.shape[0]
    n_classes = len(np.unique(y))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    n_base = len(base_models)
    meta_features = np.zeros((n_samples, n_classes * n_base), dtype=np.float32)

    # create OOF predictions
    for i, (name, model) in enumerate(base_models):
        print(f"Generating OOF predictions for {name} ...")
        oof = np.zeros((n_samples, n_classes), dtype=np.float32)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            m = model.__class__(**model.get_params())
            m.fit(X[train_idx], y[train_idx])
            probs = m.predict_proba(X[val_idx])
            oof[val_idx] = probs
            print(f"  fold {fold+1}/{n_folds} done")
        meta_features[:, i*n_classes:(i+1)*n_classes] = oof

    # train meta-model
    print("Training Logistic Regression meta-model...")
    meta_clf = LogisticRegression(multi_class='multinomial', max_iter=1000, solver='lbfgs')
    meta_clf.fit(meta_features, y)

    # retrain base models on full data
    trained_bases = {}
    base_acc = {}
    for name, model in base_models:
        print(f"Retraining {name} on full data...")
        m = model.__class__(**model.get_params())
        m.fit(X, y)
        trained_bases[name] = m
        preds = m.predict(X)
        base_acc[name] = accuracy_score(y, preds)
        print(f"  {name} accuracy (train): {base_acc[name]:.4f}")

    # evaluate meta-model on holdout split for reporting
    X_tr, X_test, y_tr, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=random_state)
    # build test meta-features
    test_meta = np.zeros((X_test.shape[0], n_classes * n_base), dtype=np.float32)
    for i, (name, _) in enumerate(base_models):
        probs = trained_bases[name].predict_proba(X_test)
        test_meta[:, i*n_classes:(i+1)*n_classes] = probs
    meta_preds = meta_clf.predict(test_meta)
    meta_acc = accuracy_score(y_test, meta_preds)
    print(f"Meta-model accuracy (holdout): {meta_acc:.4f}")
    print("Classification report for meta-model:")
    print(classification_report(y_test, meta_preds, target_names=list(label_encoder.classes_)))

    # Save everything
    save_dict = {
        'base_models': trained_bases,
        'meta_model': meta_clf,
        'label_encoder': label_encoder,
        'classes': list(label_encoder.classes_),
        'feature_columns': None,
        'n_classes': n_classes
    }
    with open(model_path, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f"Saved stacked model to {model_path}")

    # print logistic weights nicely
    coef = meta_clf.coef_
    print("\nLogistic Regression coefficients (per class):")
    for class_idx, cname in enumerate(label_encoder.classes_):
        print(f" Class '{cname}' (index {class_idx}):")
        for i, (name, _) in enumerate(base_models):
            start = i * n_classes
            end = (i+1) * n_classes
            weights = coef[class_idx, start:end]
            print(f"  {name}: {np.round(weights, 4)}   mean_abs={float(np.mean(np.abs(weights))):.4f}")
        print("")

    return save_dict, base_acc, meta_acc

# ----------------------------- Base models -----------------------------
base_models = [
    ('RandomForest', RandomForestClassifier(n_estimators=200, random_state=5)),
    ('LightGBM', LGBMClassifier(objective='multiclass', n_estimators=200, random_state=5)),
    ('XGBoost', xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='mlogloss', verbosity=0)),
    ('CatBoost', CatBoostClassifier(iterations=200, verbose=0, random_state=5))
]

# ----------------------------- Prediction helpers -----------------------------
def human_reasons_from_features(feats_df):
    reasons = []
    vals = feats_df.iloc[0].to_dict()
    if vals.get('short_url') == 1:
        reasons.append('Shortened URL detected')
    if vals.get('sus_url') == 1:
        reasons.append('Contains suspicious words (e.g., login, paypal)')
    if vals.get('use_of_ip') == 1:
        reasons.append('Uses IP address instead of domain')
    if vals.get('url_length', 0) > 200:
        reasons.append(f"Very long URL (len={vals.get('url_length')})")
    if not reasons:
        reasons.append('No obvious token-based reason; model used learned patterns.')
    return reasons

def predict_url_readable(url, model_path='stacked_model.pkl'):
    if not os.path.exists(model_path):
        print("Model not found. Train first with --train")
        return
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    trained_bases = data['base_models']
    meta = data['meta_model']
    le = data.get('label_encoder')
    classes = data.get('classes')
    n_classes = data.get('n_classes')

    feats_df = extract_features_series(pd.Series([url]))
    if 'url' in feats_df.columns:
        feats_df = feats_df.drop(columns=['url'])
    Xf = feats_df.values.astype(float)

    # collect probabilities from base models
    probs_list = []
    for name in ['RandomForest','LightGBM','XGBoost','CatBoost']:
        probs = trained_bases[name].predict_proba(Xf)
        probs_list.append(probs)
    meta_features = np.hstack(probs_list)

    pred_code = int(meta.predict(meta_features)[0])
    # map to label name
    label_name = None
    if le is not None:
        try:
            label_name = le.inverse_transform([pred_code])[0]
        except Exception:
            label_name = classes[pred_code] if classes and pred_code < len(classes) else str(pred_code)
    else:
        label_name = classes[pred_code] if classes and pred_code < len(classes) else str(pred_code)

    # compute contributions for that predicted class
    coef = meta.coef_
    class_idx = pred_code
    contributions = {}
    for i, name in enumerate(['RandomForest','LightGBM','XGBoost','CatBoost']):
        start = i * n_classes
        end = (i+1) * n_classes
        weights = coef[class_idx, start:end]
        contribution = float(np.dot(weights, probs_list[i].ravel()))
        contributions[name] = contribution
    sorted_contrib = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)

    # print results
    print("\n================ PREDICTION =================")
    print(f"URL: {url}")
    print(f"Predicted class: {label_name} (code: {pred_code})")
    print("\nTop contributing base models (higher -> more influence):")
    for name, v in sorted_contrib:
        print(f"  {name}: {v:.4f}")
    reasons = human_reasons_from_features(extract_features_series(pd.Series([url])).drop(columns=['url']))
    print("\nHeuristic reasons (quick):")
    for i, r in enumerate(reasons, 1):
        print(f" {i}. {r}")
    print("=============================================\n")

# ----------------------------- CLI -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stacking ensemble (readable outputs)')
    parser.add_argument('--train', action='store_true', help='Train stacking pipeline')
    parser.add_argument('--url', type=str, help='Predict single URL')
    parser.add_argument('--file', type=str, help='Predict URLs from a file (one per line)')
    parser.add_argument('--model', type=str, default='stacked_model.pkl', help='Model file path')
    args = parser.parse_args()

    if args.train:
        if not os.path.exists(EXCEL_PATH):
            print(f"Dataset {EXCEL_PATH} not found in {os.getcwd()}")
            sys.exit(1)
        df = pd.read_excel(EXCEL_PATH)
        print(f"Loaded dataset with shape: {df.shape}")
        if 'url' not in df.columns or 'type' not in df.columns:
            print("Dataset must contain 'url' and 'type' columns")
            sys.exit(1)
        le = LabelEncoder()
        df['type_str'] = df['type'].astype(str).str.lower().str.strip()
        df['type_code'] = le.fit_transform(df['type_str'])
        classes = list(le.classes_)
        print("Classes:", classes)

        # extract features and drop url column from numeric features
        feats = extract_features_series(df['url'])
        if 'url' in feats.columns:
            feats = feats.drop(columns=['url'])
        X = feats.values.astype(float)
        y = df['type_code'].values

        save_dict, base_acc, meta_acc = train_stacking(X, y, le, base_models, n_folds=3, random_state=5, model_path=args.model)
        print("\nTraining complete. Summary:")
        for k,v in base_acc.items():
            print(f" {k}: {v:.4f}")
        print(f" Meta-model accuracy (holdout): {meta_acc:.4f}")
        sys.exit(0)

    if args.url:
        predict_url_readable(args.url, model_path=args.model)
        sys.exit(0)

    if args.file:
        if not os.path.exists(args.file):
            print("File not found:", args.file)
            sys.exit(1)
        with open(args.file, 'r', encoding='utf-8') as fh:
            for line in fh:
                url = line.strip()
                if not url:
                    continue
                predict_url_readable(url, model_path=args.model)
        sys.exit(0)

    print("No action specified. Use --train, --url or --file.")
