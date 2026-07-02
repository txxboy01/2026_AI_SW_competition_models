import csv
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

DATA_DIR = "./data"

def load_and_process_data():
    print("[*] Loading train data...")
    samples = [json.loads(line) for line in open(os.path.join(DATA_DIR, "train.jsonl"), encoding="utf-8") if line.strip()]
    labels = {row["id"]: row["action"] for row in csv.DictReader(open(os.path.join(DATA_DIR, "train_labels.csv"), encoding="utf-8"))}

    parsed_rows = []
    y = []

    for s in samples:
        s_id = s["id"]
        if s_id not in labels: continue
            
        current_prompt = s.get("current_prompt", "")
        meta = s.get("session_meta", {})
        history = s.get("history", [])
        
        # 1위 치트키: 직전 행동 컨텍스트 반영
        last_action = "none"
        if history:
            for h in reversed(history):
                if h.get("role") == "assistant" and "name" in h:
                    last_action = h.get("name", "none")
                    break
        
        combined_text = f"[LAST_ACT: {last_action}] {current_prompt}"
        
        # 세션 메타 정형 피처 6개 추출
        tier_idx = {'enterprise': 0, 'pro': 1, 'free': 2}.get(meta.get("user_tier", "free"), 2)
        budget = meta.get("budget_tokens_remaining", 0)
        turn_index = meta.get("turn_index", 0)
        elapsed = meta.get("elapsed_session_sec", 0)
        workspace = meta.get("workspace", {})
        loc = workspace.get("loc", 0)
        git_dirty = 1 if workspace.get("git_dirty") is True else 0
        
        parsed_rows.append({
            'text': combined_text,
            'meta_features': [tier_idx, budget, turn_index, elapsed, loc, git_dirty]
        })
        y.append(labels[s_id])

    return parsed_rows, y

def main():
    data_rows, y = load_and_process_data()
    texts = [row['text'] for row in data_rows]
    meta_feats = np.array([row['meta_features'] for row in data_rows])
    y = np.array(y)
    
    # 텍스트 벡터화
    print("[*] Vectorizing text...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True, lowercase=True)
    X_text = vectorizer.fit_transform(texts).toarray()
    X_total = np.hstack([X_text, meta_feats])
    
    X_train, X_val, y_train, y_val = train_test_split(X_total, y, test_size=0.2, stratify=y, random_state=42)

    # 🛠️ 경진대회 최적화 하이퍼파라미터 튜닝 세팅
    SCORING_METRIC = 'f1_macro' # 대회 평가지표 고정
    N_ITER = 5 # 10분 컷을 위한 고속 샘플링 개수 제한

    # [튜닝 1] Random Forest
    print("\n🔍 Random Forest Tuning...")
    rf_base = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    rf_param = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
    rf_search = RandomizedSearchCV(rf_base, rf_param, n_iter=N_ITER, scoring=SCORING_METRIC, cv=3, random_state=42, n_jobs=-1)
    rf_search.fit(X_train, y_train)

    # [튜닝 2] LightGBM
    print("\n🔍 LightGBM Tuning...")
    lgbm_base = LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1, n_jobs=-1)
    lgbm_param = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [5, 10]}
    lgbm_search = RandomizedSearchCV(lgbm_base, lgbm_param, n_iter=N_ITER, scoring=SCORING_METRIC, cv=3, random_state=42, n_jobs=-1)
    lgbm_search.fit(X_train, y_train)

    # 🏁 최강의 조합: 두 최적 모델 앙상블 (Soft Voting)
    print("\n" + "="*50)
    print("🔥 소프트 보팅 앙상블 모델 결합 중...")
    ensemble = VotingClassifier(
        estimators=[('rf', rf_search.best_estimator_), ('lgbm', lgbm_search.best_estimator_)],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    
    # 검증 점수 출력
    val_preds = ensemble.predict(X_val)
    score = f1_score(y_val, val_preds, labels=ALL_CLASSES, average='macro', zero_division=0)
    print(f"🏆 최종 검증셋 Macro-F1 점수: {score:.4f}")
    print("="*50)

    # 전체 데이터 재학습 및 저장 (제출 규격 완벽 대응)
    print("[*] Saving final components for submit.zip ...")
    ensemble.fit(X_total, y)
    
    os.makedirs("./model", exist_ok=True)
    joblib.dump(ensemble, "./model/tfidf_logreg.pkl", compress=3) # 기존 script.py와 이름 동기화
    joblib.dump(vectorizer, "./model/vectorizer.pkl", compress=3)
    print("[+] 완료! 이제 모델 파일들을 압축해서 리더보드에 지르세요!")

if __name__ == "__main__":
    main()
