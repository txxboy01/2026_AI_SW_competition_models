import csv
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# 예측 대상 14개 클래스
ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

DATA_DIR = "./data"

# -------------------------------------------------------------------------
# 1. 고도화된 데이터 로드 및 피쳐 추출 함수
# -------------------------------------------------------------------------
def load_and_process_data():
    print("[*] Loading train.jsonl ...")
    samples = [json.loads(line)
               for line in open(os.path.join(DATA_DIR, "train.jsonl"), encoding="utf-8")
               if line.strip()]

    print("[*] Loading train_labels.csv ...")
    labels = {row["id"]: row["action"]
              for row in csv.DictReader(open(os.path.join(DATA_DIR, "train_labels.csv"), encoding="utf-8"))}

    parsed_rows = []
    y = []

    for s in samples:
        s_id = s["id"]
        if s_id not in labels:
            continue
            
        current_prompt = s.get("current_prompt", "")
        meta = s.get("session_meta", {})
        history = s.get("history", [])
        
        # [피쳐 1] 히스토리에서 직전 에이전트 행동(Action) 요약 추출 (인과관계 학습용) ⭐
        last_action = "none"
        if history:
            for h in reversed(history):
                if h.get("role") == "assistant" and "name" in h:
                    last_action = h.get("name", "none")
                    break
        
        # 텍스트 데이터 결합 컨텍스트 구축
        combined_text = f"[LAST_ACT: {last_action}] {current_prompt}"
        
        # [피쳐 2] 세션 메타 수치형 및 범주형 데이터 추출 ⭐
        user_tier = meta.get("user_tier", "free")
        tier_map = {'enterprise': 0, 'pro': 1, 'free': 2}
        tier_idx = tier_map.get(user_tier, 2)
        
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

# -------------------------------------------------------------------------
# 2. 메인 학습 파이프라인
# -------------------------------------------------------------------------
def main():
    # 데이터 로드
    data_rows, y = load_and_process_data()
    
    texts = [row['text'] for row in data_rows]
    meta_feats = np.array([row['meta_features'] for row in data_rows])
    y = np.array(y)
    
    print(f"Total samples: {len(texts)} | Total features: {meta_feats.shape[1]} tabular cols")

    # 학습 / 검증 데이터 분할 (정형 데이터 인덱스 동기화)
    indices = np.arange(len(texts))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
    
    # 텍스트 백터화 모델 선언 (베이스라인과 동일 사양)
    print("[*] Vectorizing text data...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=80_000,
        sublinear_tf=True, lowercase=True
    )
    
    # Train 데이터 기반 단어 사전 구축 및 변환
    X_train_text = vectorizer.fit_transform([texts[i] for i in train_idx]).toarray()
    X_train_total = np.hstack([X_train_text, meta_feats[train_idx]])
    y_train = y[train_idx]
    
    # Validation 데이터 변환
    X_val_text = vectorizer.transform([texts[i] for i in val_idx]).toarray()
    X_val_total = np.hstack([X_val_text, meta_feats[val_idx]])
    y_val = y[val_idx]

    # 모델 정의 및 학습
    print("[*] Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,          # 다중 피쳐 융합으로 수렴 안정성을 위해 max_iter 상향
        class_weight="balanced", 
        C=2.0,                  # 규제 강도
        n_jobs=-1               # 고사양 내 PC 자원 활용 가속화
    )
    model.fit(X_train_total, y_train)

    # 검증 수행
    val_pred = model.predict(X_val_total)
    macro_f1 = f1_score(y_val, val_pred, labels=ALL_CLASSES, average="macro", zero_division=0)
    print(f"▼ Validation Macro-F1: {macro_f1:.4f}")

    # -------------------------------------------------------------------------
    # 3. 전체 데이터로 재학습 & 저장 (추론 스크립트 대응을 위한 분리 저장)
    # -------------------------------------------------------------------------
    print("[*] Re-training with full data for final submission...")
    
    # 전체 데이터 변환 및 결합
    X_full_text = vectorizer.fit_transform(texts).toarray()
    X_full_total = np.hstack([X_full_text, meta_feats])
    
    model.fit(X_full_total, y)
    
    # 저장 경로 설정
    os.makedirs("./model", exist_ok=True)
    
    # [참고] 복합 피쳐를 사용하므로 객체를 각각 분리해서 저장하는 것이 추론 단계에서 훨씬 안정적입니다.
    joblib.dump(model, "./model/tfidf_logreg.pkl", compress=3)
    joblib.dump(vectorizer, "./model/vectorizer.pkl", compress=3)
    print("[+] Model and Vectorizer saved successfully inside ./model/")

if __name__ == "__main__":
    main()
