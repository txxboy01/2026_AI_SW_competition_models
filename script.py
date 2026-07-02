import csv
import json
import os
import sys
import joblib
import numpy as np

# -------------------------------------------------------------------------
# 경로 설정 (경로 안정성 확보)
# -------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_DIR = os.path.join(CURRENT_DIR, "data")
MODEL_DIR = os.path.join(CURRENT_DIR, "model")
OUT_DIR = os.path.join(CURRENT_DIR, "output")

TEST_PATH = os.path.join(TEST_DIR, "test.jsonl")
SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "tfidf_logreg.pkl")
VEC_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

REQUIRED_KEYS = ("id", "session_meta", "history", "current_prompt")

# -------------------------------------------------------------------------
# 1. 데이터 로드 및 train.py와 동일한 피쳐 엔지니어링 파이프라인
# -------------------------------------------------------------------------
def load_jsonl(path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no} JSON 파싱 실패: {e}")
            samples.append(obj)
    return samples

def build_features(samples):
    """train.py의 피쳐 추출 로직과 100% 일치해야 함"""
    ids = []
    combined_texts = []
    meta_features_list = []
    
    for s in samples:
        ids.append(s.get("id", ""))
        
        current_prompt = s.get("current_prompt", "")
        if not isinstance(current_prompt, str):
            current_prompt = "" if current_prompt is None else str(current_prompt)
            
        meta = s.get("session_meta", {})
        history = s.get("history", [])
        
        # [피쳐 1] 히스토리 직전 행동 추출
        last_action = "none"
        if history:
            for h in reversed(history):
                if h.get("role") == "assistant" and "name" in h:
                    last_action = h.get("name", "none")
                    break
                    
        combined_text = f"[LAST_ACT: {last_action}] {current_prompt}"
        combined_texts.append(combined_text)
        
        # [피쳐 2] 세션 메타 데이터 추출
        user_tier = meta.get("user_tier", "free")
        tier_map = {'enterprise': 0, 'pro': 1, 'free': 2}
        tier_idx = tier_map.get(user_tier, 2)
        
        budget = meta.get("budget_tokens_remaining", 0)
        turn_index = meta.get("turn_index", 0)
        elapsed = meta.get("elapsed_session_sec", 0)
        
        workspace = meta.get("workspace", {})
        loc = workspace.get("loc", 0)
        git_dirty = 1 if workspace.get("git_dirty") is True else 0
        
        meta_features_list.append([tier_idx, budget, turn_index, elapsed, loc, git_dirty])
        
    return ids, combined_texts, np.array(meta_features_list)

# -------------------------------------------------------------------------
# 2. 제출 정렬 매핑 유틸
# -------------------------------------------------------------------------
def load_sample_submission(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    return fieldnames, rows

def merge_predictions(sub_rows, ids, preds):
    pred_map = dict(zip(ids, preds))
    for row in sub_rows:
        p = pred_map.get(row["id"])
        if p is not None:
            row["action"] = p
    return sub_rows

def save_submission(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# -------------------------------------------------------------------------
# 3. 추론 메인 루프
# -------------------------------------------------------------------------
def main():
    print("[*] 2026 AI Coding Agent Action Predictor Inference Engine Running...")
    
    # 모델 및 벡터라이저 컴포넌트 각각 로드
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH)):
        print(f"[-] Error: Model components not found at {MODEL_DIR}")
        sys.exit(1)
        
    print("Load model components...")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)

    # 테스트 데이터 로드
    print("Load test data...")
    if not os.path.exists(TEST_PATH):
        print(f"[-] Error: Test data not found at {TEST_PATH}")
        sys.exit(1)
    samples = load_jsonl(TEST_PATH)

    # 피쳐 빌드
    print("Build features...")
    ids, texts, meta_feats = build_features(samples)

    # 예측 수행 (텍스트 변환 + 메타 결합)
    print("Inference model...")
    if texts:
        X_text = vectorizer.transform(texts).toarray()
        X_total = np.hstack([X_text, meta_feats])
        preds = model.predict(X_total)
        preds = [str(p) for p in preds]
    else:
        preds = []

    # 결과 병합 및 저장
    print("Build submission...")
    fieldnames, sub_rows = load_sample_submission(SAMPLE_SUB_PATH)
    sub_rows = merge_predictions(sub_rows, ids, preds)
    save_submission(OUT_PATH, fieldnames, sub_rows)
    print(f"[+] Process Successfully Completed! Saved: {OUT_PATH}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"[-] Critical Error: {e}")
        traceback.print_exc()
        sys.exit(1)
