import os
import json
import pandas as pd
import numpy as np
import pickle

def load_model():
    with open(os.path.join('models', 'agent_router_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join('models', 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def parse_jsonl(file_path):
    parsed_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # 메타정보 1단계 복구 및 내재화
            meta = item['session_meta']
            
            parsed_data.append({
                'id': item['id'],
                'current_prompt': item['current_prompt'],
                'user_tier': meta['user_tier'],
                'budget_tokens': meta['budget_tokens_remaining'],
                'turn_index': meta['turn_index']
            })
    return pd.DataFrame(parsed_data)

def predict(model, vectorizer, data):
    # 텍스트 피처 벡터화 + 메타 피처 결합 로직 고도화
    texts = data['current_prompt'].fillna('')
    X_text = vectorizer.transform(texts).toarray()
    
    # 카테고리형 데이터 수치 매핑 (예시 가공)
    tier_map = {'enterprise': 0, 'pro': 1, 'free': 2}
    X_meta = data['user_tier'].map(tier_map).fillna(2).values.reshape(-1, 1)
    
    X_total = np.hstack([X_text, X_meta])
    return model.predict(X_total)

if __name__ == "__main__":
    print("[*] 2026 AI Coding Agent Action Predictor Engine Running...")
    try:
        model, vectorizer = load_model()
        # 규격에 따른 서버 자동 주입 test.jsonl 로드
        test_df = parse_jsonl(os.path.join('data', 'test.jsonl'))
        
        preds = predict(model, vectorizer, test_df)
        
        # 14개 클래스 이름 대소문자 매핑 검증 안전 출력
        submission = pd.DataFrame({'id': test_df['id'], 'action': preds})
        submission.to_csv(os.path.join('output', 'submission.csv'), index=False)
        print("[+] submission.csv generated successfully inside output/")
    except Exception as e:
        print(f"[-] Critical Error: {e}")
