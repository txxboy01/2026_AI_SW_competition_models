import os
import pandas as pd

print("[*] Evaluation Server Started.")

# 주최 측 평가 서버가 자동으로 매핑해주는 경로 확인용 방어 코드
if os.path.exists("data/test.csv"):
    test_df = pd.read_csv("data/test.csv")
    print(f"[+] Successfully loaded test data. Shape: {test_df.shape}")

    # 임시 submission 파일 생성 시뮬레이션
    os.makedirs("output", exist_ok=True)
    submission = pd.DataFrame({"prediction": [0] * len(test_df)})
    submission.to_csv("output/submission.csv", index=False)
    print("[+] Submission.csv successfully generated.")
else:
    print("[-] Data directory not found. Running in local test mode.")
