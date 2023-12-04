import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 데이터 불러오기
# 여기서는 데이터가 CSV 파일 형태로 있다고 가정합니다.
data = pd.read_csv(r"C:\Users\82105\OneDrive\문서\GitHub\Credit-card-fraud-Prediction\data\card_transdata.csv")

# 데이터 전처리
# 여기서는 간단히 숫자형 특성만 사용하고 정규화를 수행합니다.
X = data.drop('fraud', axis=1)
y = data['fraud']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# 학습 데이터와 테스트 데이터로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 로지스틱 회귀 모델 생성 및 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 모델 평가
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Precision: {precision_score(y_test, y_pred)}')
print(f'Recall: {recall_score(y_test, y_pred)}')
print(f'F1 Score: {f1_score(y_test, y_pred)}')

# 혼동 행렬 출력
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')
