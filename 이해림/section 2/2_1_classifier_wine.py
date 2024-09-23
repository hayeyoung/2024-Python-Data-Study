from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# 데이터 불러오기
data = load_wine()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['label'] = data.target

# 데이터분할
# x는 레이블 컬럼 드롭한 나머지, y는 레이블 컬럼
X = df.drop(['label'], axis = 1) 
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

# 학습
model = DecisionTreeClassifier(random_state=42, max_depth = 2)  # 개체 생성
model.fit(X_train, y_train) # 연산 후 모델 생성

# 평가
score = model.score(X_test, y_test) # 테스트 set을 넣어 평가 가능

# 예측
y_pred = model.predict([X_test.iloc[10]]) #predict