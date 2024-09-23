# 패키지 로딩
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# 데이터 로딩
iris = load_iris()

# 로딩한 데이터를 데이터 프레임으로 생성
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label'] = iris.target

# 중복값 제거
iris_df.drop_duplicates(keep='first', inplace=True)

# 데이터분할
X = iris_df.iloc[:, :4]
y = iris_df['label']

# 학습용 데이터와 테스트용 데이터로 나눔
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

print("학습용 데이터 수:{0}, 테스트용 데이터 수: {1}".format(len(X_train), len(X_test)))

# DecisionTreeClassifier 객체 생성 
model = DecisionTreeClassifier(random_state=42)
# 학습용 데이터로 학습 수행
model.fit(X_train, y_train)


#테스트용 데이터로 모델 평가
score = model.score(X_test, y_test)
print("테스트셋의 정확도:{:.2f}".format(score))

# 신규 값 예측
new = np.array([[5.4, 4, 1.5, 0.2]])
y_pred = model.predict(new)
print("예측결과:{}".format(y_pred))