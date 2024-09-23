#1. 교차검증

# KFold #전체 데이터셋을 k번 접었다는 뜻
# 세 덩어리중 첫블럭이 학습용 그 외 검증용
# 다음은 두번째 블럭이 학습용 그 외 검증용
# 마지막으로 마지막 블럭이 학습용 그 외 검증용
from sklearn.model_selection import KFold

kf = KFold(n_splits=3)
for train_index, test_index in kf.split(X): 
    print('-------------------------------------------')
    print("학습용:", train_index)
    print("\n학습용 레이블", y[train_index].unique())
    print("\n\n테스트용:", test_index)
    print("\n테스트용 레이블", y[test_index].unique())

    # StratifiedKFold # 중간에 데이터들이 범주에 따라 나누어 짐 (kFold는 단순히 데이터 분할)
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3)
for train_index, test_index in skf.split(X, y):
    print('-------------------------------------------')
    print("학습용:", train_index)
    print("\n학습용 레이블", y[train_index].unique())
    print("\n\n테스트용:", test_index)
    print("\n테스트용 레이블", y[test_index].unique())

    
from sklearn.model_selection import cross_validate

scores = cross_validate(tree_model, X_train, y_train, cv=3, return_estimator=True)
scores
# 모델 객체를 tree안에 담아 인자로 설정 -> 학습시킬 때, 3 덩어리로 나누어 교차검증하라

# 3개의 분류기 평가
for i in range(3):     
    score = scores['estimator'][i].score(X_test, y_test) #테스트
    print('{0}번째 의사결정나무 정확도: {1:.2f}'.format(i+1, score))

#2. 스케일조절

# 라이브러리 로딩
import pandas as pd
from sklearn.datasets import load_iris

# 데이터 불러오기
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

df.describe()

from sklearn.preprocessing import MinMaxScaler

minmax_scaler = MinMaxScaler() # 객체 설정-> 데이터fit-> 데이터transform
minmax_scaler.fit(X)
minmax_scaled_data = minmax_scaler.transform(X)
minmax_scaled_df = pd.DataFrame(data=minmax_scaled_data, columns=iris.feature_names)
minmax_scaled_df.describe()

from sklearn.preprocessing import StandardScaler # 표준편차와 평균에 초점을

standard_scaler = StandardScaler()
standard_scaler.fit(X)
standard_scaled_data = standard_scaler.transform(X)
standard_scaled_df = pd.DataFrame(data=standard_scaled_data, columns=iris.feature_names)
standard_scaled_df.describe()

#3. 차원축소 # 일부의 속성만 사용하거나 기존의 데이터를 줄여서 활용 #잘 설명할 수 있는 데이터만을 쓰자

# 라이브러리 로딩
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# 데이터 불러오기
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# PCA
pca = PCA(n_components= 4)
pca.fit(X)
print(pca.explained_variance_ratio_)

# scree plot
plt.plot(pca.explained_variance_ratio_, 'o--') #plot 사용 시각화

import seaborn as sns

sns.scatterplot(x= df['pca_1'], y= df['pca_2'], hue=df['species'], legend ='auto') # 굳이 4개까지는 아닌 2개로도 최적화 가능하다

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# 데이터분할
y = df['species']
X = df.iloc[:, 5:]  #pca_1, pca_2 컬럼만 사용

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

# 학습 - DecisionTreeClassifier 
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# 평가
score = tree_model.score(X_test, y_test)
print('의사결정나무 정확도: {0:.2f}'.format(score))