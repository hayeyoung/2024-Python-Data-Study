# 나눔폰트 설치(설치 후, 런타임 재시작 필요)
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf

import matplotlib.pyplot as plt

plt.rc('font', family='NanumBarunGothic') #한글폰트 설정
plt.rc('axes', unicode_minus=False) #마이너스 기호 표시

import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('exercise4.csv', index_col=None)
df.head()

df.describe() #데이터 탐색 필요

# 데이터 분포 확인 - 히스토그램
nrows = 5
ncols = 1

fig, axs = plt.subplots(nrows, ncols)
fig.set_size_inches(8, 16)

for i in range(len(df.columns)):
    sns.histplot(x=df.columns[i], data=df, kde=True, bins=30, ax=axs[i])

# 데이터 분포 확인 - 산점도
nrows = 4
ncols = 1

fig, axs = plt.subplots(nrows, ncols)
fig.set_size_inches(8, 16)

for i in range(len(df.columns) - 1):
    sns.scatterplot(x=df.columns[i], y='평균 주택 가격', data=df, ax=axs[i])

# 데이터 분포 확인 - 변수 간 상관관계
df.corr()

# 데이터 분포 확인 - 변수 간 상관관계(히트맵)
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), linewidths=1, annot=True)

np.abs(df.corr()['평균 주택 가격']).sort_values(ascending=False)

df.sort_values(by=['평균 주택 가격']).tail()

# 결측값 확인
df.isna().sum()

# 중복값 확인
df.duplicated().sum()

#데이터 전처리
y = df['평균 주택 가격']
X = df.drop(['평균 주택 가격'],axis=1,inplace=False)

from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

#학습
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print('weight:', lr_model.coef_)
print('bias:',lr_model.intercept_)

coef = pd.Series(data=np.round(lr_model.coef_, 1), index=X.columns )
coef

coef_sort = coef.sort_values(ascending=False)
sns.barplot(x=coef_sort.values, y=coef_sort.index)

#평가
from sklearn.metrics import mean_squared_error, r2_score

def printRegressorResult(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print('MSE : {0:.3f} , RMSE : {1:.3f}, r2 : {2:.3f}'
    .format(mse , rmse, r2))

y_pred = lr_model.predict(X_test)
printRegressorResult(y_test, y_pred)

#성능 개선 #독립변수 선택
from sklearn.feature_selection import SelectKBest, f_regression #SelectKBest

# k = 3
X_selected = SelectKBest(score_func = f_regression, k = 3) #도움이 되는 featuer 3개만 뽑아줘라
X_selected.fit_transform(X, y)
features = X.columns[X_selected.get_support()]    
print('features = {}'.format(features))

X_selected = df[features].copy()
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

lr_model2 = LinearRegression()
lr_model2.fit(X_train, y_train)

y_pred = lr_model2.predict(X_test)
printRegressorResult(y_test, y_pred)

#성능 개선 #다항회귀 모델
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X_selected)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)

poly_model = LinearRegression()
poly_model.fit(X_train, y_train)

y_pred = poly_model.predict(X_test)
printRegressorResult(y_test, y_pred)  

X_selected.columns

#성능 개선 #데이터 스케일 변환 
# 데이터 분포 확인 - 히스토그램
nrows = 1
ncols = 4

fig, axs = plt.subplots(nrows, ncols)
fig.set_size_inches(20, 4)

sns.histplot(x= X_selected.columns[0], data = X_selected, kde=True, bins=30, ax=axs[0])
sns.histplot(x= X_selected.columns[1], data = X_selected, kde=True, bins=30, ax=axs[1])
sns.histplot(x= X_selected.columns[2], data = X_selected, kde=True, bins=30, ax=axs[2])
sns.histplot(x= y, data = y, kde=True, bins=30, ax=axs[3])

print(X_selected.skew())
print('\n평균 주택 가격: {0:.2f}'.format(y.skew()))

X_selected['인구 밀집도'] = np.log1p(X_selected['인구 밀집도'])
y = np.log1p(y)

# 데이터 분포 확인 - 히스토그램
nrows = 1
ncols = 4

fig, axs = plt.subplots(nrows, ncols)
fig.set_size_inches(20, 4)

sns.histplot(x= X_selected.columns[0], data = X_selected, kde=True, bins=30, ax=axs[0])
sns.histplot(x= X_selected.columns[1], data = X_selected, kde=True, bins=30, ax=axs[1])
sns.histplot(x= X_selected.columns[2], data = X_selected, kde=True, bins=30, ax=axs[2])
sns.histplot(x= y, data = y, kde=True, bins=30, ax=axs[3])

print(X_selected.skew())
print('\n평균 주택 가격: {0:.2f}'.format(y.skew()))

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

lr_model3 = LinearRegression()
lr_model3.fit(X_train, y_train)

y_pred = lr_model3.predict(X_test) 
mse = mean_squared_error(np.expm1(y_test), np.expm1(y_pred))
rmse = np.sqrt(mse)
r2 = r2_score(np.expm1(y_test), np.expm1(y_pred))
print('MSE : {0:.3f} , RMSE : {1:.3f}, r2 : {2:.3f}'.format(mse , rmse, r2))

#성능 개선 #의사결정나무 기반 회귀 모델
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)

from lightgbm import LGBMRegressor

lgb_model = LGBMRegressor(n_estimators=100, random_state=42)
lgb_model.fit(X_train, y_train)

model_list = [lr_model, forest_model, lgb_model]

for model in model_list:
    model.fit(X_train , y_train)
    y_preds = model.predict(X_test)
    mse = mean_squared_error(y_test, y_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_preds)
    model_name = model.__class__.__name__
    print('{0} MSE : {1:.3f} , RMSE : {2:.3f}, r2 : {3:.3f}'.format(model_name, mse , rmse, r2)) 

from lightgbm import LGBMRegressor

lgb_model2 = LGBMRegressor(n_estimators=100)
evals = [(X_train , y_train), (X_test, y_test)]
lgb_model2.fit(X_train, y_train, early_stopping_rounds=10, eval_set=evals)
y_preds = lgb_model2.predict(X_test)
mse = mean_squared_error(y_test, y_preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_preds)
print('MSE : {0:.3f} , RMSE : {1:.3f}, r2 : {2:.3f}'.format(mse , rmse, r2))

import lightgbm as lgb
lgb.plot_metric(lgb_model2)

lgb.plot_importance(lgb_model2)

#교차 검증
from sklearn.model_selection import cross_validate

# cv: 3개의 train, test set fold 로 나누어 학습 
scores = cross_validate(lr_model, X, y, scoring="neg_mean_squared_error", cv=3, return_train_score=True, return_estimator=True)
print('Scores', scores)

mse = (-1 * scores['train_score'])
print('MSE:', mse)

rmse  = np.sqrt(-1 * scores['train_score'])
print('RMSE:', rmse)

print('RMSE 평균: {0:.3f} '.format(np.mean(rmse)))