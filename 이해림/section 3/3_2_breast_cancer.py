import pandas as pd
import seaborn as sns
import sklearn datasets import breast_cancer

breast_cancer = load_breast_cancer()
breast_cancer_df = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
breast_cancer_df['label'] = breast_cancer.target
breast_cancer_df.head()

#1. 데이터 탐색 및 준비
breast_cancer_df.shape

#데이터프레임의 요약 정보
breast_cancer_df.info()

#데이터프레임의 통계정보
breast_cancer_df.describe()

breast_cancer_df['label'].unique()
# 분석할 타겟은 몇가지 종류로 구성되어 있나

#결측치 확인
breast_cancer_df.isnull().sum()
# 결측치 확인 후 해당 데이터 삭제

#독립변수와 종속변수 분할
X = breast_cancer_df.iloc[:,:30]
y = breast_cancer_df['label']

from sklearn.model_selection import train_test_split
#학습용 데이터와 테스트용 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y)

print('학습용 데이터 {}개, 테스트용 데이터 {}개'.format(len(X_train), len(X_test)))

#2. 학습
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

#KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
neighbor_model = KNeighborsClassifier(n_neighbors=5)
neighbor_model.fit(X_train, y_train)

#SVC
from sklearn.svm import SVC
svm_model = SVC()
svm_model.fit(X_train, y_train)

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
forest_model = RandomForestClassifier(n_estimators=300, random_state=42)
forest_model.fit(X_train, y_train)

#LogisticRegression
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train, y_train)

#GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
gbm_model = GradientBoostingClassifier(random_state=42)
gbm_model.fit(X_train, y_train)

#XGBClassifier
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt

xgb_model = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=3, random_state=42)
xgb_model.fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(10, 12))
plot_importance(xgb_model, ax=ax)

#LGBMClassifier
from lightgbm import LGBMClassifier, plot_importance
from lightgbm import plot_importance

lgb_model = LGBMClassifier(n_estimators=300, random_state=42)
lgb_model.fit(X_train, y_train)

#plot_importance()를 이용하여 feature 중요도 시각화
fig, ax = plt.subplots(figsize=(10, 12))
plot_importance(lgb_model, ax=ax)

#평가 지표
# 각 모델의 혼동행렬, 정확도, 정밀도, 재현율, F1 score, AUC 구하기
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score

def get_clf_eval(y_test, pred=None):
  confusion = confusion_matrix(y_test, pred)
  accuracy = accuracy_score(y_test, pred)
  precision = precision_score(y_test, pred)
  recall = recall_score(y_test, pred)
  f1 = f1_score(y_test, pred) # 데이터마다 어떤 평가 지표가 중요한지는 다 다르다
  print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1: {3:.4f}'.format(accuracy, precision, recall, f1))
  print('혼동행렬')
  print(confusion)

model_list = [dt_model, neighbor_model, svm_model, forest_model, logistic_model, gbm_model, xgb_model, lgb_model]

for model in model_list: # 모델리스트의 값을 하나씩 꺼내어 담음
  pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, pred)
  model_name = model.__class__.__name__
  print('\n{} 성능지표:'.format(model_name))
  get_clf_eval(y_test, pred) #실제값과 예측값을 인자로 설정하여 각 주요 성능 지표가 어떠한 값을 나타내는 지 판단

# 성능개선 #early_stopping기술 구현
from xgboost import XGBClassifier

xgb_model2 = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=3, random_state = 42)
evals = [(X_test, y_test)] #실상은 한번더 분할하는게 맞지만 여기선 편하게
xgb_model2.fit(X_train, y_train, early_stopping_rounds=100, eval_set=evals, verbose=True) #백번을 기다려도 게속 학습이 진행된다면 중단하겠다
xgb_pred = xgb_model2.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print('XGBoost EarlyStopping 정확도: {0:.4f}'.format(xgb_accuracy))

#early_stopping기술 구현
from lightgbm import LGBMClassifier
# 24.04.11: LGBM v4.0.0 이상버전 param 변경에 따른 callback 추가
from lightgbm import early_stopping

lgb_model2 = LGBMClassifier(n_estimators=300, random_state = 42)
evals = [(X_test, y_test)]
# 24.04.11: LGBM v4.0.0 이상버전 param 변경에 따른 callback 추가
earlystop_callback = early_stopping(stopping_rounds=3)

# 24.04.11: LGBM v4.0.0 이상버전 param 변경
lgb_model2.fit(X_train, y_train, callbacks=[earlystop_callback], eval_set=evals)
# lgb_model2.fit(X_train, y_train, stopping_rounds=100, eval_set=evals, verbose=True)
lgb_pred = lgb_model2.predict(X_test)
lgb_accuracy = accuracy_score(y_test, lgb_pred)
print('LGBMClassifier EarlyStopping 정확도: {0:.4f}'.format(lgb_accuracy))

#VotingClassifier(앙상블 모델) #생성, 학습, 평가
from sklearn.ensemble import VotingClassifier
voting_model = VotingClassifier(estimators=[('LR', logistic_model),\
                                            ('KNN', neighbor_model)],
                                 voting='soft') #어떤것을 앙상블에서 사용할 지 설정
voting_model.fit(X_train, y_train)
pred = voting_model.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print('VotingClassifier 정확도: {0:.4f}'.format(accuracy))

from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators':[100, 300, 500],
    'max_depth' : [6, 8, 10, 12],
    'min_samples_leaf' : [8, 12, 18],
    'min_samples_split' : [8, 16, 20]
}
#RandomForestClassifier 객체 생성 후 GridSearchCV 수행
rf_clf = RandomForestClassifier(random_state=0, n_jobs=-1)
grid_cv = GridSearchCV(rf_clf, param_grid=params, cv=2, n_jobs=-1)
grid_cv.fit(X_train , y_train)

print('최적 하이퍼 파라미터:\n:', grid_cv.best_params_)
print('최고 예측 정확도: {0:.4f}'.format(grid_cv.best_score_))

from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators':[100, 500],
    'learning_rate': [0.05, 0.1]
}

gbm_model = GradientBoostingClassifier(random_state=10)

grid_cv = GridSearchCV(gbm_model, param_grid=params, cv=2, verbose=1)
grid_cv.fit(X_train, y_train)
print('최적 하이퍼 파라미터:\n', grid_cv.best_params_)
print('최고 예측 정확도: {0:.4f}'.format(grid_cv.best_score_))