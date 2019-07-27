# 배깅, 엑스트라 트리, 에이다 부스트
# 또 다른 앙상블인 배깅, 엑스트라 트리, 에이다 부스트에 대해 나온다

#%%
%matplotlib inline
#%%
from IPython.display import display
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

import mglearn
#from preamble import *
import os
import graphviz
os.environ["PATH"] += os.pathsep + 'C:/Users/ddtthh/Desktop/namhyo/machine_learning/Lib/site-packages/graphviz/bin'

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier

Xm, ym = make_moons(n_samples=100, noise=0.25, random_state=3)
Xm_train, Xm_test, ym_train, ym_test = train_test_split(Xm,ym, stratify=ym, random_state=42)

cancer = load_breast_cancer()
Xc_train, Xc_test, yc_train, yc_test = train_test_split(cancer.data, cancer.target, random_state=0)

# 배깅(Bootstrap aggregating)
# 배깅은 중복을 허용한 랜덤 샘플링으로 만든 훈련 세트를 사용하여 각각 다르게 학습한다
# 랜덤 포레스트의 특징과 거의 같다
# 분류기가 preict_proba() 메서드를 지원 하는 경우 확률 값을 평균하여 예측을 수행
# 지원 안하는 경우 가장 빈도가 높은 클래스 레이블이 예측 결과가 된다

bagging = BaggingClassifier(LogisticRegression(),n_estimators=100, oob_score=True, n_jobs=-1, random_state=42)
bagging.fit(Xc_train, yc_train)

#oob_score = True로 지정하면 매개변수는 부트스트래핑에 포함되지 않은 샘플을 기반으로 훈련된 모델을 평가한다
# oob_score 값을 통해 테스트 세트의 성능을 짐작할수 있다.(RandomForestClassifier도 지원원
# 왜냐하면 oob_score는 안쓴 데이터들을 기준으로 평가하기 때문이다
print("훈련 세트 정확도: {:.3f}".format(bagging.score(Xc_train, yc_train)))
print("테스트 세트 정확도: {:.3f}".format(bagging.score(Xc_test, yc_test)))
print("OOB 세트 정확도: {:.3f}".format(bagging.oob_score_))

bagging = BaggingClassifier(DecisionTreeClassifier(),n_estimators=5, n_jobs=-1, random_state=42)
bagging.fit(Xm_train, ym_train)


fig, axes = plt.subplots(2,3,figsize=(20,10))
#for i, (ax,tree) in enumerate(zip(axes.ravel(), bagging.estimators_)):
#    ax.set_title("tree {}".format(i))
#    mglearn.plots.plot_tree_partition(Xm,ym,tree,ax=ax)

#mglearn.plots.plot_2d_separator(bagging, Xm, fill=True, ax=axes[-1,-1], alpha=.4)
#axes[-1,-1].set_title("bagging")
#mglearn.discrete_scatter(Xm[:,0], Xm[:,1],ym)
#plt.show()

# 배깅은 랜덤 포레스트와 달리 max_samples 매개변수에서 부트스트랩 샘플의 크기를 지정 할 수 있다.

#------------------------------------------

# 엑스트라 트리

# 랜덤 포레스트와 비슷하지만 후보 특성을 무작위로 분할한 다음 최적의 분할을 찾는다.
# 랜덤 포레스트와 달리 DecisionTreeClassifier(splitter='random)
# 저게 splitter = random일 경우 무작위로 분할한 후보 노드 중에서 최선의 분할을 찾는다
# 무작위성을 증가시키면 일반적으로 모델의 편향이 늘어나지만 분산이 감소한다

xtree=ExtraTreesClassifier(n_estimators=5, n_jobs=-1, random_state=0) 
xtree.fit(Xm_train, ym_train)

fig, axes = plt.subplots(2,3,figsize=(20,10))
#for i, (ax,tree) in enumerate(zip(axes.ravel(), xtree.estimators_)):
#    ax.set_title("tree {}".format(i))
#    mglearn.plots.plot_tree_partition(Xm,ym,tree,ax=ax)

#mglearn.plots.plot_2d_separator(xtree, Xm, fill=True, ax=axes[-1,-1],alpha=.4)
#axes[-1,-1].set_title("extra tree")
#mglearn.discrete_scatter(Xm[:,0], Xm[:,1],ym)
#plt.show()

# 이렇듯 엑스트라 트리는 랜덤 포레스트랑 거의 비슷하다
# 따라서 일반적으로 랜덤 포레스트가 일반화가 더 쉬워서 잘 사용된다

#-----------------------------------------------------------------

# 에이다 부스트(Adapting Boosting)
# 그래디언트 부스팅 처럼 약한 학습기를 사용한다
# 차이점은 이전의 모델이 잘못 분류한 샘플에 가중치를 높여서 다음 모델을 학습시킨다.
# 기본값으로 분류는 DecisionTreeClassifier(max_depth=1)를 사용하고, 회귀는 DecisionTreeClassifier(max_depth=3)을 ㄱ사용한다
# base_estimator 매개변수에서 다른 모델을 지정 가능하다
# 그래디언트 부스팅과 마찬가지로 순차적으로 학습해야해서 n_jobs 매개변수를 지원하지 않는다

ada = AdaBoostClassifier(n_estimators=5,random_state=42)
ada.fit(Xm_train,ym_train)

fig, axes = plt.subplots(2,3,figsize=(20,10))
for i, (ax,tree) in enumerate(zip(axes.ravel(),ada.estimators_)):
    ax.set_title("tree {}".format(i))
    mglearn.plots.plot_tree_partition(Xm,ym,tree,ax=ax)

mglearn.plots.plot_2d_separator(ada, Xm, fill=True, ax=axes[-1,-1],alpha=.4)
axes[-1,-1].set_title("ada boost")
mglearn.discrete_scatter(Xm[:,0],Xm[:,1],ym)
plt.show()

# 에이다 부스트 분류는 깊이가 1인 결정 트리를 사용하기 떄문에 각 트리의 결정 경계가 직선 하나이다

ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(Xc_train,yc_train)

print("훈련 세트 정확도: {:.3f}".format(ada.score(Xc_train,yc_train)))
print("테스트 세트 정확도: {:.3f}".format(ada.score(Xc_test,yc_test)))

# 아주 얕은 트리를 앙상블 했기 때문에 일반화 성능이 조금 더 향상되었다





#%%
