# 결정트리의 앙상블
# 앙상블은 여러 머신러닝 모델을 연결하여 더 강력한 모델을 만드는 기법
# 랜덤 포레스트와 그래디언트 부스팅 결정 트리는 둘 다 모델을 구성하는 기본 요소로 결정트리를 사용한다
# 랜덤 포레스트

# 결정트리의 주요 단점은 훈련 데이터에 과대적합이 되는 것이다.
# 그러나 랜덤 포레스트는 이 문제를 회피할 수 있다.
# 조금씩 다른 여러 결정 트리의 묶음이다.
# 서로 다른 방향으로 과대 적합된 트리를 많이 만들어서 결과를 평균내서 과대적합을 막는다
# 트리 생성시 무작위성을 주입한다
# 데이터 포인트를 무작위로 선택하거나 분할 테스트에서 특성을 무작위로 선택하는 방법이다

# 랜덤 포레스트 구축
# 데이터의 부트스트랩 샘플을 생성한다
# n_samples 개의 데이터 포인트 중에서 무작위로 데이터 n_sample만큼 반복 추출한다
# 원래 데이터셋 크기와 같지만 어떤 데이터는 누락되거나 중복 될 수 있다.
# 각 노드에서 후보 특성을 무작위로 선택한 후 이 후보들 중에 최선의 테스틑를 찾는다
# 몇 개의 특성을 고를지는 max_features 매개변수로 조정할 수 있다.
# 부트스트랩 샘플링은 랜덤 포레스트의 트리가 조금씩 다른 데이터셋을 이요해 만들어지도록 한다
# 각 노드에서 특서으이 일부만 이용하기 떄문에 트리의 각 분기는 각기 다른 특성 부분 집합을 사용한다
# 핵심 매개변수는 max_features이다
# max_features를 n_features로 설정하면 모든 특성이 들어가기 때문에 무작위성이 들어가지 않는다
# max_features = 1로 설정하면 트리의 분기는 테스트할 특성을 고를 필요가 없게 되어 그냥 무작위로 선택한 특성의 임계값만 찾는다
# max_features 값을 크게 하면 랜덤 포레스트 트리들은 매우 비슷해지고 가장 두드러진 특성을 이용해 데이터에 잘 맞춰진다
# max_features 값을 낮추면 트리들은 많이 달라지고 각 트리는 데이터에 맞춰 깊이가 깊어진다.
#%%
%matplotlib inline
#%%
from IPython.display import display
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

import mglearn
import os

import graphviz
os.environ["PATH"] += os.pathsep + 'C:/Users/ddtthh/Desktop/namhyo/machine_learning/Lib/site-packages/graphviz/bin'
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_moons
from sklearn.datasets import load_breast_cancer #유방암 데이터

def plot_feature_importances_cancer(model):
    #결정 트리의 특성 중요도를 차트로 보여줌
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("특성 중요도")
    plt.ylabel("특성")
    plt.ylim(-1, n_features)


X,y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,random_state=42)

forest = RandomForestClassifier(n_estimators=5, random_state=2)

forest.fit(X_train,y_train)

#시각화
#fig, axes = plt.subplots(2,3,figsize=(20,10))
#for i, (ax, tree) in enumerate(zip(axes.ravel(),forest.estimators_)):
    #ax.set_title("tree {}".format(i))
    #mglearn.plots.plot_tree_partition(X,y,tree,ax=ax)

#mglearn.plots.plot_2d_separator(forest,X,fill=True,ax=axes[-1,-1],alpha=.4)
#axes[-1,-1].set_title("random forest")
#mglearn.discrete_scatter(X[:,0],X[:,1],y)

# 유방암 데이터로 만들기
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train,y_train)

print("훈련 세트 정확도: {:.3f}".format(forest.score(X_train,y_train)))
print("테스트 세트 정확도: {:.3f}".format(forest.score(X_test,y_test)))

# 물론 단일 결정 트리에서 했던 것처럼 사전 가지치기를 할수 있다.(max_features)
# 결정 트리처럼 특성 중요도를 제공하는데 각 트리의 특성 중요도를 취합하여 계산한다
# 일반적으로 랜덤 포레스트에서 제공하는 특성 중요도가 하나의 트리에서 제공하는 것보다 더 신뢰할 만하다

plot_feature_importances_cancer(forest)

# 전 파트랑 비교해 봤을때 훨씬 많은 특성이 0이상의 중요도를 가진다
# 랜덤 포레스트는 알고리즘이 가능성 있는 많은 경우를 고려해서 단일 트리보다 더 넓은 시각으로 데이터를 바라볼 수 있다

#회귀와 분류에 있어 랜덤 포레스트는 가장 널리 쓰이는 머신러닝 알고리즘이다.
# 속도가 빨라지기 위해서는 cpu 코어를 여러개 돌려야하는데 이 때 쓰는 방법이 n_jobs = -1로 두는 것이다
# random_state에 영향을 크게 받으며 트리가 많을 수록 random_state 값의 변화에 따른 변동이 적다
# 만약 같은 결과를 내고 싶으면 random_state 값을 고정해야한다.

# 그러나 텍스트 데이터 같이 차원이 높고 희소한 데이터에는 작동하지 않아서 선형 모델을 써야한다
# 선형 모델보다 예측과 훈련이 느리다
# 중요 매개변수는 n_estimators, max_features이고 max_depth 같은 사전 가지치기 옵션이 있다
# n_estimators는 값이 클 수록 좋다
# max_features는 각 트리가 얼마나 무작위가 될지를 결정하며 작은 max_features는 과대적합을 줄여준다. 일반적으로 기본값을 쓴다
# 분류는 max_Features = sqrt(n_features)
# 회귀는 max_features = n_features
# 가끔 max_features나 max_leaf_nodes 매개변수를 추가하면 성능이 향상된다


#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

# 그래디언트 부스팅 회귀 트리

# 여러개의 결정 트리를 묶어 강력한 모델을 만드는 앙상블 방법 중 하나이다.
# 이름이 희귀지만 분류나 회귀 둘 다 사용 가능하다
# 무작위성이 없고 강한 사전 치기를 사용한다 => 메모리도 적게 쓱도 예측도 빠르다
# 간단한 모델(약한 학습기)를 많이 연결한다
# 트리가 많이 추가될수록 성능이 좋아진다
# 이전 트리의 오차를 얼마나 강하게 보정하는것을 제어하는 매개변수인 learning_rate는 가장 중요한 매개변수이다
# 학습률이 크면 트리는 보정을 강하게 하기 떄문에 복잡한 모델을 만든다
# n_estimators 값을 키우면 앙상블에 트리가 더 많이 추가되어 모델의 복잡도가 커지고 훈련 세트에서의 실수를 바로 잡을 수 있다

X_train,X_test,y_train,y_test = train_test_split(cancer.data, cancer.target, random_state = 0)
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train,y_train)
print("훈련 세트: {:.3f}".format(gbrt.score(X_train,y_train)))
print("테스트 세트: {:.3f}".format(gbrt.score(X_test,y_test)))

# 훈련 세트가 100퍼인거보면 과대적합이 되었다. 
# 이를 극복하기 위해선 최대 깊이를 줄여 사전 가지치기를 강하게 하거나 학습률을 낮출 수 있다

gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train,y_train)
print("훈련 세트: {:.3f}".format(gbrt.score(X_train,y_train)))
print("테스트 세트: {:.3f}".format(gbrt.score(X_test,y_test)))

# ================================================

gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train,y_train)
print("훈련 세트: {:.3f}".format(gbrt.score(X_train,y_train)))
print("테스트 세트: {:.3f}".format(gbrt.score(X_test,y_test)))

gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train,y_train)
plot_feature_importances_cancer(gbrt)

# 랜덤 포레스트와 비슷한 특성을 강조하지만 그래디언트 부스팅은 일부 특성을 완전히 무시한다
# 비슷한 종류의 데이터에서는 더 안정적인 랜덤 포레스트를 먼저 적용한다
# 그래디언트 부스팅을 적용할려면 xgboost 패키지와 파이썬 인터페이스를 검토해 보는 것이 좋다

# 가장 큰 단점은 매개변수를 잘 조정해야한다는 것과 훈련 시간이 길다는 것이다
# 특성의 스케일을 조정하지 않아도 되고 이진 특성이나 연속적인 특성에서도 잘 동작한다
# 하지만 텍스트처럼 희소한 고차원 데이터에서는 작동하지 않는다
# 중요 매개변수는 n_estimators와(트리 개수 지정)정 learning_rate(오차를 보정하는 정도를 조절절
# learning_rate를 낮추면 비슷한 복잡도의 모델을 만들기 위해서 더 많은 트리를 추가해야한다
# 래덤 포레스트랑 달리 n_estimators크게 하면 모델이 복잡해지고 과대적합이 된다. 따라서 두 매개변수를 적절히 조절해야한다
# 또한 max_depth(max_leaf_nodes)로 복잡도를 낮출 수 있다
# 보통 max_depth는 최대 5정도로 맞춰서 작업한다




#%%
