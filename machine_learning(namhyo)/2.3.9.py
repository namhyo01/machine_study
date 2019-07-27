# 신경망(딥러닝)
# 딥러닝이 많은 머신러닝 애플리케이션에서 매우 희망적인 성과를 보여주지만, 특정 분야에 정교하게 적용되어 있을때가 많다
# 다층 퍼셉트론(MLP)을 다룬다(피드 포워드 신경망, 신경망이라고도 한다.)

# 신경망 모델 : 여러 단계를 거쳐 결정을 만들어내는 선형 모델의 일반화된 모습이라고 볼 수 있다.

# MLP에서는 가중치 합을 만드는 과정이 여러 번 반복되며, 먼저 중간 단꼐를 구성하는 은닉 유닛을 계산하고 최종 결과를 산출하기 위해 다시 가중치 합을 계산한다
# 선형 모델보다 강력하게 만들려면 각 은닉 유닛의 가중치 합을 계산한 후 그 결과에 비선형 함수인 렐루나 하이퍼블릭 탄젠트를 적용한다
# 이 함수 결과의 가중치 합을 계산해서 출력을 만든다
# 렐루 함수는 0 이하를 잘라버리고 tahn 함수는 낮은 입력값에 대해서는 -1로 수렴하고 큰 입력값에 대해서는 1로 수렴한다

# tahn을 이용할때 쓰이는 공식의 변수들은 
# w: 입력 x와 은닉충 h 사이의 가중치
# v: 은닉충 h와 출력 y의 가중치이다.
# v와 w는 훈련데이터에서 학습하고, x는 입력 특성이며, y는 계산된 출력, h는 중간 계산 값이다
# 중요한 매개변수는 은닉충의 유닛 개수이다.
# 또한 은닉충을 한개만이 아닌 더 추가할 수 있다.

# 이와 같은 많은 신경망이 생기면 이것을 딥러닝이라고한다

# 신경망 튜닝
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
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X,y = make_moons(n_samples=100, noise=0.25,random_state=3)
X_train, X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=42)

#mlp = MLPClassifier(solver="lbfgs", random_state=0).fit(X_train,y_train)
#mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
#mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
#plt.ylabel("feature 1")
#plt.xlabel("feature 0")

# 이렇듯 매우 비션형적이지만 비교적 매끄러운 결정 경계를 만들었다
# MLP는 기본적으로 은닉 유닛 100개를 사용하는데, 개수를 줄이면 복잡도가 낮아진다

#mlp = MLPClassifier(solver="lbfgs", random_state=0,hidden_layer_sizes=[10]).fit(X_train,y_train)
#mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
#mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
#plt.xlabel("feature 0")
#plt.ylabel("feature 1")


# 매끄러운 결정 경계를 원한다면 은닉 유닛을 추가하거나, 은닉충을 추가하거나, tanh함수를 사용할수 있다

# 10개의 유닛으로 된 두개의 은닉충
#mlp = MLPClassifier(solver="lbfgs", random_state=0,hidden_layer_sizes=[10,10]).fit(X_train,y_train)
#mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
#mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
#plt.xlabel("feature 0")
#plt.ylabel("feature 1")

# tanh 활성화 함수가 적용된 10개의 유닛으로 만들어진 두개의 은닉층

mlp = MLPClassifier(solver="lbfgs",activation='tanh', random_state=0,hidden_layer_sizes=[10,10]).fit(X_train,y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
plt.xlabel("feature 0")
plt.ylabel("feature 1")

# MLPClassifier에서 이런 역할을 하는 매개변수는 alpha(선형 회귀 모델)이랑 똑같다.
# 기본값은 매우 낮게(거의 규제하지 않음)
# 따라서 알파 값이 작으면 작을수록 더 복잡하고(규제가 작아지니), 은닉층의 수가 많으면 많을 수록 더 날카로워 진다니

# 학습을 시작하기 전에 가중치를 무작위로 설정하며 이 무작위한 초기화가 모델의 학습에 영향을 준다(random_state값값
# 초깃값이 다르면 모델이 많이 달라질 수 있다.
# 신경망이 크고 복잡도가 적절할수록 이런 점이 정확도에 미치는 영향이 크지 않지만 기억은 해야한다.

# 일반적인 MLP의 정확도는 높지만 다른 모델만큼은 아니다
# SVC예제에서는 데이터 스킬이 영향을 미치는데 신경망도 모든 입력 평균은 0, 분산은 1이되도록 변형하는것이 좋다

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,random_state = 0)
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train,y_train)
# 훈련 세트 각 특성의 평균을 구한다
mean_on_train = X_train.mean(axis=0)
# 훈련 세트 각 특성의 표준 편차를 계산한다
std_on_train = X_train.std(axis=0)

#  데이터에서 평균을 빼고 표준편차로 나누면
# 평균 0, 표준 편차 1인 데이터로 변형된다
X_train_scaled = (X_train-mean_on_train) / std_on_train
# (훈련 데이터의 평균과 표준 편차를 이용해해 같은 변환을 테스트 세트에도 합니다
X_test_scaled = (X_test-mean_on_train)/std_on_train

mlp = MLPClassifier(random_state=0).fit(X_train_scaled,y_train)

print("훈련 세트 정확도: {:.3f}".format(mlp.score(X_train_scaled,y_train)))
print("테스트 세트 정확도: {:.3f}".format(mlp.score(X_test_scaled,y_test)))

# 여기까지 하면 최대 반복횟수에 도달했다고 경고가 뜬다 따라서  max_iter을 증가시켜줘야한다


mlp = MLPClassifier(random_state=0,max_iter=1000).fit(X_train_scaled,y_train)

print("훈련 세트 정확도: {:.3f}".format(mlp.score(X_train_scaled,y_train)))
print("테스트 세트 정확도: {:.3f}".format(mlp.score(X_test_scaled,y_test)))

# 일반화를 더 올리기 위해ㅔ alpha 매개변수를 1로 올리면 된다

mlp = MLPClassifier(random_state=0,max_iter=1000,alpha=1).fit(X_train_scaled,y_train)

print("훈련 세트 정확도: {:.3f}".format(mlp.score(X_train_scaled,y_train)))
print("테스트 세트 정확도: {:.3f}".format(mlp.score(X_test_scaled,y_test)))


# 신경망은 학습이 오래 걸리고 데이터 전처리에 조심해야한다.
# 다른 종류의 특성을 가진 데이터라면 트리 기반 모델이 더 잘 작동할 수 있다.
# 신경망 튜닝은 더 열심히 준비해야한다

# 신경망에서 가장 중요한 매개변수는 은닉충의 개수랑 각 은닉충의 유닛수이다
# 각 은닉충의 유닛수는 보통 입력 특성의 수와 비슷하게 설정하지만 수천 초중반을 넘는 일은 거의 없다
# 복잡도의 측정치는 학습된 가중치 또는 계수의 수이다
# ex) 특성 100개랑 은닉 유닛 100 개를 가진 이진 분류라면 입력층과 첫번째 은닉충 사이에는 편향을 포함해 100*100+100 = 100100 가중치가있다.
# 그리고 은닉충과 출력층 사이에 100*1+1의 가중치가 더 있어서 총 가중치는 10201이다.
# 만약 은닉 유닛이 100개인 유닛층을 추가하면 첫번쨰 은닉충에서 두번째 은닉충으로 100*10+100 = 100100이 추가되서 20301개의 가중치가 된다
# 신경망 매개변수를 조정하는 일반적인 방법은 먼저 충분히 과대적합되어서 문제를 해결할만한 큰 모델을 만든다
# 그다음 신경망 구조를 줄이거나 규제를 강화시켜 일반화를 증가시킨다
# solver 매개변수를 사용해 모델을 학습시키는 방법 또는 매개변수에 학습에 사용할 알고리즘을 지정할 수 있다.
# solver 매개변수에는 쉽게 사용할 수 있는 옵션이 두가지가 있다. 기본값은 adam으로 대부분 잘 작동하지만 데이터 스케일의 조정이 필요하다(위에서함함
# 다른 하나는 lbfgs로 안정적이지만 규모가 큰 모델이나 대량의 데이터 셋에서는 속도가 느리다
# 그리고 마지막으로 고급 옵션인 sgd가 있는데 이것은 여러 매개변수랑 튜닝하여 최선의 결과를 만들수 있다.

#%%
