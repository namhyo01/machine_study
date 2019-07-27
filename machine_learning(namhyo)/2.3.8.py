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
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
cancer = load_breast_cancer()
# 커널 서포트 벨터 머신(SVM)

# SVM은 입력 데이터에서 단순한 초평면으로 정의 되지 않는 더 복잡한 모델을 만들 수 있도록 확장한 것이다

# 선형 모델과 비선형 특성
# 직선과 초평면은 유연하지 못해서 저차원 데이터셋에서는 선형 모델이 매우 제한적이다
# 선형모델을 유연하게 만드는 방법은 특성끼리 곱하거나 특성을 거듭제곱 하는 식으로 새로운 특성을 추가하는 것이다

X,y = make_blobs(centers=4, random_state=8)
y = y%2

#mglearn.discrete_scatter(X[:,0],X[:,1],y)
#plt.xlabel("feature 0")
#plt.ylabel("feature 1")
#linear_svm = LinearSVC().fit(X,y)
#mglearn.plots.plot_2d_separator(linear_svm, X)
#mglearn.discrete_scatter(X[:,0],X[:,1],y)
#plt.xlabel("feature 0")
#plt.ylabel("feature 1")

# (특성 0, 특성 1)의 2차원 데이터 포인터가 아니라 (특성 0, 특성 1, 특성 1**2)의 3차원 데이터 포인터로 표현된다
# 두번쨰 특성을 제곱하여 추가한다
#X_new = np.hstack([X,X[:,1:]**2])
from mpl_toolkits.mplot3d import Axes3D, axes3d
#linear_svm_3d = LinearSVC().fit(X_new,y)
#coef, intercept = linear_svm_3d.coef_.ravel(),  linear_svm_3d.intercept_
#선형 결정 경계 그리기
#figure = plt.figure()
#ax = Axes3D(figure, elev=-152, azim=-26)
mask =y == 0
#xx = np.linspace(X_new[:,0].min()-2, X_new[:,0].max()+2,50)
#yy = np.linspace(X_new[:,1].min()-2,X_new[:,1].max()+2,50)

XX, YY = np.meshgrid(xx,yy)
ZZ = (coef[0]*XX+coef[1]*YY+intercept)/-coef[2]
#ax.plot_surface(XX,YY,ZZ,rstride=8,cstride=8,alpha=0.3)
#ax.scatter(X_new[mask,0],X_new[mask,1],X_new[mask,2],c='b',cmap=mglearn.cm2,s=60,edgecolor='k')
#ax.scatter(X_new[~mask,0], X_new[~mask,1],X_new[~mask,2],c='r', marker = '^', cmap=mglearn.cm2, s=60, edgecolor='k')

#ax.set_xlabel("feature 0")
#ax.set_ylabel("feature 1")
#ax.set_zlabel("feature 1**2")

ZZ = YY **2
#dec = linear_svm_3d.decision_function(np.c_[XX.ravel(),YY.ravel(), ZZ.ravel()])
#plt.contour(XX,YY,dec.reshape(XX.shape), levels=[dec.min(),0,dec.max()], cmap=mglearn.cm2,alpha=0.5)
#mglearn.discrete_scatter(X[:,0],X[:,1],y)
#plt.xlabel("feature 0")
#plt.ylabel("feature 1")
#이제는 직선보다 타원에 가까운 모습을 보인다

# 커널 기법
# 데이터셋에 비선형 특성을 추가하여 선형 모델을 강력하게 만들었다
# 하지만  어떤 특성을 추가해야할지 모르고 특성을 많이 추가하면 연산 비용이 커진다
# 새로운 특성을 만들지 않고 고차원에서 분류기를 학습시키는 기법을 커널 기법이라고 한다
# 실제로 데이터들을 확장하지 않고 확장된 특성에 대한 데이터 포인터들의 거리를 계산한다(스칼라)
# 원래 특성의 가능한 조합을 지정된 차수까지 모두 계산(특성 1 **2 * 특성2 **5) 하는 다항식 커널
# 가우시안 커널이라고 불리는 RBF 커널이 있다.(무한한 특성 공간에 매핑, 모든 차수의 모든 다항식을 고려한다. 하지만 특성 중요도는 고차항이 되면 될수록 줄어든다)

# SVM 이해하기
# 학습이 진행되는 동안 svm은 각 훈련 데이터 포인터가 두 클래스 사이의 결정 경계를 구분하는데 얼마나 중요한지 알수 있다.
# 두 클래스 사이의 경계에 위치한 포인터들이다
# 이러한 데이터 포인터들을 서포트 벡터라고 한라
# 새로운 데이터 푕ㄴ트에 대해 예측하려면 각 서포트 벡터와의 거리를 측정한다(SVC객체의 dual_coef_ 속성에 저장된다다
# 데이터 포인트 사이의 거리는 가우시안 커널에 의해 계싼된다
# r은 가우시안 커널의 폭을 제어하는 매개변수이다

X,y = mglearn.tools.make_handcrafted_dataset()

svm = SVC(kernel='rbf', C=10,gamma=0.1).fit(X,y)
#mglearn.plots.plot_2d_separator(svm,X,eps=.5,edgecolor='white') # 선그어주기
#데이터 포인터 그리기
#mglearn.discrete_scatter(X[:,0],X[:,1],y)
#서포트 벡터
sv = svm.support_vectors_
# dual_coef_의 부호에 의해 서포트 벡터의 클래스 레이블이 결정된다
#sv_labels = svm.dual_coef_.ravel()>0
#mglearn.discrete_scatter(sv[:,0],sv[:,1],sv_labels,s=15,markeredgewidth=3)
#plt.xlabel("feature 0")
#plt.ylabel("feature 1")

# 여기서 svm은 매우 부드러운 비선형 경계를 만들었다
# gamma 매개변수는 r의 가우시안 커널 폭의 역수이다
# gamma 매개변수는 하나의 훈련 샘플이 미치는 영향의 범위를 결정한다
# 따라서 작은 감마값은 모델의 복잡도를 낮추고, 큰 감마 값은 더 복잡한 것을 만든다
# 작은 값은 넓은 영역을 듯하며 큰 값이라면 영향이 미치는 범위가 제한적이다
# 즉 가우시안 커널의 반경이 클 수록 훈련 샘플의 영향 범위도 커진다. 
# C 매개변수는 선형 모델에서 사용한 것 과 비슷한 규제 매개변수이다. 
# 이 매개변수는 각 포인트의 중요도를 제한한다

fig, axes = plt.subplots(3,3,figsize=(15,10))
for ax, C in zip(axes,[-1,0,3]):
    for a, gamma in zip(ax,range(-1,2)):
        mglearn.plots.plot_svm(log_C=C,log_gamma=gamma, ax =a)

axes[0,0].legend(["class 0", "class 1","class 0 support", "class 1 support"])

# SVM은 매개변수 설정과 데이터 스케일에 매우 민감하다. 
# 특히 입력 특성으,ㅣ 범위가 비슷해야 한다.
# 각 특성의 최솟값, 최대값을 로그 스케일로 보자
#plt.boxplot(X_train, manage_xticks=False)
#plt.yscale("symlog")
#plt.xlabel("feature list")
#plt.ylabel("feature size")

# 특서의 자릿수 자체가 다른 문제는 SVM에서는 영향이 아주 크다 이를 해결하기 위해선 데이터 전처리가 필요로하다

# 이 문제를 해결하는 방법 하나는 특성 값의 범위가 비슷해지도록 조정하는 것이다.
# 커널 SVM에서는 모든 특성 값을 평균이 0이고 단위 분산이 되도록 하거나 0과 1사이로 맞추는 방법을 사용한다

X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,random_state=0)

# 훈련 세트에서 특성별 최솟값 계산
min_on_training = X_train.min(axis=0)
# 훈련 세트에서 특성별 (최댓값 - 최솟값) 범위 계산
range_on_training = (X_train-min_on_training).max(axis=0)

# 훈련 데이터에 최솟값을 뺴고 범위로 나누면
# 각 특성에 대해 최솟값은 0 최댓값은 1이다
X_train_scaled = (X_train-min_on_training)/range_on_training

# 테스트 세트에도 같은 작업을 적용한다
# 하지만 훈련 세트에서 계산한 최솟값과 범위를 사용합니다
X_test_scaled = (X_test-min_on_training)/range_on_training

svc =SVC()
svc.fit(X_train_scaled,y_train)

print("훈련 세트 정확도: {:.3f}".format(svc.score(X_train_scaled,y_train)))
print("테스트 세트 정확도: {:.3f}".format(svc.score(X_test_scaled,y_test)))


# C나 gamma를 통해서 더 복잡한 모델을 만들 수 있다(증가로로
svc = SVC(C=1000)
svc.fit(X_train_scaled,y_train)

print("훈련 세트 정확도: {:.3f}".format(svc.score(X_train_scaled,y_train)))
print("테스트 세트 정확도: {:.3f}".format(svc.score(X_test_scaled,y_test)))

# SVM의 장점은 다양한 데이터셋에서도 잘 작동하며 데이터 특성이 얼마 안되도 복잡한 결정 경계를 만들 수 있다.
# 그러나 샘플이 많을 떄는 잘 맞지 않고 데이터 전처리와 매개변수 설정에 신경을 많이 써야한다.
# 그래서 요즘은 잘 안쓴다.


#%%
