# k-최근접 이웃 알고리즘
from header import *

%matplotlib inline

#%%
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsRegressor
X,y = mglearn.datasets.make_forge()

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 0)
clf = KNeighborsClassifier(n_neighbors=3)
#훈련세트를 사용하요 분류 모델을 학습시킨다
clf.fit(X_train, y_train)
#테스트 세트 예측
print("테스트 세트 예측: ",clf.predict(X_test))
#정확도 검사
print("정확도: {:.2f}".format(clf.score(X_test,y_test)))

#당연하게도 n_neighbors 의 값이 커지면 모델의 복잡도가 더 낮아지고, 값이 적으면 모델의 복잡도가 더 커진다
# 과연 모델의 복잡도와 일반화 사이의 관계를 알아보자(예시로 유방암 데이터셋을 사용합니다)
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer['data'],cancer['target'],stratify=cancer['target'],random_state = 66)

training_accuracy = []
test_accuracy = []
# 1에서 10까지의 n_neighbors를 적용
neighbors_settings = range(1,11)

for n_neighbors in neighbors_settings:
    #모델 생성
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    #훈련 세트 정확도 저장
    training_accuracy.append(clf.score(X_train,y_train))
    #일반화 정확도 저장
    test_accuracy.append(clf.score(X_test,y_test))

plt.plot(neighbors_settings, training_accuracy, label="훈련 정확도") 
plt.plot(neighbors_settings, test_accuracy,label="테스트 정확도")
plt.ylabel("정확도")
plt.xlabel("n_neighbors")
plt.legend()

#k=최근접 이웃 회귀
#wave 데이터셋을 이용해서 이웃이 하나인 최근접 이웃을 만든다

mglearn.plots.plot_knn_regression(n_neighbors=1)
mglearn.plots.plot_knn_regression(n_neighbors=3)

#회귀를 위한 k-최긎버 이웃 알고리즘은 KNeighborsRegressor에 구현되어져 있다

X,y = mglearn.datasets.make_wave(n_samples=40)
#wave 데이터 셋을 훈련 세트와 테스트 세트로 나눈다
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 0)

#이웃의 수를 3으로 하여 모델 객체를 만듭니다
reg = KNeighborsRegressor(n_neighbors=3)
#모델을 학습 시킨다
reg.fit(X_train, y_train)

print("테스트 세트 예측:\n", reg.predict(X_test))
# 또한 score메소드를 사용해서 모델을 평가할 수 있다. 회귀일 때는 R^2을 반환하며 0과 1사이의 수 에서 1은 예측이 완벽, 0은 훈련 세트의 출력값인 y_train의 평균으로만 예측하는 모델이 되는 경우다
# R^2는 음수가 될 수 있다. 이는 예측과 타깃이 상반되 경향을 가지는 경우이다

print("테스트 세트 R^2: {:.2f}".format(reg.score(X_test,y_test)))

#이렇게 회귀에서도 n_neighbors가 많아야만 더 안정한 예측을 얻게 된다

#KNeighbors 분류기에는 중요한 매개변수가 2개가 있다. 데이터 포인터 사이의 거리를 재는 방법과 이웃의 수이다.
# 그중 거리 재는 방법은 유클라디안 거리 방식을 채용한다
# 하지만 예측이 느리고 많은 특성을 처리하는 능력이 부족해 현업에서는 잘안쓰인다
# 대신 선형 모델이 이런 단점이 없는 알고리즘이다

#%%
