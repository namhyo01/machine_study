# 결정트리
# 예/아니오 라는 입장의 트리를 만들어서 나타내는 구조이다

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
from sklearn.tree import export_graphviz # .dot파일 형식으로 만들어줌(시각화)
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

def plot_feature_importances_cancer(model):
    #결정 트리의 특성 중요도를 차트로 보여줌
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("특성 중요도")
    plt.ylabel("특성")
    plt.ylim(-1, n_features)

#mglearn.plots.plot_animal_tree()

#결정트리를 만들때 질문 목록을 합습시키기 위해 하는 질문들을 테스트라고한다.
# 보통 특성 i는 값 a보다 큰가? 라는 형태이다.
# 가능한 모든 테스트에서 타깃값에 대해 가장 많은 정보를 가진 것을 고른다.
# 당연하게도 모든 리프노드가 순수 노드가 될때까지 진행하면  메우 복잡해지고 과대적합이된다.
# 그래서 트리 생성을 일찍 중단하는(사전 가지치기기랑 데이터 포인트가 적은 노드를 제거하는(사후 가지치기) 방법이 있다.
# 사전 가지치기 방법은 트리의 최대 깊이나 리프의 최대 개수를 제한하거나, 분할하기 위한 포인트의 최소 개수를 지정하는 것이다

#scikit-learn은 사전 가지치기만 지원한다

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify = cancer.target, random_state = 42)
tree= DecisionTreeClassifier(random_state=0)
tree.fit(X_train,y_train)

print("훈련 세트 정확도: {:.3f}".format(tree.score(X_train,y_train)))
print("테스트 세트 정확도: {:.3f}".format(tree.score(X_test,y_test)))
#선가지치기후
tree= DecisionTreeClassifier(max_depth=4,random_state=0)
tree.fit(X_train,y_train)

print("훈련 세트 정확도: {:.3f}".format(tree.score(X_train,y_train)))
print("테스트 세트 정확도: {:.3f}".format(tree.score(X_test,y_test)))

# .dot형식으로 저장
export_graphviz(tree,out_file="tree.dot", class_names=["악성","양성"],feature_names=cancer.feature_names, impurity=False, filled=True)

with open("tree.dot","r",encoding='utf-8') as f:
    dot_graph = f.read()
#display(graphviz.Source(dot_graph)) 출력



# 트리를 만드는 결정에 각 틍성이 얼마나 중요한지를 평가하는 특성 중요도 이다. 
# 이 값은 0과 1사이의 숫자로 0은 전혀 사용되지 않았고, 1은 완벽하게 타깃 클래스를 예측했다는 의미이다
# 다합치면 1이된다.

print("특성 중요도:\n", tree.feature_importances_)

#plot_feature_importances_cancer(tree) 차트화

# 첫 번째 노드에서 사용한 특성("worst radius")가 가장 중요한 특성으로 나타난다
# 선형모델의 계수(w)와 달리 특성 중요도는 항상 양수이다.
# 하지만 이 특성이 양성인지 악성인지 알 수가 없다
# 특성과 클래스 사이에는 간단하지 않은 관계가 있을수 있다

#tree = mglearn.plots.plot_tree_not_monotone() 잘 못만나는버젼
#display(tree)
#이경우에는 X[1]의 관계와 출력 클래스와의 관계는 단순하게 비례 또는 반비례 하지 않다

# 현재까지는 분류 트리를 보았지만 회귀 결정 트리에서도 비슷하게 적용된다
# 하지만 분류랑은 다르게 회귀에서는 외샵(extraploation), 즉 훈련 데이터의 범위 밖의 포인트에 대해 예측을 할 수 없다

# 참고로 그래프를 로그 스케일로 그리면 약간의 굴곡을 제외하고는 선형적으로 나타나서 비교적 예측하기가 쉬워진다
# 모델을 훈련하고 예측을 수행한 다음 로그 스케일을 되돌리기 위해 지수 함수를 적용한다
# 전체 데이터 셋을 돌리지만 테스트 데이터셋과의 비교가 관심대상이다.

#ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH,"ram_price.csv"))
ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "ram_price.csv"))

#plt.semilogy(ram_prices.date, ram_prices.price)
plt.xlabel("년")
plt.ylabel("가격 ($/Mbyte_")

# 2000년 이전을 훈련 데이터로, 이후를 테스트 데이터로 만든다
data_train = ram_prices[ram_prices.date<2000]
data_test = ram_prices[ram_prices.date>=2000]

# 가격 예측을 위해 날짜 특성만을 이용한다
X_train = data_train.date[:,np.newaxis]
# 데이터와 타깃 사이의 관계를 간단하게 만들기 위해 로그 스케일로 바꾼다
y_train = np.log(data_train.price)

tree = DecisionTreeRegressor().fit(X_train,y_train)
linear_reg = LinearRegression().fit(X_train,y_train)

# 예측은 전체 기간에 대해서 수행한다
X_all = ram_prices.date[:, np.newaxis]

pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)

# 예측한 값의 로그 스케일을 되돌린다
price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)


plt.semilogy(data_train.date, data_train.price, label="train data")
plt.semilogy(data_test.date,data_test.price,label='test data')
plt.semilogy(ram_prices.date, price_tree,label='tree predict')
plt.semilogy(ram_prices.date, price_lr, label='linear predict')
plt.legend()

# 선형 모델은 직선으로 데이터를 근사함
# 테스트 데이터 2000년 이후를 꽤 정확히 예측한다
# 반면 트리 모델은 훈련 데이터는 완벽하계 예측한다
# 그러나 모델이 가진 데이터 범위 밖으로 나가면 단순히 마지막 포인트를 이요해 예측하는게 전부이다
# 트리 모델은 훈련 데이터 밖의 새로운 데이터를 예측할 능력이 없다
# 이것이 모든 트리 기반 모델의 공통의 단점이다


#%%
