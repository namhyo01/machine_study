# 선형 모델

# 입력 특성에 대한 선형 함수를 만들어 예측을 수행
# 선형 회귀 모델은 회귀 계수가 선형 결합으로 표현 할 수 있는것(즉 x나 y가 아닌 것들이 선형으로 표현 가능하면 된다.w같은것 말이다)
# y = w[0]*x[0] + w[1]*y[1] + .... + w[p]*x[p]+b
#x[0]~x[p]는 하나의 데이터 포인트에 대한 특성
#w,b는 모델이 학습할 파라미터이다
#y는 모델이 만들어 낸 예측값이다
# w[0]은 기울기고 b는 y절편이다
#특성이 많아지면 w는 각 특성에 해당하는 기울기를 모두 가진다
from header import *
%matplotlib inline
#%%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
mglearn.plots.plot_linear_regression_wave()

#회귀를 위한 선형 모델은 특성이 하나일 떈 직선, 두개.일 떈 평면, 더 높은 차원에서는 초 평면이 된다
# 1차원 데이터만 보면 이것은 매우 과한 가정이다.(제약이 많다)
# 하지만 특성이 만흥ㄴ 데이터셋이라면 선형 모델은 훌륭한 성능을 낼 수 있다
# 훈련 데이터보다 특성이 더 만흔 경우 어떤 타깃 y도 선형함수로 완벽하게 모델링 가능하다.
# 알고리즘이 주어진 데이터로부터 학습하는 파라미터를 흔히 모델 파라미터 라고부른다.(파라미터, 계수라고도 한다다
# 모델이 스스로 학습할 수 없어 사람이 설정해야하는 파라미터를 하이퍼파라미터(통칭 매개변수라고 부른다)

#선형 회귀(최소제곱법)
# 선형 회귀는 예측과 훈련 세트에 있는 타깃 y사이의 평균 제곱 오차를 최소화하는 파라미터 w와 b를 구한다
# 예측값과 타깃값의 차이를 제곱하여 더한 후 샘플의 개수로 나눈다 : 평균 제곱 오차
# 매개 변수가 없는 것이 장점이나 모델의 복잡도를 제어할 수 없다

X,y  = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)
lr = LinearRegression().fit(X_train,y_train)
# 기울기 파라미터는 lr.coef_속성에 저장되어있다(w)
# 절편 파라미터인 b는 intercept_속성에 저장되어있다
print("lr.coef: ",lr.coef_)
print("lr.intercept: ",lr.intercept_)

print("훈련 세트 점수: {:.2f}".format(lr.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lr.score(X_test, y_test)))

#두 점수가 비슷하나 매우 낮은 것을 알 수 있다. 이러한 의미는 과소 적합이라는 것을 의미한다
# 반대로 너무 고차원 데이터셋에서는 선형 모델의 성능이 매우 떨어진다(과대적합)


#리지 회귀

#규제 : 가중치의 절댓값을 작게 만든다. 즉 가중치의 모든 원소를 0에 가깝게 만들어서 모든 특성이 출력에 주는 여향을 최소한으로 만든다. 
# a를 더 많이 커지게 하면 규제가 커지므로 w가 0에 가까워 진다. 따라서 데이터 특성을 더 많이 무시하기 때문에 정확도는 나빠지지만 복잡도가 낮아져 더 일반화된 모델이 된다
#라소 회귀 : L1규제를 쓰며, 모돈 특성을 다 이용하지 않는다.

#분류용 선형 모델 다 규제를 사용한다.: 다만 얘는 C를 쓰는데. C>0 = > 규제는 작아진다


#다중 클래스 분류용 선형 모델
# 로지스틱 회귀만 제외하고 많은 선형 분류 모델은 태생적으로 이진 분류만을 지원한다.
# 따라서 다중 클래스로 확장하기 위해서는 일대다 기법을 이용한다.
# 방법은 각 클래스를 다른 모든 클래스와 구분하도록 이진 분류 모델을 학습시킨다.
# 예측을 할떄 모든 이진 분류기가 작동하여 가장 높은 점수를 내는 분류기의 클래스를 예측값으로 선택한다.
# 결국 각각의 분류기가 w와 b를 가지므로 이 공식이 사용된다
# w[0]*x[0]+w[1]*x[1]+...+w[p]*x[p]+b

#%%
