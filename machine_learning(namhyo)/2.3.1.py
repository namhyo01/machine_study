#2창 지도학습

#지도 학습에는 분류와 회귀가 있다
#분류 :  미리 정의된 클래스 레이블 중 하나를 예측 (이진 분류와 다중 분류로 나누어 진다) 여기서 2중 분류는 yes/no에서 사용된다
#회귀 : 연속적인 숫자, 부동소수점(실수)를 예측한다 ex) 교육수준, 나이, 주거지를 바탕으로 연간 소득을 예측(출력 값에 연속성이 있으면 회귀 문제이다)

#일반화 : 모델이 데이터를 정확하게 예측한다면 훈련 세트에서 테스트 세트로 일반화 되었다고 한다
#일반화를 잘하긴 위해서는 훈련 세트만이 아닌 테스트 세트에서도 해야한다. 너무 많은 정보로 복잡한 모델을 만드는 것은 과대 적합이라고한다
#반대로 너무 간단한 모델은 과소 적합이라고 한다.
# 모델이 복잡하면 복잡할 수록 더 정확한 훈련데이터의 결과를 나오지만 반대로 새로운 데이터에는 잘 일반화가 되지 못한다.
# 최고의 모델은 일반화 성능이 최대가 되는 모델이다

# 데이터 셋에 다양하고 많은 데이터 포인트가 있을 수록 과대적합 없이 더 정확한 모델을 만들 수 있다.(데이터가 많으면 많을 수록 좋다)

# 특성이 적은 데이터 셋(저차원 데이터 셋)이 특성이 많은 데이터 셋(고차원 데이터 셋)에 그대로 유지되지 않을 수 있다
# 이런 사실을 알고 있다면 알고리즘을 배울 때 저차원 데이터셋을 사용하는 것이 좋다

#예를 들어 scikit-learn에 들어가 있는 유방암 종양의 임상데이터의 데이터 셋에서 알아보자(benign : 양성, malignant : 음성)
from header import *
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston

cancer = load_breast_cancer()
print("cancer.keys():\n",cancer.keys())
print("유방암 데이터 형태\n",cancer['data'].shape) #569개의 데이터 포인트를 가지며 특성은 30개이다
print("클래스별 샘플 개수:\n",{n: v for n, v in zip(cancer['target_names'], np.bincount(cancer['target']))})


# 회귀 분석용 데이터셋으로는 보스턴 주택가격 데이터 셋을 사용한다
boston = load_boston()
print("데이터의 형태",boston['data'].shape)
#여기서 13개의 특성뿐만 아니라 특성끼리 곱하여 의도적 으로 확장한다 이렇게 특성을 유도하는 것을 특성 공학이라고 한다
X,y = mglearn.datasets.load_extended_boston()
print("X.shpae", X.shape)