#데이터 적재
# 붓꽃 데이터셋(dictionary 형식처럼 생겼다)
from header import *
%matplotlib inline
#%%
from sklearn.datasets import load_iris #scikit-learn 모듈
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier #k-최근접 이웃 알고리즘
iris_dataset = load_iris()

#iris_dataset.keys() : 키값
#iris_dataset['target_names'] : 예측하는 붓꽃 품종의 이름
#iris_dataset['feature_names'] : 각 특성을 설명
#iris_dataset['data'] : 실제데이터는 target과 datㅁ필드에 들어있다.
#iris_dataset['data'].shape: data의 크기
#머신러닝에서 각 아이템을 샘플이라하며 속성을 특성이라고 한다
#그러므로 data의 크기는 샘플수 * 특성수이다
#iris_dataset['target'] : 품종을 타나탠다
#iris_dataset['target].shape : 크기
#target의 이름은 target_names에서 확인 가능(0 , 1, 2)순으로 나타내어져 있다

print("iris_dataset 키:\n", iris_dataset.keys())
print("타깃의 이름: ", iris_dataset['target_names'])
print("data의 처음 다섯행 \n", iris_dataset['data'][:5])
#    iris_dataset.feature_names

#성과측정
#머신러닝에서 평가목적으로 쓰는 데이터를 훈련데이터라고한다
#그럼 나머지 데이터는 잘 작동하는지 측정하기 위해 사용되며 테스트 데이터라고한다
#X는 데이터, y는 레이블로 가정한다.(레이블은 특정 데이터에 대한 출력이다(ex) 붓꽃(데이터)에 대한 꽃의 품종을 레이블이라고한다)
#X는 데이터이므로 2차 배열형식이고, y는 1차 배열이다
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state = 0)    

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=8,cmap=mglearn.cm3)

#k-최근접 이웃 알고리즘 
knn = KNeighborsClassifier(n_neighbors=1)#이웃 한개만 찾겠다
#훈련 데이터셋으로부터 모델을 만들기 위해선 fit메소드 이용해야함
knn.fit(X_train,y_train)

#예측하기
X_new = np.array([[5,2.9,1,0.2]]) #이 길이를 가진 붗꽃은 과연 어떤 품종일까
prediction = knn.predict(X_new)
print("예측: ",prediction)
print("예측한 타깃(품종)의 이름:", iris_dataset['target_names'][prediction])

#평가하기
#테스트 세트에 있는 데이터를 실제 레이블과 비교 => 정확도 체크
y_pred = knn.predict(X_test)
print("예측 값:\n",y_pred)
#정확도 체크법    
print("테스트 세트의 정확도: {:.2f}".format(knn.score(X_test,y_test)))


#%%


#%%
