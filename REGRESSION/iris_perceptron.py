from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

# setosa 분류하기
class IrisBinaryClassification:
    def __init__(self, epochs, lr, random_state):
        np.random.seed(random_state)

        self.epochs = epochs
        self.lr = lr
        self.perceptron_weights = np.array(np.random.randn(2))
        self.perceptron_b = np.random.randn()
        self.logistic_weights = np.array(np.random.randn(2))
        self.logistic_b = np.random.randn()

    def perceptron_fit(self, inputs, labels):
        for _ in range(self.epochs):
            for input, label in zip(inputs, labels):
                # prediction
                linear_output =  np.dot(input, self.perceptron_weights) + self.perceptron_b
                pred = 1 if linear_output > 0 else 0
                
                # 잘못된 예측일 때만 가중치 업데이트
                if pred!=label:
                    err = label-linear_output   # layer 1개라서 가능 , 다층 layer일 경우 미분 필요
                    self.perceptron_weights += self.lr*err
                    self.perceptron_b += self.lr * err
        
        print(1)

    def perceptron_predict(self, inputs):
        linear_output =  np.dot(inputs, self.perceptron_weights) + self.perceptron_b
        pred = (linear_output >= 0).astype(int)
        return pred
    
    def logistic_fit(self, inputs, labels):
        for _ in range(self.epochs):
            linear_output =  np.dot(inputs, self.logistic_weights) + self.logistic_b
            sigmoid_output = self.sigmoid(linear_output)
            
            err = self.logloss(sigmoid_output,labels)
            self.logistic_weights += self.lr*err
            self.logistic_b += self.lr*err

    def logistic_predict(self, inputs):
        linear_output =  np.dot(inputs, self.logistic_weights) + self.logistic_b
        sigmoid_output = self.sigmoid(linear_output)
        pred = (sigmoid_output >= 0.5).astype(int)
        return pred
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def logloss(self,preds,labels):
        return sum(np.log(preds)*labels - (1-labels)*np.log(1-preds))/preds.shape[0]
    
iris = load_iris()

idx = np.in1d(iris.target, [0,2])
idx2 = np.in1d(iris.target, [2])
inputs = iris.data[idx,0:2]
iris.target[idx2]=1
labels = iris.target[idx]


x_sepal_length_min, x_sepal_length_max = inputs[:,0].min()-1, inputs[:,0].max()+1
x_sepal_width_min, x_sepal_width_max = inputs[:,1].min()-1, inputs[:,1].max()+1

model = IrisBinaryClassification(epochs=50, lr=0.001, random_state=24)
model.perceptron_fit(inputs, labels)
model.logistic_fit(inputs, labels)
xx, yy = np.meshgrid(np.linspace(x_sepal_length_min, x_sepal_length_max, 1000),
            np.linspace(x_sepal_width_min, x_sepal_width_max, 1000))
perceptron_pred = model.perceptron_predict(np.c_[xx.ravel(), yy.ravel()])
perceptron_pred = perceptron_pred.reshape(xx.shape)

logistic_pred = model.perceptron_predict(np.c_[xx.ravel(), yy.ravel()])
logistic_pred = logistic_pred.reshape(xx.shape)

plt.rc('font', family='Malgun Gothic')
id = [10, 22, 36, 49]
_, axes = plt.subplots(1, 2, figsize=(12,5))

axes[0].contour(xx, yy, perceptron_pred, levels=[0], colors='blue', linestyles='dashed', label="퍼셉트론")
axes[0].scatter(inputs[:,0], inputs[:,1], c=labels, edgecolors='k')

axes[0].scatter(inputs[id, 0], inputs[id, 1], c='r')
for i in id:
    axes[0].annotate(i, xy=(inputs[i, 0], inputs[i, 1] + 0.1))
axes[0].grid(False)
axes[0].set_title("퍼셉트론 판별영역")

axes[0].set_xlabel("sepal length")
axes[0].set_ylabel("sepal width")

axes[1].contour(xx, yy, logistic_pred, levels=[0.5], colors='red', linestyles='solid', label="로지스틱 회귀")
axes[1].scatter(inputs[:,0], inputs[:,1], c=labels,edgecolors='k')

axes[1].scatter(inputs[id, 0], inputs[id, 1], c='r')
for i in id:
    axes[1].annotate(i, xy=(inputs[i, 0], inputs[i, 1] + 0.1))
axes[1].grid(False)
axes[1].set_title("로지스틱 판별영역")

axes[1].set_xlabel("sepal length")
axes[1].set_ylabel("sepal width")

plt.show()