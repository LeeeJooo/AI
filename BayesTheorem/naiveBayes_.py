import numpy as np
from collections import defaultdict
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class NaiveBayes():
    def __init__(self, type='GaussianNB'):
        self.class_unique = None
        self.n_class = None
        self.n_feature = None
        self.mu_ = None
        self.var_ = None
        self.type = type
        self.class_count_ = None
        self.epsilon_ = None
        self.class_prior_ = None

    def fit(self, X, y):
        # 1. 초기화
        self.class_unique = np.unique(y)
        self.n_class = len(self.class_unique)
        self.n_feature = X.shape[1]
        self.mu_ = np.zeros((self.n_class, self.n_feature))
        self.var_ = np.zeros((self.n_class, self.n_feature))
        self.class_count_ = np.zeros(self.n_class)
        self.epsilon_ = 1e-9 * np.var(X, axis=0).max()

        # 2. 클래스 별 조건부 확률 구하기
        for cls in self.class_unique:
            i = self.class_unique.searchsorted(cls) # cls의 인덱스 반환
            X_i = X[y==cls, :]

            # 평균, 분산 구하기
            mu_new, var_new = self._update_mean_variance(
                  self.class_count_[cls], self.mu_[i, :], self.var_[i:,], X_i
            )

            self.class_count_[i] = np.sum(y==cls)   # 클래스 별 데이터 개수
            self.mu_[i, :] = mu_new
            self.var_[i, :] = var_new
        
        self.var_[:,:] += self.epsilon_

        self.class_prior_ = self.class_count_/self.class_count_.sum()

        return self
    
    def _update_mean_variance(self, n_past, mu_past, var_past, X):
        if X.shape[0]==0: 
            return mu_past, var_past
        
        n_new = X.shape[0]
        mu_new = np.mean(X, axis=0) # (n_feature,)
        var_new = np.var(X, axis=0) # (n_feature,)

        if n_past == 0:
            return mu_new, var_new
        
        n_total = float(n_past+n_new)
        mu_total = (n_past*mu_past + n_new*mu_new) / n_total
 
        # ssd = n x var
        ssd_past = n_past * var_past
        ssd_new = n_new * var_new
        ssd_total = ssd_past + ssd_new + (n_new*n_past/n_total) * (mu_past-mu_new)**2
        var_total = ssd_total/n_total

        return mu_total, var_total
    
    def predict(self, X):
        probs = self._joint_log_likelihood(X)

        return self.class_unique[np.argmax(probs, axis=1)]

    def _joint_log_likelihood(self, X):
        probs = []

        for c_i in range(self.n_class):
            p = np.log(self.class_prior_[c_i])
            p -= 0.5 * np.sum(np.log(2.0 * np.pi * self.var_[c_i,:]))
            p -= 0.5 * np.sum(((X-self.mu_[c_i,:]) ** 2) / self.var_[c_i,:], axis=1)
            probs.append(p)
   
        probs = np.array(probs).T
        return probs
    
if __name__ == "__main__":
    print('s')
    gnb = NaiveBayes(type='GaussianNB')
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = gnb.fit(X, y)
    y_pred = model.predict(X_test)
    print(f'acc = {(y_test==y_pred).sum()}')