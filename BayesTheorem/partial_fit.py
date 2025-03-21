    def _partial_fit(self, X, y, classes=None, _refit=False, sample_weight=None):
        """Actual implementation of Gaussian NB fitting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        classes : array-like of shape (n_classes,), default=None
            List of all the classes that can possibly appear in the y vector.

            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        _refit : bool, default=False
            If true, act as though this were the first time we called
            _partial_fit (ie, throw away any past fitting and start over).

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
        """
        if _refit:
            self.classes_ = None

        first_call = _check_partial_fit_first_call(self, classes)
        X, y = validate_data(self, X, y, reset=first_call)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        # If the ratio of data variance between dimensions is too small,
        # it will cause numerical errors.
        # To address this,
        # we artificially boost the variance by epsilon,
        # a small fraction of the standard deviation of the largest dimension.
        self.epsilon_ = self.var_smoothing * np.var(X, axis=0).max()
        '''
        >> np.var(X, axis=0) : X의 FEATURE 별 분산 구하기
        >> np.var(X, axis=0).max() : 최대 분산 SELECT
        >> self.var_smoothing : 분산을 평활화(smoothing) 하는 역할
        '''

        if first_call:
            # This is the first call to partial_fit:
            # initialize various cumulative counters
            n_features = X.shape[1]
            n_classes = len(self.classes_)
            self.theta_ = np.zeros((n_classes, n_features))
            self.var_ = np.zeros((n_classes, n_features))

            self.class_count_ = np.zeros(n_classes, dtype=np.float64)

            # Initialise the class prior
            # Take into account the priors
            if self.priors is not None:
                priors = np.asarray(self.priors)
                # Check that the provided prior matches the number of classes
                if len(priors) != n_classes:
                    raise ValueError("Number of priors must match number of classes.")
                # Check that the sum is 1
                if not np.isclose(priors.sum(), 1.0):
                    raise ValueError("The sum of the priors should be 1.")
                # Check that the priors are non-negative
                if (priors < 0).any():
                    raise ValueError("Priors must be non-negative.")
                self.class_prior_ = priors
            else:
                # Initialize the priors to zeros for each class
                self.class_prior_ = np.zeros(len(self.classes_), dtype=np.float64)
        else:
            if X.shape[1] != self.theta_.shape[1]:
                msg = "Number of features %d does not match previous data %d."
                raise ValueError(msg % (X.shape[1], self.theta_.shape[1]))
            # Put epsilon back in each time
            self.var_[:, :] -= self.epsilon_
            '''
            분산 평활화화
            '''

        classes = self.classes_

        unique_y = np.unique(y) # Label의 고유값 집합합
        unique_y_in_classes = np.isin(unique_y, classes)
        '''
        classes : 모들이 학습해야 하는 전체 클래스 레이블 집합
        unique_y : 데이터에서 실제로 관측된 고유 클래스 레이블 집합
        unique_y_in_classes : unique_y가 classes에 포함되는지 여부

        >> 데이터의 일부만 사용할 경우 classes와 unique_y가 다를 수 있음
        '''

        if not np.all(unique_y_in_classes):
            raise ValueError(
                "The target label(s) %s in y do not exist in the initial classes %s"
                % (unique_y[~unique_y_in_classes], classes)
            )

        for y_i in unique_y:
            i = classes.searchsorted(y_i)   # np.searchsorted() : 정렬된 배열에서 특정 값이 삽입될 위치를 반환.
            X_i = X[y == y_i, :] # 레이블이 y_i인 X만.

            if sample_weight is not None:
                sw_i = sample_weight[y == y_i]
                N_i = sw_i.sum()
            else:
                sw_i = None
                N_i = X_i.shape[0]  # n_samples

            new_theta, new_sigma = self._update_mean_variance(
                n_past=self.class_count_[i],    # 클래스 별 샘플 개수수
                mu=self.theta_[i, :],           # 클래스별 평균
                var=self.var_[i, :],            # 클래스별 분산
                X=X_i,                          # 현재 클래스 i에 속한 새로운 샘플 데이터
                sample_weight=sw_i              # 새로운 샘플의 가중치 벡터터
            )

            self.theta_[i, :] = new_theta   # 평균
            self.var_[i, :] = new_sigma     # 분산
            self.class_count_[i] += N_i

        self.var_[:, :] += self.epsilon_

        # Update if only no priors is provided
        if self.priors is None:
            # Empirical prior, with sample_weight taken into account
            self.class_prior_ = self.class_count_ / self.class_count_.sum()

        return self