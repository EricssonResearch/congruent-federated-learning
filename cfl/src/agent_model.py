from sklearn.model_selection import train_test_split


class AgentModel(object):
    def __init__(self, federated_scheme, data_normalizer, local_trainer, evaluator, validation_ratio=None):
        self.fl = federated_scheme
        self.data_normalizer = data_normalizer
        self.data_normalizer.__setattr__('fit_flag', False)
        self._local_trainer = local_trainer
        self.evaluator = evaluator
        self.validation_ratio = validation_ratio
        self._y_hat = None
        self._eval_dict = None

    @property
    def eval_dict(self):
        return self._eval_dict

    @property
    def y_hat(self):
        return self._y_hat

    @property
    def local_trainer(self):
        return self._local_trainer

    def _normalize(self, x, y):
        if self.validation_ratio is None:
            x_train = x
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.validation_ratio, shuffle=True)
        if self.data_normalizer.fit_flag is False:
            if len(x_train.shape) == 4:
                x_ = x_train.copy()
                x_ = x_.reshape(x_.shape[0], x_.shape[-1] * x_.shape[-2])
                self.data_normalizer.fit(x_)
            else:
                self.data_normalizer.fit(x_train)
            self.data_normalizer.__setattr__('fit_flag', True)

    def fit(self, x, y):
        if self.validation_ratio is None:
            x_train = x
            x_test = x[0: 2]
            y_train = y
            y_test = y[0: 2]
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.validation_ratio, shuffle=False)
        if self.data_normalizer.fit_flag is False:
            if len(x_train.shape) == 4:
                x_ = x_train.copy()
                x_ = x_.reshape(x_.shape[0], x_.shape[-1] * x_.shape[-2])
                self.data_normalizer.fit(x_)
                x_ = self.data_normalizer.transform(x_)
                x_train = x_.reshape(x_train.shape[0], 1, x_train.shape[-1], x_train.shape[-2])
            else:
                self.data_normalizer.fit(x_train)
                x_train = self.data_normalizer.transform(x_train)
            self.data_normalizer.__setattr__('fit_flag', True)

        self._local_trainer = self.fl.learn_local(x_train=x_train,
                                                  y_train=y_train,
                                                  x_val=x_test,
                                                  y_val=y_test,
                                                  trainer=self._local_trainer
                                                  )

    def predict(self, x):
        if len(x.shape) == 4:
            x_ = x.copy()
            x_ = x_.reshape(x_.shape[0], x_.shape[-1] * x_.shape[-2])
            x_ = self.data_normalizer.transform(x_)
            x = x_.reshape(x.shape[0], 1, x.shape[-1], x.shape[-2])
        else:
            x = self.data_normalizer.transform(x)
        self._y_hat = self.fl.predict_local(x=x, trainer=self._local_trainer)

    def evaluate(self, y):
        self._eval_dict = self.fl.evaluate(y=y, y_hat=self._y_hat, evaluator=self.evaluator)

    def fit_predict_evaluate(self, x_train, y_train, x_test, y_test):
        if len(y_test.shape) == 1:
            y_test = y_test.reshape(-1, 1)
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
        self.fit(x_train, y_train)
        self.predict(x_test)
        self.evaluate(y_test)
        return self._eval_dict

    def predict_evaluate(self, x_train, y_train, x_test, y_test):
        self._normalize(x_train, y_train)
        self.predict(x=x_test)
        self.evaluate(y_test)
        return self._eval_dict
