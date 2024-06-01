class Kannisto:
    def __init__(
            self,
            fit_ages_interval=[60, 80],
            predict_ages_interval=[81, 100],
            age_groups=5
    ):
        self.mx = None
        self.all_ages = None
        self.observed = None
        self.fit_ages_interval = fit_ages_interval
        self.predict_ages_interval = predict_ages_interval
        self.age_groups = age_groups
        self.process_ages()
        self.coef = None
        self.fitted = None
        self.residuals = None

    def fit(self, mx):
        self.mx = mx
        self.all_ages = mx.index.values
        self.observed = mx.values
        self.estimate()
        return self

    def transform(self):
        mx = self.mx
        x = self.predict_ages
        c = self.coef[0]
        d = self.coef[1]
        fitted = pd.Series(c * np.exp(d * x) / (1 + c * np.exp(d * x)), x)
        orig_ages = list(set(self.all_ages) - set(self.predict_ages))
        self.fitted = pd.concat([mx[orig_ages], fitted]).sort_index()
        return self.fitted

    def fit_transform(self, mx):
        self.fit(mx)
        return self.transform()

    def process_ages(self):
        f_start = self.fit_ages_interval[0]
        f_end = self.fit_ages_interval[1] + 1
        p_start = self.predict_ages_interval[0]
        p_end = self.predict_ages_interval[1] + 1
        if self.age_groups == 1:
            self.fit_ages = np.arange(f_start, f_end, 1)
            self.predict_ages = np.arange(p_start, p_end, 1)
        elif self.age_groups == 5:
            self.fit_ages = np.arange(f_start, f_end, 5)
            self.predict_ages = np.arange(p_start, p_end, 5)
        else:
            raise ValueError('Only 1 or 5 year age groups are currently supported')

    def estimate(self):
        x = self.fit_ages
        values = self.mx[self.fit_ages]
        y = np.log(values) - np.log(1 - values)
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        c = np.exp(model.params.iloc[0])
        d = model.params.iloc[1]
        self.coef = [c, d]
        self.fitted = pd.Series(c * np.exp(d * x) / (1 + c * np.exp(d * x)), x)
        self.residuals = values - self.fitted
