class LeeCarter:
    def __init__(self):
        self.is_fitted = False

    def __str__(self):
        return f"LeeCarter forecasting model. Is fitted={self.is_fitted}"

    def fit(
            self,
            mxt_df: pd.DataFrame,
            sex: str,
            stopyear: object = None,
            adjust: object = None,
            smooth=None,
            regularize=False
    ) -> object:
        self.mxt_m, self.ages_v, self.years_v = self.unwrap_mxt_df(mxt_df)
        self.sex = sex
        self.regularize = regularize
        self.log_mxt_m = np.log(mxt_df)
        if smooth == 'spline':
            self.log_mxt_m = preprocess.smooth_bivariate_spline(self.log_mxt_m, s=3)
        self.ax_v, self.kt_v, self.bx_v = self.fit_kt_bx()
        self.kt_se_s = self.get_ktse()
        self.log_mxt_fit_m = self.fit_mx()
        self.mxt_fit_m = np.exp(self.log_mxt_fit_m)
        self.e0_fit_v = self.fit_e0()
        self.e0_actual_v = self.get_e0()
        self.kt_model = self.fit_kt_forecasting_model()
        if adjust == "e0":
            self.kt_model = None
            self.kt_v = self.adjust_by_e0()
            self.log_mxt_fit_adjusted_m = np.outer(self.kt_v, self.bx_v) + self.ax_v
            self.mxt_fit_adjusted_m = np.exp(self.log_mxt_fit_adjusted_m)
            self.kt_model = self.fit_kt_forecasting_model()
            self.e0_fit_v = np.array([life_expectancy(
                mx, self.ages_v, sex=self.sex, restype='e0') for mx in self.mxt_fit_adjusted_m])
            self.e0_match = np.allclose(self.e0_actual_v, self.e0_fit_v)
        self.stopyear = stopyear
        self.is_fitted = True
        return self

    def forecast(self, n=1):
        forecast = self.kt_model.get_forecast(steps=n)
        kt_mean = forecast.predicted_mean
        kt_std = np.sqrt(forecast.var_pred_mean)
        kt_forecast_conf_int = forecast.conf_int(alpha=0.01)
        mx_predicted = [np.exp(self.ax_v + kt * self.bx_v) for kt in kt_mean]
        e0_predicted = [life_expectancy(mx, self.ages_v, self.sex, restype='e0') for mx in mx_predicted]
        res = {
            'kt': kt_mean,
            'kt_std': kt_std,
            'mx': mx_predicted,
            'e0': e0_predicted
        }
        res = [kt_mean, e0_predicted, kt_forecast_conf_int, kt_std, mx_predicted]
        return res

    def adjust_by_e0(self):
        def optimize(guess, e0_actual_s, ax_v, bx_v):
            mx_fitted_v = np.exp(ax_v + guess * bx_v)
            e0_fitted_s = life_expectancy(mx_fitted_v, self.ages_v, self.sex)
            return abs(e0_actual_s - e0_fitted_s)

        kt_adj_v = []
        for i in range(len(self.kt_v)):
            e0_actual_s = self.e0_actual_v[i]
            result = minimize_scalar(optimize, args=(e0_actual_s, self.ax_v, self.bx_v))
            kt_adj_s = result.x
            kt_adj_v.append(kt_adj_s)

        return np.array(kt_adj_v)

    def unwrap_mxt_df(self, mxt_df):
        mxt_m = mxt_df.values
        ages_v = np.array(mxt_df.columns.astype(int))
        years_v = np.array(mxt_df.index.astype(int))
        return mxt_m, ages_v, years_v

    def fit_kt_bx(self):
        ax_v = np.array(self.log_mxt_m.mean(axis=0))
        Axt_m = self.log_mxt_m.sub(ax_v, axis=1)
        U_m, S_v, V_m = np.linalg.svd(Axt_m, full_matrices=True)
        sumv_s = np.sum(V_m[0, :])
        s = S_v[0]
        kt_v = U_m[:, 0] * sumv_s * s
        bx_v = V_m[0, :] / sumv_s
        return ax_v, kt_v, bx_v

    def fit_mx(self, restype='log_mxt_m'):
        centered_log_mxt_fit_m = np.outer(self.kt_v, self.bx_v)
        log_mxt_fit_m = centered_log_mxt_fit_m + self.ax_v
        return log_mxt_fit_m

    def fit_e0(self):
        e0_fit_v = np.array([life_expectancy(
            mx, self.ages_v, sex=self.sex, restype='e0') for mx in self.mxt_fit_m])
        return e0_fit_v

    def get_e0(self):
        e0_actual_v = np.array([life_expectancy(
            mx, self.ages_v, sex=self.sex, restype='e0') for mx in self.mxt_m])
        return e0_actual_v

    def fit_kt_forecasting_model(self):
        model = ARIMA(self.kt_v, order=(1, 1, 1))
        model_fit = model.fit()
        return model_fit


    def get_ktse(self):
        x = np.arange(1, len(self.years_v) + 1)
        kt_se_s = linregress(x, self.kt_v).stderr
        return kt_se_s
