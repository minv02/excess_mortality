class TOPALS:
    def __init__(self):
        self.k = None
        self.p = None
        self.w = None
        self.a = None
        self.b = None

    def initialize_params(self, std, knot_positions, boundaries, penalty_precision=2):
        self.std = np.array(std)
        self.knot_positions = knot_positions
        self.boundaries = boundaries
        self.penalty_precision = penalty_precision

        a = len(self.std)
        age = np.arange(0, a)
        self.b = patsy.bs(
            degree=1,
            x=age,
            knots=self.knot_positions,
            include_intercept=False
        )
        self.k = self.b.shape[1]

        d1 = np.diff(np.eye(self.k), 1, axis=0)
        self.p = self.penalty_precision * d1.T @ d1

        g = len(self.boundaries) - 1
        nages = np.diff(self.boundaries)

        self.w = np.zeros((g, a))
        offset = 0
        for i in range(g):
            start = offset
            end = nages[i] + offset
            self.w[i, start:end] = 1 / nages[i]
            offset += nages[i]

        self.a = np.zeros((1, self.k))[0]

    def q(self, alpha):
        mu = np.exp(self.std + self.b @ alpha)
        m = self.w @ mu
        likelihood = np.sum(self.D * np.log(m) - self.N * m)
        penalty = 1 / 2 * alpha.T @ self.p @ alpha
        return likelihood - penalty

    def next_alpha(self, alpha):
        mu = np.exp(self.std + self.b @ alpha)
        m = self.w @ mu
        d_hat = self.N * m
        x = self.w @ np.diag(mu) @ self.b
        a = np.diag(self.N / m)
        y = (self.D - d_hat) / self.N + x @ alpha
        updated_alpha = np.linalg.solve(x.T @ a @ x + self.p, x.T @ a @ y)
        return updated_alpha

    def fit(self, N, D, std, knot_positions, boundaries, max_iter=50, alpha_tol=0.00005):
        """
        Train the model using the TOPALS method.

        Parameters:
        - N (array): Array of N values.
        - D (array): Array of D values.
        - std (array): Array of standard deviation values.
        - knot_positions (list): List of knot positions for basis functions.
        - boundaries (array): Array of boundaries for determining weight coefficients.
        - max_iter (int): Maximum number of iterations.
        - alpha_tol (float): Allowable difference between consecutive alpha values for convergence.

        Returns:
        self
        """
        self.N = np.array(N)
        self.D = np.array(D)
        self.initialize_params(std, knot_positions, boundaries)
        n_iter = 0
        while n_iter < max_iter:
            n_iter += 1
            last_param = self.a
            self.a = self.next_alpha(self.a)
            change = self.a - last_param
            converge = np.all(np.abs(change) < alpha_tol)
            if converge or n_iter == max_iter:
                break
        self.logmx = self.std + (self.b @ self.a)
        return self

    def transform(self):
        return self.logmx

    def fit_transform(self, N, D, std, knot_positions, boundaries, max_iter=100, alpha_tol=0.00005):
        """
        Train the model using the TOPALS method.

        Parameters:
        - N (array): Array of N values.
        - D (array): Array of D values.
        - std (array): Array of standard deviation values.
        - knot_positions (list): List of knot positions for basis functions.
        - boundaries (array): Array of boundaries for determining weight coefficients.
        - max_iter (int): Maximum number of iterations.
        - alpha_tol (float): Allowable difference between consecutive alpha values for convergence.

        Returns:
        logmx
        """
        self.fit(N, D, std, knot_positions, boundaries, max_iter=max_iter, alpha_tol=alpha_tol)
        return self.logmx
