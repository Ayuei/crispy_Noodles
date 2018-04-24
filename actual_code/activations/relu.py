class relu():

    def __activate__(self, x):
        return np.maximum(x,0)

    def __deriv__(self, x):

        x[x <= 0] = 0
        x[x > 0] = 1

        return x

