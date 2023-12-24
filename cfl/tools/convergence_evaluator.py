class NumberOfRounds(object):
    def __init__(self, n_rounds):
        self.n_rounds = n_rounds

    def terminate(self, r):
        if r >= self.n_rounds:
            flag = True
        else:
            flag = False
        return flag
