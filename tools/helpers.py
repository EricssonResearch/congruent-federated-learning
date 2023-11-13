from torch.autograd import Variable


class TorchFunctions(object):
    """ Contains useful functions in torch. Functions are static methods. """
    @staticmethod
    def make_float_variable(x):
        return Variable(x).float()

    @staticmethod
    def make_long_variable(x):
        return Variable(x).long()
