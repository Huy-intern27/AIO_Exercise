import math


def is_number(n):
    try:
        float(n)
    except ValueError:
        return False

    return True


def activate_function(x, func_name):

    if not is_number(x):
        return 'x must be number'

    x = float(x)

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def relu(x):
        return max(0, x)

    def elu(x, alpha=0.01):
        return x if x > 0 else alpha * (math.exp(-x) - 1)

    functions = {
        "sigmoid": sigmoid,
        "relu": relu,
        "elu": elu
    }

    if func_name not in functions:
        return f'{func_name} is not supported'

    result = functions[func_name](x)

    return f'{func_name}: f({x}) = {result}'
