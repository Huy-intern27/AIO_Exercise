def factorial(n):
    if n == 0:
        return 1

    return n * factorial(n - 1)

def cal_sin(x, n):
    if n <= 0:
        return 'n must be greater than zero'

    return sum(((-1) ** i) * ((x ** (2 * i + 1)) / factorial(2 * i + 1)) for i in range(n))

def cal_cos(x, n):
    if n <= 0:
        return 'n must be greater than zero'

    return sum(((-1) ** i) * ((x ** (2 * i)) / factorial(2 * i)) for i in range(n))

def cal_sinh(x, n):
    if n <= 0:
        return 'n must be greater than zero'

    return sum((x ** (2 * i + 1)) / factorial(2 * i + 1) for i in range(n))

def cal_cosh(x, n):
    if n <= 0:
        return 'n must be greater than zero'

    return sum((x ** (2 * i)) / factorial(2 * i) for i in range(n))

