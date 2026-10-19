import sys
import math


def read_system(filename):
    with open(filename) as f:
        data = f.read().split()
    n = int(data[0])
    if len(data) != 1 + n*n + n:
        raise ValueError("неверный формат файла")
    A = [[float(data[1 + i*n + j]) for j in range(n)] for i in range(n)]
    b = [float(data[1 + n*n + i]) for i in range(n)]
    return A, b


def build_alpha_beta(A, b):
    n = len(A)
    alpha = [[0.0] * n for _ in range(n)]
    beta = [0.0] * n
    for i in range(n):
        if A[i][i] == 0.0:
            raise ValueError("нулевой элемент на главной диагонали, строка %d" % i)
        beta[i] = b[i] / A[i][i]
        for j in range(n):
            if j != i:
                alpha[i][j] = -A[i][j] / A[i][i]
    return alpha, beta


def norm_matrix(M):
    r = 0.0
    for row in M:
        s = 0.0
        for v in row:
            s += abs(v)
        if s > r:
            r = s
    return r


def norm_upper(M):
    n = len(M)
    r = 0.0
    for i in range(n):
        s = 0.0
        for j in range(i + 1, n):
            s += abs(M[i][j])
        if s > r:
            r = s
    return r


def norm_vector(v):
    r = 0.0
    for t in v:
        if abs(t) > r:
            r = abs(t)
    return r


def diagonally_dominant(A):
    n = len(A)
    for i in range(n):
        s = 0.0
        for j in range(n):
            if j != i:
                s += abs(A[i][j])
        if abs(A[i][i]) <= s:
            return False
    return True


def apriori_estimate(q, beta, eps):
    if q <= 0.0 or q >= 1.0:
        return None
    nb = norm_vector(beta)
    if nb == 0.0:
        return 0
    value = (math.log10(eps) - math.log10(nb) + math.log10(1.0 - q)) / math.log10(q)
    return int(math.ceil(value - 1.0)) + 1


def simple_iteration(alpha, beta, q, eps, max_iter=10000):
    n = len(beta)
    x = beta[:]
    coef = q / (1.0 - q) if q < 1.0 else 1.0
    for k in range(1, max_iter + 1):
        xn = [0.0] * n
        for i in range(n):
            s = beta[i]
            for j in range(n):
                s += alpha[i][j] * x[j]
            xn[i] = s
        shift = 0.0
        for i in range(n):
            if abs(xn[i] - x[i]) > shift:
                shift = abs(xn[i] - x[i])
        x = xn
        if coef * shift <= eps:
            return x, k, coef * shift
    raise ValueError("метод простой итерации не сошёлся за %d итераций" % max_iter)


def seidel(alpha, beta, q, qc, eps, max_iter=10000):
    n = len(beta)
    x = beta[:]
    coef = qc / (1.0 - q) if q < 1.0 else 1.0
    for k in range(1, max_iter + 1):
        shift = 0.0
        for i in range(n):
            s = beta[i]
            for j in range(n):
                s += alpha[i][j] * x[j]
            if abs(s - x[i]) > shift:
                shift = abs(s - x[i])
            x[i] = s
        if coef * shift <= eps:
            return x[:], k, coef * shift
    raise ValueError("метод Зейделя не сошёлся за %d итераций" % max_iter)


def lu_reference(A, b):
    n = len(A)
    B = [row[:] for row in A]
    perm = list(range(n))
    tiny = 1e-12
    for k in range(n - 1):
        p = k
        for i in range(k + 1, n):
            if abs(B[i][k]) > abs(B[p][k]):
                p = i
        if p != k:
            B[k], B[p] = B[p], B[k]
            perm[k], perm[p] = perm[p], perm[k]
        if abs(B[k][k]) < tiny:
            raise ValueError("матрица вырождена")
        for i in range(k + 1, n):
            c = B[i][k] / B[k][k]
            B[i][k] = c
            for j in range(k + 1, n):
                B[i][j] -= c * B[k][j]
    if abs(B[n-1][n-1]) < tiny:
        raise ValueError("матрица вырождена")
    pb = [b[perm[i]] for i in range(n)]
    y = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += B[i][j] * y[j]
        y[i] = pb[i] - s
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += B[i][j] * x[j]
        x[i] = (y[i] - s) / B[i][i]
    return x


def max_error(x, exact):
    r = 0.0
    for i in range(len(x)):
        if abs(x[i] - exact[i]) > r:
            r = abs(x[i] - exact[i])
    return r


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "input3.txt"
    eps = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01

    A, b = read_system(filename)
    n = len(A)

    print("Файл данных:", filename)
    print("Размерность системы: %d" % n)
    print("Заданная точность eps = %g" % eps)

    alpha, beta = build_alpha_beta(A, b)
    q = norm_matrix(alpha)
    qc = norm_upper(alpha)

    print()
    print("Приведение к виду x = alpha*x + beta:")
    print("   диагональное преобладание: %s" % ("есть" if diagonally_dominant(A) else "нет"))
    print("   ||alpha|| = %.6f" % q)
    print("   ||C|| (верхний треугольник alpha) = %.6f" % qc)
    if q >= 1.0:
        print("   достаточное условие сходимости не выполнено")
    else:
        print("   достаточное условие сходимости ||alpha|| < 1 выполнено")
        est = apriori_estimate(q, beta, eps)
        if est is not None:
            print("   априорная оценка числа итераций: не менее %d" % est)

    exact = lu_reference(A, b)
    print()
    print("Эталонное решение (LU-разложение с выбором ведущего элемента):")
    for i in range(n):
        print("   x[%d] = %14.9f" % (i, exact[i]))

    x1, k1, e1 = simple_iteration(alpha, beta, q, eps)
    x2, k2, e2 = seidel(alpha, beta, q, qc, eps)

    print()
    print("Метод простой итерации")
    print("   итераций: %d" % k1)
    print("   оценка погрешности на последнем шаге: %.6f" % e1)
    for i in range(n):
        print("   x[%d] = %14.9f" % (i, x1[i]))
    print("   фактическое отклонение от эталона: %.3e" % max_error(x1, exact))

    print()
    print("Метод Зейделя")
    print("   итераций: %d" % k2)
    print("   оценка погрешности на последнем шаге: %.6f" % e2)
    for i in range(n):
        print("   x[%d] = %14.9f" % (i, x2[i]))
    print("   фактическое отклонение от эталона: %.3e" % max_error(x2, exact))

    print()
    print("Сравнение по скорости сходимости")
    print("   простая итерация: %d итераций" % k1)
    print("   Зейдель:          %d итераций" % k2)
    if k2 < k1:
        print("   Зейдель быстрее в %.2f раза" % (k1 / k2))
    elif k1 < k2:
        print("   простая итерация быстрее в %.2f раза" % (k2 / k1))
    else:
        print("   методы сошлись за одинаковое число итераций")


try:
    main()
except FileNotFoundError:
    print("Ошибка: файл не найден")
except ValueError as e:
    print("Ошибка:", e)