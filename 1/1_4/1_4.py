import sys
import math


def read_matrix(filename):
    with open(filename) as f:
        data = f.read().split()
    n = int(data[0])
    if len(data) != 1 + n*n:
        raise ValueError("неверный формат файла")
    return [[float(data[1 + i*n + j]) for j in range(n)] for i in range(n)]


def is_symmetric(A, eps=1e-9):
    n = len(A)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j] - A[j][i]) > eps:
                return False
    return True


def identity(n):
    E = [[0.0] * n for _ in range(n)]
    for i in range(n):
        E[i][i] = 1.0
    return E


def matmul(X, Y):
    n = len(X)
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for t in range(n):
                s += X[i][t] * Y[t][j]
            C[i][j] = s
    return C


def transpose(M):
    n = len(M)
    T = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            T[i][j] = M[j][i]
    return T


def jacobi(A, eps):
    n = len(A)
    if not is_symmetric(A):
        raise ValueError("матрица не симметрична, метод вращений неприменим")

    A = [row[:] for row in A]
    V = identity(n)
    steps = []
    k = 0

    while True:
        max_val = 0.0
        sum_sq = 0.0
        p = 0
        q = 1
        for i in range(n):
            for j in range(i + 1, n):
                sum_sq += A[i][j] * A[i][j]
                if abs(A[i][j]) > max_val:
                    max_val = abs(A[i][j])
                    p = i
                    q = j

        t = math.sqrt(sum_sq)
        if t < eps:
            break

        if A[p][p] == A[q][q]:
            phi = math.pi / 4.0
        else:
            phi = 0.5 * math.atan(2.0 * A[p][q] / (A[p][p] - A[q][q]))

        c = math.cos(phi)
        s = math.sin(phi)

        U = identity(n)
        U[p][p] = c
        U[q][q] = c
        U[p][q] = -s
        U[q][p] = s

        A = matmul(matmul(transpose(U), A), U)
        V = matmul(V, U)

        steps.append((k, p, q, phi, t))
        k += 1

    vals = [A[i][i] for i in range(n)]
    return vals, V, k, steps


def lambda_matrix(values):
    n = len(values)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = values[i]
    return L


def max_diff(X, Y):
    n = len(X)
    d = 0.0
    for i in range(n):
        for j in range(n):
            if abs(X[i][j] - Y[i][j]) > d:
                d = abs(X[i][j] - Y[i][j])
    return d


def show(name, M):
    print(name)
    for row in M:
        line = ""
        for v in row:
            line += "%12.6f" % v
        print("   " + line)
    print()


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "input4.txt"
    eps = float(sys.argv[2]) if len(sys.argv) > 2 else 0.001

    A = read_matrix(filename)
    n = len(A)

    print("Файл данных:", filename)
    print("Размерность матрицы: %d" % n)
    print("Заданная точность eps = %g" % eps)
    print()

    show("Исходная матрица A:", A)

    vals, V, k, steps = jacobi(A, eps)

    print("Ход итераций:")
    for (num, p, q, phi, t) in steps:
        print("   шаг %d: обнуляем a[%d][%d], phi = %9.6f, t(A) = %10.6f"
              % (num, p, q, phi, t))
    print("   всего итераций: %d" % k)
    print()

    print("Собственные значения:")
    for i in range(n):
        print("   lambda[%d] = %14.9f" % (i, vals[i]))
    print()

    print("Собственные векторы (столбцы матрицы V):")
    for j in range(n):
        line = ""
        for i in range(n):
            line += "%12.6f" % V[i][j]
        print("   x%d = (%s )" % (j, line))
    print()

    show("Матрица собственных векторов V:", V)
    show("Диагональная матрица Lambda:", lambda_matrix(vals))

    AV = matmul(A, V)
    VL = matmul(V, lambda_matrix(vals))
    show("Проверка A*V:", AV)
    show("Проверка V*Lambda:", VL)
    print("Максимальное расхождение A*V и V*Lambda: %.3e" % max_diff(AV, VL))
    print()

    print("Ортогональность собственных векторов:")
    for a in range(n):
        for b in range(a + 1, n):
            s = 0.0
            for i in range(n):
                s += V[i][a] * V[i][b]
            print("   (x%d, x%d) = %12.3e" % (a, b, s))


try:
    main()
except FileNotFoundError:
    print("Ошибка: файл не найден")
except ValueError as e:
    print("Ошибка:", e)