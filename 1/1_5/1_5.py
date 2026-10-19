import sys
import math
import cmath
import numpy as np


def read_matrix(filename):
    with open(filename) as f:
        data = f.read().split()
    n = int(data[0])
    if len(data) != 1 + n * n:
        raise ValueError("неверный формат файла")
    return [[float(data[1 + i * n + j]) for j in range(n)] for i in range(n)]


def multiply(X, Y):
    n = len(X)
    C = [[0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for t in range(n):
                s += X[i][t] * Y[t][j]
            C[i][j] = s
    return C


def print_matrix(M):
    for row in M:
        line = ""
        for v in row:
            line += "%12.6f" % v
        print("   " + line)


def qr_decomposition(A):
    n = len(A)
    R = [row[:] for row in A]
    Q = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for k in range(n - 1):
        b = [R[i][k] for i in range(k, n)]

        norm = math.sqrt(sum(val ** 2 for val in b))
        sign = 1 if b[0] >= 0 else -1

        v = [0.0 for i in range(n)]

        for i in range(len(b)):
            v[k + i] = b[i]

        v[k] += sign * norm

        vtv = sum(val ** 2 for val in v)

        if vtv < 1e-300:
            continue

        H = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

        for i in range(n):
            for j in range(n):
                H[i][j] -= 2 * v[i] * v[j] / vtv

        R = multiply(H, R)
        Q = multiply(Q, H)

    for i in range(n):
        for j in range(i):
            R[i][j] = 0.0

    return Q, R


def block_values(A, i):
    a = A[i][i]
    b = A[i][i + 1]
    c = A[i + 1][i]
    d = A[i + 1][i + 1]

    D = (a + d) ** 2 - 4 * (a * d - b * c)

    l1 = ((a + d) + cmath.sqrt(D)) / 2
    l2 = ((a + d) - cmath.sqrt(D)) / 2

    return l1, l2


def qr_eigenvalues(A, eps, max_iter=1000):
    n = len(A)
    Ak = [row[:] for row in A]
    old_vals = None

    for k in range(1, max_iter + 1):
        Q, R = qr_decomposition(Ak)
        Ak = multiply(R, Q)

        vals = []
        i = 0
        ready = True

        while i < n:
            s = math.sqrt(sum(Ak[j][i] ** 2 for j in range(i + 1, n)))

            if s < eps:
                vals.append(complex(Ak[i][i]))
                i += 1
            else:
                l1, l2 = block_values(Ak, i)
                vals.append(l1)
                vals.append(l2)
                i += 2

        if old_vals is None:
            ready = False
        else:
            for i in range(n):
                if abs(vals[i] - old_vals[i]) > eps:
                    ready = False
                    break

        if ready:
            return vals, k, Ak

        old_vals = vals

    raise ValueError("QR-алгоритм не сошёлся за %d итераций" % max_iter)


def print_values(vals, eps):
    for i in range(len(vals)):
        val = vals[i]
        if abs(val.imag) < eps:
            print("   lambda%d = %.6f" % (i + 1, val.real))
        else:
            print("   lambda%d = %.6f %+.6fi" % (i + 1, val.real, val.imag))


filename = sys.argv[1] if len(sys.argv) > 1 else "qr.txt"
eps = float(sys.argv[2]) if len(sys.argv) > 2 else 0.001

A = read_matrix(filename)
n = len(A)

Q, R = qr_decomposition(A)
QR = multiply(Q, R)
vals, k, Ak = qr_eigenvalues(A, eps)

print("Файл данных:", filename)
print("Размерность матрицы: %d" % n)
print("Заданная точность eps = %g" % eps)

print("\nMatrix A:")
print_matrix(A)

print("\nMatrix Q:")
print_matrix(Q)

print("\nMatrix R:")
print_matrix(R)

print("\nQ * R:")
print_matrix(QR)

print("\nQR-алгоритм сошёлся за %d итераций" % k)

print("\nMatrix A^(k):")
print_matrix(Ak)

print("\nEigenvalues:")
print_values(vals, eps)


complex_A = [[1.0, -2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 3.0]]
complex_vals, ck, cAk = qr_eigenvalues(complex_A, eps)

print("\nMatrix with complex conjugate eigenvalues:")
print_matrix(complex_A)

print("\nEigenvalues (наш QR-алгоритм, %d итераций):" % ck)
print_values(complex_vals, eps)

print("\nComplex eigenvalues (NumPy check):")
for val in np.linalg.eigvals(np.array(complex_A)):
    print("   ", val)