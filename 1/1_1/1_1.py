import sys

def read_system(filename):
    with open(filename) as f:
        data = f.read().split()

    n = int(data[0])
    if len(data) != 1 + n*n + n:
        raise ValueError("неверный формат файла")

    A = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(float(data[1 + i*n + j]))
        A.append(row)

    b = []
    for k in range(n):
        b.append(float(data[1 + n*n + k]))

    return A, b

def copy_matrix(A):
    B = []
    for i in A:
        B.append(i[:])
    return B


def lu_decompose(A):
    B = copy_matrix(A)
    n = len(B)
    perm = list(range(n))
    swaps = 0
    eps = 1e-12

    for k in range(n - 1):
        p = k
        for i in range(k + 1, n):
            if abs(B[i][k]) > abs(B[p][k]):
                p = i
        if p != k:
            B[k], B[p] = B[p], B[k]
            perm[k], perm[p] = perm[p], perm[k]
            swaps += 1
        if abs(B[k][k]) < eps:
            raise ValueError("матрица вырождена")

        for i in range(k + 1, n):
            c = B[i][k] / B[k][k]
            B[i][k] = c
            for j in range(k + 1, n):
                B[i][j] -= c * B[k][j]

    if abs(B[n - 1][n - 1]) < eps:
        raise ValueError("матрица вырождена")

    return B, perm, swaps


def lu_solve(LU, perm, b):
    n = len(LU)

    pb = [b[perm[i]] for i in range(n)]

    y = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += LU[i][j] * y[j]
        y[i] = pb[i] - s

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += LU[i][j] * x[j]
        x[i] = (y[i] - s) / LU[i][i]

    return x


def lu_inverse(LU, perm):
    n = len(LU)
    inv = [[0.0] * n for _ in range(n)]
    for k in range(n):
        e = [0.0] * n
        e[k] = 1.0
        col = lu_solve(LU, perm, e)
        for i in range(n):
            inv[i][k] = col[i]
    return inv


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


def split_lu(LU):
    n = len(LU)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i > j:
                L[i][j] = LU[i][j]
            elif i == j:
                L[i][j] = 1.0
                U[i][j] = LU[i][j]
            else:
                U[i][j] = LU[i][j]
    return L, U


def perm_matrix(perm):
    n = len(perm)
    P = [[0.0] * n for _ in range(n)]
    for i in range(n):
        P[i][perm[i]] = 1.0
    return P


def lu_det(LU, swaps):
    det = (-1)**swaps
    n = len(LU)
    for i in range(n):
        det *= LU[i][i]
    return det


def max_diff(X, Y):
    """Максимальное по модулю расхождение между двумя матрицами."""
    n = len(X)
    d = 0.0
    for i in range(n):
        for j in range(n):
            current = abs(X[i][j] - Y[i][j])
            if current > d:
                d = current
    return d


def identity(n):
    """Единичная матрица n×n."""
    E = [[0.0] * n for _ in range(n)]
    for i in range(n):
        E[i][i] = 1.0
    return E


def show(name, M):
    print(name)
    for row in M:
        print("   " + "  ".join(f"{v:10.6f}" for v in row))
    print()


def main():
    eps = 1e-9

    filename = sys.argv[1] if len(sys.argv) > 1 else "input.txt"
    print(f"Файл данных: {filename}")


    try:
        A, b = read_system(filename)
        n = len(A)

        LU, perm, swaps = lu_decompose(A)
        L, U = split_lu(LU)
        P = perm_matrix(perm)
        inv = lu_inverse(LU, perm)
        x = lu_solve(LU, perm, b)
        det = lu_det(LU, swaps)

        show("Исходная матрица A:", A)
        show("L:", L)                                    # п.1
        show("U:", U)
        show("L*U:", matmul(L, U))                       # п.2

        print("Решение системы Ax = b:")                 # п.3
        for i in range(n):
            print(f"   x[{i}] = {x[i]:12.6f}")
        print()

        show("Обратная матрица A^-1:", inv)              # п.4
        print(f"det A = {det:.6f}")                      # п.5
        print(f"число перестановок строк = {swaps}")
        print()

        show("Проверка A*A^-1:", matmul(A, inv))         # п.6
        d1 = max_diff(matmul(A, inv), identity(n))
        print(f"   макс. отклонение от E: {d1:.2e}  -> {'OK' if d1 < eps else 'FAIL'}")
        print()

        show("Проверка L*U:", matmul(L, U))              # п.7
        show("Проверка P*A:", matmul(P, A))
        d2 = max_diff(matmul(L, U), matmul(P, A))
        print(f"   макс. расхождение L*U и P*A: {d2:.2e}  -> {'OK' if d2 < eps else 'FAIL'}")
    except FileNotFoundError:
        print(f"Ошибка: файл '{filename}' не найден")
    except ValueError as e:
        print(f"Ошибка: {e}")


main()