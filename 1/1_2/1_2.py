import sys


def read_tridiag(filename):
    with open(filename) as f:
        data = f.read().split()
    n = int(data[0])
    if len(data) != 1 + 4*n:
        raise ValueError("неверный формат файла")
    a    = [float(data[1 + i]) for i in range(n)]
    diag = [float(data[1 + n + i]) for i in range(n)]
    c    = [float(data[1 + 2*n + i]) for i in range(n)]
    d    = [float(data[1 + 3*n + i]) for i in range(n)]
    return a, diag, c, d


def thomas(a, diag, c, d):
    n = len(d)
    eps = 1e-12
    b = [-diag[i] for i in range(n)]

    P = [0.0] * (n + 1)
    Q = [0.0] * (n + 1)

    for i in range(n):
        den = b[i] - a[i]*P[i]
        if abs(den) < eps:
            raise ValueError(f"метод неприменим: нулевой знаменатель на шаге {i}")
        P[i+1] = c[i] / den
        Q[i+1] = (a[i]*Q[i] - d[i]) / den

    x = [0.0] * n
    x[n-1] = Q[n]
    for i in range(n - 2, -1, -1):
        x[i] = P[i+1] * x[i+1] + Q[i+1]

    return x, P, Q


def check_stability(a, diag, c, P):
    n = len(diag)
    for i in range(n):
        if abs(diag[i]) < abs(a[i]) + abs(c[i]):
            print(f"  предупреждение: нет диагонального преобладания в строке {i}")
    for i in range(1, n + 1):
        if abs(P[i]) > 1.0:
            print(f"  предупреждение: |P[{i}]| = {abs(P[i]):.4f} > 1, метод неустойчив")


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "input2.txt"
    print(f"Файл данных: {filename}")
    try:
        a, diag, c, d = read_tridiag(filename)
        x, P, Q = thomas(a, diag, c, d)

        print("\nПрогоночные коэффициенты:")
        for i in range(1, len(P)):
            print(f"   P[{i}] = {P[i]:12.6f}    Q[{i}] = {Q[i]:12.6f}")

        print("\nПроверка условий:")
        check_stability(a, diag, c, P)

        print("\nРешение системы:")
        for i in range(len(x)):
            print(f"   x[{i}] = {x[i]:12.6f}")
    except FileNotFoundError:
        print(f"Ошибка: файл '{filename}' не найден")
    except ValueError as e:
        print(f"Ошибка: {e}")


main()