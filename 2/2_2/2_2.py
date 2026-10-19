import sys
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A_PARAM = 2.0
LN10 = math.log(10.0)


def f1(x1, x2):
    return x1 - math.cos(x2) - 1.0


def f2(x1, x2):
    return x2 - math.log10(x1 + 1.0) - A_PARAM


def F(x):
    return [f1(x[0], x[1]), f2(x[0], x[1])]


def jacobian(x):
    x1, x2 = x
    return [[1.0, math.sin(x2)],
            [-1.0 / ((x1 + 1.0) * LN10), 1.0]]


def phi1(x1, x2):
    return 1.0 + math.cos(x2)


def phi2(x1, x2):
    return A_PARAM + math.log10(x1 + 1.0)


def PHI(x):
    return [phi1(x[0], x[1]), phi2(x[0], x[1])]


def phi_jacobian(x):
    x1, x2 = x
    return [[0.0, -math.sin(x2)],
            [1.0 / ((x1 + 1.0) * LN10), 0.0]]


def read_params(filename):
    with open(filename) as fin:
        data = fin.read().split()
    if len(data) != 6:
        raise ValueError("нужно 6 чисел: a g1 g2 g3 g4 eps")
    a = float(data[0])
    g = [float(data[i]) for i in range(1, 5)]
    eps = float(data[5])
    if eps <= 0.0 or g[0] >= g[1] or g[2] >= g[3]:
        raise ValueError("некорректные параметры")
    return a, g, eps


def norm_vec(v):
    r = 0.0
    for t in v:
        if abs(t) > r:
            r = abs(t)
    return r


def norm_mat(M):
    r = 0.0
    for row in M:
        s = 0.0
        for v in row:
            s += abs(v)
        if s > r:
            r = s
    return r


def det2(M):
    return M[0][0] * M[1][1] - M[0][1] * M[1][0]


def max_phi_norm(g, parts=60):
    lo1, hi1, lo2, hi2 = g
    q = 0.0
    for i in range(parts + 1):
        for j in range(parts + 1):
            x1 = lo1 + (hi1 - lo1) * i / parts
            x2 = lo2 + (hi2 - lo2) * j / parts
            n = norm_mat(phi_jacobian([x1, x2]))
            if n > q:
                q = n
    return q


def phi_maps_into(g, parts=60):
    lo1, hi1, lo2, hi2 = g
    for i in range(parts + 1):
        for j in range(parts + 1):
            x1 = lo1 + (hi1 - lo1) * i / parts
            x2 = lo2 + (hi2 - lo2) * j / parts
            y = PHI([x1, x2])
            if y[0] < lo1 - 1e-12 or y[0] > hi1 + 1e-12:
                return False
            if y[1] < lo2 - 1e-12 or y[1] > hi2 + 1e-12:
                return False
    return True


def simple_iteration(x0, g, q, eps, max_iter=10000):
    x = x0[:]
    coef = q / (1.0 - q)
    history = []
    for k in range(1, max_iter + 1):
        xn = PHI(x)
        d = norm_vec([xn[0] - x[0], xn[1] - x[1]])
        est = coef * d
        history.append((k, xn[:], d, est, norm_vec(F(xn))))
        x = xn
        if est <= eps:
            return x, k, history
    raise ValueError("метод простой итерации не сошёлся")


def newton(x0, eps, max_iter=1000):
    x = x0[:]
    history = []
    for k in range(1, max_iter + 1):
        fv = F(x)
        J = jacobian(x)
        dJ = det2(J)
        if abs(dJ) < 1e-14:
            raise ValueError("матрица Якоби вырождена, метод Ньютона неприменим")
        A1 = [[fv[0], J[0][1]],
              [fv[1], J[1][1]]]
        A2 = [[J[0][0], fv[0]],
              [J[1][0], fv[1]]]
        d1 = det2(A1)
        d2 = det2(A2)
        xn = [x[0] - d1 / dJ, x[1] - d2 / dJ]
        d = norm_vec([xn[0] - x[0], xn[1] - x[1]])
        history.append((k, xn[:], d, norm_vec(F(xn)), d1, d2, dJ))
        x = xn
        if d <= eps:
            return x, k, history
    raise ValueError("метод Ньютона не сошёлся")


def plot_png(g, sol, filename="graph.png"):
    lo1, hi1, lo2, hi2 = g
    xs2 = [0.0 + 4.0 * i / 800 for i in range(801)]
    c1x = [1.0 + math.cos(t) for t in xs2]
    xs1 = [-0.9 + 3.9 * i / 800 for i in range(801)]
    c2y = [A_PARAM + math.log10(t + 1.0) for t in xs1]

    plt.figure(figsize=(9, 6))
    plt.plot(c1x, xs2, label="f1 = 0:  x1 = 1 + cos(x2)")
    plt.plot(xs1, c2y, label="f2 = 0:  x2 = 2 + lg(x1 + 1)")
    plt.plot([sol[0]], [sol[1]], "ko", markersize=7, label="решение")
    plt.axvspan(lo1, hi1, color="orange", alpha=0.2)
    plt.axhspan(lo2, hi2, color="orange", alpha=0.2)
    plt.axhline(0, color="black", linewidth=0.6)
    plt.axvline(0, color="black", linewidth=0.6)
    plt.xlim(-1.0, 3.0)
    plt.ylim(0.0, 4.0)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Графическое отделение решения системы")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()


def plot_convergence(h1, h2, sol, filename="convergence.png"):
    k1 = [h[0] for h in h1]
    e1 = [max(norm_vec([h[1][0] - sol[0], h[1][1] - sol[1]]), 1e-18) for h in h1]
    k2 = [h[0] for h in h2]
    e2 = [max(norm_vec([h[1][0] - sol[0], h[1][1] - sol[1]]), 1e-18) for h in h2]
    plt.figure(figsize=(9, 5))
    plt.semilogy(k1, e1, "o-", label="простая итерация")
    plt.semilogy(k2, e2, "s-", label="метод Ньютона")
    plt.xlabel("номер итерации k")
    plt.ylabel("погрешность ||x_k - x*||")
    plt.title("Зависимость погрешности от числа итераций")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()


def main():
    global A_PARAM
    filename = sys.argv[1] if len(sys.argv) > 1 else "input.txt"
    A_PARAM, g, eps = read_params(filename)

    print("Система нелинейных уравнений, вариант 5, a = %g" % A_PARAM)
    print("   f1(x1, x2) = x1 - cos(x2) - 1 = 0")
    print("   f2(x1, x2) = x2 - lg(x1 + 1) - %g = 0" % A_PARAM)
    print()
    print("Область G: x1 в [%g, %g], x2 в [%g, %g]" % (g[0], g[1], g[2], g[3]))
    print("Заданная точность eps = %g" % eps)
    print()

    x0 = [(g[0] + g[1]) / 2.0, (g[2] + g[3]) / 2.0]
    print("Начальное приближение (центр области): x1 = %.4f, x2 = %.4f" % (x0[0], x0[1]))
    print()

    print("Приведение к виду x = phi(x) (формула 2.21):")
    print("   x1 = phi1 = 1 + cos(x2)")
    print("   x2 = phi2 = %g + lg(x1 + 1)" % A_PARAM)
    print()

    q = max_phi_norm(g)
    print("Проверка теоремы 2.4, условие (2.25):")
    print("   max ||phi'(x)|| по области G = %.6f" % q)
    if q >= 1.0:
        raise ValueError("условие сходимости не выполнено: q = %.4f" % q)
    print("   q = %.6f < 1  -- условие выполнено" % q)
    print("   phi(G) не выходит за G: %s" % ("да" if phi_maps_into(g) else "НЕТ"))
    print()

    x2n, k2, h2 = newton(x0, eps)
    x1n, k1, h1 = simple_iteration(x0, g, q, eps)
    sol = x2n

    print("МЕТОД НЬЮТОНА (формулы 2.13, 2.14, 2.20)")
    print("   матрица Якоби:")
    print("      df1/dx1 = 1                      df1/dx2 = sin(x2)")
    print("      df2/dx1 = -1/((x1+1)*ln10)       df2/dx2 = 1")
    print("   итераций: %d" % k2)
    print("   x1 = %.12f" % x2n[0])
    print("   x2 = %.12f" % x2n[1])
    print("   невязка ||F(x)|| = %.3e" % norm_vec(F(x2n)))
    print()
    print("   k |      x1       |      x2       |    det A1    |    det A2    |    det J     |   ||dx||")
    for (k, xk, d, nf, d1, d2, dJ) in h2:
        print("  %2d | %13.9f | %13.9f | %12.3e | %12.3e | %12.6f | %10.3e"
              % (k, xk[0], xk[1], d1, d2, dJ, d))
    print()

    print("МЕТОД ПРОСТОЙ ИТЕРАЦИИ (формулы 2.23)")
    print("   итераций: %d" % k1)
    print("   x1 = %.12f" % x1n[0])
    print("   x2 = %.12f" % x1n[1])
    print("   невязка ||F(x)|| = %.3e" % norm_vec(F(x1n)))
    print()
    print("   k |      x1       |      x2       |    ||dx||    |  оценка (2.26) |  ||F(x)||")
    for (k, xk, d, est, nf) in h1:
        print("  %2d | %13.9f | %13.9f | %12.3e | %14.3e | %10.3e"
              % (k, xk[0], xk[1], d, est, nf))
    print()

    print("ЗАВИСИМОСТЬ ПОГРЕШНОСТИ ОТ ЧИСЛА ИТЕРАЦИЙ")
    print("   за точное решение принят результат метода Ньютона")
    print()
    print("   k |  простая итерация  | отношение |      Ньютон        | отношение")
    p1 = None
    p2 = None
    for k in range(1, max(len(h1), len(h2)) + 1):
        e1 = norm_vec([h1[k-1][1][0] - sol[0], h1[k-1][1][1] - sol[1]]) if k <= len(h1) else None
        e2 = norm_vec([h2[k-1][1][0] - sol[0], h2[k-1][1][1] - sol[1]]) if k <= len(h2) else None
        s1 = "%18.6e" % e1 if e1 is not None else " " * 18
        s2 = "%18.6e" % e2 if e2 is not None else " " * 18
        r1 = "%9.4f" % (e1 / p1) if (e1 is not None and p1 not in (None, 0.0)) else " " * 9
        r2 = "%9.4f" % (e2 / p2) if (e2 is not None and p2 not in (None, 0.0)) else " " * 9
        print("  %2d |%s |%s |%s |%s" % (k, s1, r1, s2, r2))
        if e1 is not None:
            p1 = e1
        if e2 is not None:
            p2 = e2
    print()
    print("   простая итерация: отношение примерно постоянно -- линейная сходимость")
    print("   Ньютон: отношение быстро убывает -- квадратичная сходимость")
    print()

    plot_png(g, sol)
    plot_convergence(h1, h2, sol)
    print("Графики сохранены: graph.png, convergence.png")


try:
    main()
except FileNotFoundError:
    print("Ошибка: файл не найден")
except ValueError as e:
    print("Ошибка:", e)