import sys
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def f(x):
    return math.cos(x) + 0.25 * x - 0.5


def df(x):
    return -math.sin(x) + 0.25


def d2f(x):
    return -math.cos(x)


def read_params(filename):
    with open(filename) as f_in:
        data = f_in.read().split()
    if len(data) != 4:
        raise ValueError("нужно 4 числа: a b step eps")
    a = float(data[0])
    b = float(data[1])
    step = float(data[2])
    eps = float(data[3])
    if step <= 0.0 or eps <= 0.0 or a >= b:
        raise ValueError("некорректные параметры")
    return a, b, step, eps


def tabulate(a, b, step):
    xs = []
    x = a
    while x <= b + 1e-12:
        xs.append(x)
        x += step
    return xs


def separate_roots(a, b, step):
    xs = tabulate(a, b, step)
    segments = []
    for i in range(len(xs) - 1):
        if f(xs[i]) * f(xs[i+1]) < 0.0:
            segments.append((xs[i], xs[i+1]))
    return segments


def monotone_check(seg, parts=1000):
    a, b = seg
    pos = 0
    neg = 0
    for i in range(parts + 1):
        x = a + (b - a) * i / parts
        if df(x) > 0.0:
            pos += 1
        elif df(x) < 0.0:
            neg += 1
    return pos == 0 or neg == 0


def phi_maps_into(seg, M, parts=1000):
    a, b = seg
    for i in range(parts + 1):
        x = a + (b - a) * i / parts
        y = x - f(x) / M
        if y < a - 1e-12 or y > b + 1e-12:
            return False
    return True


def plot_png(a, b, segments, filename="graph.png"):
    xs = [a + (b - a) * i / 1000 for i in range(1001)]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    ax[0].plot(xs, [math.cos(x) for x in xs], label="f1(x) = cos(x)")
    ax[0].plot(xs, [0.5 - 0.25 * x for x in xs], label="f2(x) = 0.5 - 0.25x")
    for (l, r) in segments:
        ax[0].axvspan(l, r, color="orange", alpha=0.3)
    ax[0].axhline(0, color="black", linewidth=0.8)
    ax[0].set_title("Отделение корня: cos(x) = 0.5 - 0.25x")
    ax[0].set_xlabel("x")
    ax[0].grid(True, linestyle=":")
    ax[0].legend()

    ax[1].plot(xs, [f(x) for x in xs], color="crimson",
               label="f(x) = cos(x) + 0.25x - 0.5")
    for (l, r) in segments:
        ax[1].axvspan(l, r, color="orange", alpha=0.3, label="отрезок с корнем")
    ax[1].axhline(0, color="black", linewidth=0.8)
    ax[1].set_title("График f(x)")
    ax[1].set_xlabel("x")
    ax[1].grid(True, linestyle=":")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()


def start_point(seg):
    a, b = seg
    return a


def lipschitz_M(a, b, parts=1000):
    best = 0.0
    for i in range(parts + 1):
        x = a + (b - a) * i / parts
        d = df(x)
        if abs(d) > abs(best):
            best = d
    return best


def simple_iteration(seg, eps, max_iter=10000):
    a, b = seg
    M = lipschitz_M(a, b)
    q = 0.0
    for i in range(1001):
        x = a + (b - a) * i / 1000.0
        g = abs(1.0 - df(x) / M)
        if g > q:
            q = g
    if q >= 1.0:
        raise ValueError("условие сходимости не выполнено: q = %.4f" % q)
    inside = phi_maps_into(seg, M)

    x = start_point(seg)
    history = []
    for k in range(1, max_iter + 1):
        xn = x - f(x) / M
        shift = abs(xn - x)
        est = q / (1.0 - q) * shift
        history.append((k, xn, shift, est, abs(f(xn))))
        x = xn
        if est <= eps:
            return x, k, history, q, M, inside
    raise ValueError("метод простой итерации не сошёлся")


def newton(seg, eps, max_iter=1000):
    a, b = seg
    if f(a) * d2f(a) > 0.0:
        x = a
    elif f(b) * d2f(b) > 0.0:
        x = b
    else:
        raise ValueError("условие теоремы 2.2 не выполнено ни на одном конце")
    history = []
    for k in range(1, max_iter + 1):
        d = df(x)
        if abs(d) < 1e-14:
            raise ValueError("производная близка к нулю, метод Ньютона неприменим")
        xn = x - f(x) / d
        shift = abs(xn - x)
        history.append((k, xn, shift, abs(f(xn))))
        x = xn
        if shift <= eps:
            return x, k, history
    raise ValueError("метод Ньютона не сошёлся")


def plot_convergence(h1, h2, exact, filename="convergence.png"):
    k1 = [h[0] for h in h1]
    e1 = [max(abs(h[1] - exact), 1e-18) for h in h1]
    k2 = [h[0] for h in h2]
    e2 = [max(abs(h[1] - exact), 1e-18) for h in h2]
    plt.figure(figsize=(9, 5))
    plt.semilogy(k1, e1, "o-", label="простая итерация")
    plt.semilogy(k2, e2, "s-", label="метод Ньютона")
    plt.xlabel("номер итерации k")
    plt.ylabel("погрешность |x_k - x*|")
    plt.title("Зависимость погрешности от числа итераций")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "input.txt"
    a, b, step, eps = read_params(filename)

    print("Уравнение:  cos(x) + 0.25*x - 0.5 = 0")
    print("Отрезок поиска: [%g, %g], шаг табуляции %g" % (a, b, step))
    print("Заданная точность eps = %g" % eps)
    print()

    print("Таблица значений f(x):")
    xs = tabulate(a, b, 0.25)
    for x in xs:
        print("   f(%6.2f) = %10.6f" % (x, f(x)))
    print()

    segments = separate_roots(a, b, step)
    if len(segments) == 0:
        raise ValueError("на отрезке не найдено перемен знака")
    print("Отрезки с переменой знака:")
    for (l, r) in segments:
        print("   [%.4f, %.4f]:  f(%.4f) = %+.6f,  f(%.4f) = %+.6f"
              % (l, r, l, f(l), r, f(r)))
    print()

    plot_png(a, b, segments)
    print("График отделения корня сохранён в graph.png")
    print()

    positive = []
    for (l, r) in segments:
        if r > 0.0:
            positive.append((l, r))
    if len(positive) == 0:
        raise ValueError("положительный корень не найден")
    seg = positive[0]
    print("Берём положительный корень на [%.4f, %.4f]" % seg)
    print()
    print("Проверка теоремы 2.1 (единственность корня):")
    print("   f(a)*f(b) = %+.6f < 0  -- корень на отрезке есть" % (f(seg[0]) * f(seg[1])))
    if monotone_check(seg):
        print("   f'(x) сохраняет знак на отрезке: f'(a) = %+.6f, f'(b) = %+.6f"
              % (df(seg[0]), df(seg[1])))
        print("   функция строго монотонна -- корень единственный")
    else:
        print("   ВНИМАНИЕ: f'(x) меняет знак, монотонность не гарантирована")
    print()

    x1, k1, h1, q, M, inside = simple_iteration(seg, eps)
    x2, k2, h2 = newton(seg, eps)

    print("МЕТОД ПРОСТОЙ ИТЕРАЦИИ")
    print("   вид:  x = x - f(x)/M,  M = %.6f" % M)
    print("   (замечание на стр. 46: x = x - lambda*f(x), lambda = 1/M)")
    print("   проверка теоремы 2.3:")
    print("      max|phi'(x)| = q = %.6f  < 1  -- выполнено" % q)
    print("      phi(x) не выходит за [a, b]: %s" % ("да" if inside else "НЕТ"))
    d0 = abs(h1[0][1] - start_point(seg))
    if q > 0.0:
        need = math.log(eps * (1.0 - q) / d0) / math.log(q) - 1.0
        print("      априорная оценка (2.7): нужно не менее %d итераций"
              % max(1, int(math.ceil(need))))
    print("   итераций: %d" % k1)
    print("   корень x = %.12f" % x1)
    print("   невязка |f(x)| = %.3e" % abs(f(x1)))
    print()
    print("   k |        x_k        |   |x_k - x_k-1|  |   оценка погр.  |     |f(x_k)|")
    for (k, xk, sh, est, fx) in h1:
        print("  %3d | %17.12f | %14.6e | %14.6e | %14.6e" % (k, xk, sh, est, fx))
    print()

    print("МЕТОД НЬЮТОНА")
    print("   вид:  x = x - f(x)/f'(x)   (формула 2.2)")
    print("   выбор x0 по теореме 2.2, условие f(x0)*f''(x0) > 0:")
    print("      f(a)*f''(a) = %+.6f,  f(b)*f''(b) = %+.6f"
          % (f(seg[0]) * d2f(seg[0]), f(seg[1]) * d2f(seg[1])))
    print("      взят конец x0 = %.4f" % (seg[0] if f(seg[0]) * d2f(seg[0]) > 0 else seg[1]))
    print("   итераций: %d" % k2)
    print("   корень x = %.12f" % x2)
    print("   невязка |f(x)| = %.3e" % abs(f(x2)))
    print()
    print("   k |        x_k        |   |x_k - x_k-1|  |     |f(x_k)|     | верных знаков")
    for (k, xk, sh, fx) in h2:
        digits = 0 if sh <= 0 else max(0, int(-math.log10(sh)))
        print("  %3d | %17.12f | %14.6e | %14.6e | %6d" % (k, xk, sh, fx, digits))
    print()

    print("ЗАВИСИМОСТЬ ПОГРЕШНОСТИ ОТ ЧИСЛА ИТЕРАЦИЙ")
    exact = x2
    print("   за точное значение принят результат метода Ньютона")
    print()
    print("   k |  простая итерация  | отношение |      Ньютон       | отношение")
    prev1 = None
    prev2 = None
    for k in range(1, max(len(h1), len(h2)) + 1):
        e1 = abs(h1[k-1][1] - exact) if k <= len(h1) else None
        e2 = abs(h2[k-1][1] - exact) if k <= len(h2) else None
        s1 = "%18.6e" % e1 if e1 is not None else " " * 18
        s2 = "%18.6e" % e2 if e2 is not None else " " * 18
        r1 = "%9.4f" % (e1 / prev1) if (e1 is not None and prev1 not in (None, 0.0)) else " " * 9
        r2 = "%9.4f" % (e2 / prev2) if (e2 is not None and prev2 not in (None, 0.0)) else " " * 9
        print("  %3d |%s |%s |%s |%s" % (k, s1, r1, s2, r2))
        if e1 is not None:
            prev1 = e1
        if e2 is not None:
            prev2 = e2
    print()
    print("   простая итерация: отношение погрешностей примерно постоянно")
    print("   (линейная сходимость, первый порядок)")
    print("   Ньютон: отношение быстро убывает, число верных знаков удваивается")
    print("   (квадратичная сходимость, второй порядок)")
    print()

    plot_convergence(h1, h2, exact)
    print("График сходимости сохранён в convergence.png")


try:
    main()
except FileNotFoundError:
    print("Ошибка: файл не найден")
except ValueError as e:
    print("Ошибка:", e)