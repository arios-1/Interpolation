from bisect import bisect_right
from fractions import Fraction

MAX_POINTS = 10

# input
def read_number(prompt):
    while True:
        text = input(prompt).strip()
        try:
            return Fraction(text)
        except (ValueError, ZeroDivisionError):
            print("  Please enter a number, e.g. 2, -1.5, or 3/4.")


def get_inputs_from_user():
    while True:
        try:
            m = int(input(f"How many data points (2 to {MAX_POINTS})? "))
            if 2 <= m <= MAX_POINTS:
                break
        except ValueError:
            pass
        print(f"  Please enter a whole number from 2 to {MAX_POINTS}.")

    points = []
    for i in range(m):
        print(f"Point {i + 1}:")
        x = read_number("  x = ")
        y = read_number("  y = ")
        points.append((x, y))

    while True:
        kind = input("Boundary condition: (N)atural or (C)lamped? ").strip().lower()
        if kind in ("n", "natural", ""):
            return points, "natural", None, None
        if kind in ("c", "clamped"):
            print("Clamped needs the slope y' at both ends.")
            s0 = read_number("  y'(x0) = ")
            sn = read_number("  y'(xn) = ")
            return points, "clamped", s0, sn
        print("  Type N or C.")


# core algorithm
def prepare_points(points):
    points = sorted(points, key=lambda p: p[0])
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    for i in range(1, len(xs)):
        if xs[i] == xs[i - 1]:
            raise ValueError(f"Two points share x = {xs[i]}; x-values must be distinct "
                             "(otherwise some h_i = 0 and we would divide by zero).")
    return xs, ys


def build_system(xs, d, kind="natural", s0=None, sn=None):
    n = len(xs) - 1
    h = [xs[i + 1] - xs[i] for i in range(n)]
    size = n + 1
    lower = [Fraction(0)] * size
    diag = [Fraction(0)] * size
    upper = [Fraction(0)] * size
    t = [Fraction(0)] * size

    for i in range(1, n):
        lower[i] = h[i - 1]
        diag[i] = 2 * (h[i - 1] + h[i])
        upper[i] = h[i]
        t[i] = 3 / h[i] * (d[i + 1] - d[i]) - 3 / h[i - 1] * (d[i] - d[i - 1])

    if kind == "natural":
        diag[0], t[0] = Fraction(1), Fraction(0)
        diag[n], t[n] = Fraction(1), Fraction(0)
    else:  # clamped
        diag[0], upper[0] = 2 * h[0], h[0]
        t[0] = 3 * (d[1] - d[0]) / h[0] - 3 * s0
        lower[n], diag[n] = h[n - 1], 2 * h[n - 1]
        t[n] = 3 * sn - 3 * (d[n] - d[n - 1]) / h[n - 1]

    return h, lower, diag, upper, t


def solve_tridiagonal(lower, diag, upper, t):
    size = len(diag)
    up = [Fraction(0)] * size     # modified upper diagonal
    rhs = [Fraction(0)] * size    # modified right-hand side

    up[0] = upper[0] / diag[0]
    rhs[0] = t[0] / diag[0]
    for i in range(1, size):
        pivot = diag[i] - lower[i] * up[i - 1]
        up[i] = upper[i] / pivot if i < size - 1 else Fraction(0)
        rhs[i] = (t[i] - lower[i] * rhs[i - 1]) / pivot

    b = [Fraction(0)] * size
    b[-1] = rhs[-1]
    for i in range(size - 2, -1, -1):
        b[i] = rhs[i] - up[i] * b[i + 1]
    return b


def spline_coefficients(xs, ys, kind="natural", s0=None, sn=None):
    d = list(ys)
    h, lower, diag, upper, t = build_system(xs, d, kind, s0, sn)
    b = solve_tridiagonal(lower, diag, upper, t)

    # Step 5: back out a_i and c_i from b_i and d_i
    pieces = []
    for i in range(len(xs) - 1):
        a_i = (b[i + 1] - b[i]) / (3 * h[i])
        c_i = (d[i + 1] - d[i]) / h[i] - h[i] * (2 * b[i] + b[i + 1]) / 3
        pieces.append((a_i, b[i], c_i, d[i]))
    return pieces


def find_piece(xs, x):
    i = bisect_right(xs, x) - 1          # index of the last x_i that is <= x
    return min(max(i, 0), len(xs) - 2)   # clamp to a valid piece number


def evaluate(xs, pieces, x, derivative=0):
    """Step 6: evaluate g(x), g'(x) or g''(x) using the right piece."""
    x = Fraction(x)
    i = find_piece(xs, x)
    a, b, c, d = pieces[i]
    u = x - xs[i]
    if derivative == 0:
        return ((a * u + b) * u + c) * u + d
    if derivative == 1:
        return (3 * a * u + 2 * b) * u + c
    return 6 * a * u + 2 * b


def check_spline(xs, ys, pieces, kind, s0, sn):
    """Step 7: confirm every condition from the notes actually holds."""
    n = len(pieces)
    for i, (a, b, c, d) in enumerate(pieces):
        hi = xs[i + 1] - xs[i]
        assert d == ys[i]                                   # g_i(x_i) = y_i
        assert a * hi**3 + b * hi**2 + c * hi + d == ys[i + 1]   # g_i(x_{i+1}) = y_{i+1}
        if i < n - 1:
            a2, b2, c2, _ = pieces[i + 1]
            assert 3 * a * hi**2 + 2 * b * hi + c == c2     # g' continuous
            assert 6 * a * hi + 2 * b == 2 * b2             # g'' continuous
    if kind == "natural":
        assert evaluate(xs, pieces, xs[0], 2) == 0
        assert evaluate(xs, pieces, xs[-1], 2) == 0
    else:
        assert evaluate(xs, pieces, xs[0], 1) == s0
        assert evaluate(xs, pieces, xs[-1], 1) == sn


# formatting
def fmt(v):
    """Show a Fraction as an integer or a/b."""
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def piece_string(x_i, a, b, c, d):
    """Write a_i(x-x_i)^3 + b_i(x-x_i)^2 + c_i(x-x_i) + d_i, skipping zero terms."""
    shift = f"(x - {fmt(x_i)})" if x_i >= 0 else f"(x + {fmt(-x_i)})"
    terms = [(a, shift + "^3"), (b, shift + "^2"), (c, shift), (d, "")]
    out = ""
    for coef, part in terms:
        if coef == 0:
            continue
        sign = "-" if coef < 0 else "+"
        mag = fmt(abs(coef))
        body = mag + part if (abs(coef) != 1 or part == "") else part
        out += (("-" if sign == "-" else "") + body) if not out else f" {sign} {body}"
    return out or "0"


# Output
def cubic_spline(points, kind="natural", s0=None, sn=None):
    """Run the whole algorithm and print the spline. Returns (xs, pieces)."""
    xs, ys = prepare_points(points)
    pieces = spline_coefficients(xs, ys, kind, s0, sn)
    check_spline(xs, ys, pieces, kind, s0, sn)

    print(f"\n{kind.capitalize()} cubic spline:")
    for i, (a, bb, c, d) in enumerate(pieces):
        print(f"  g{i}(x) = {piece_string(xs[i], a, bb, c, d)},   {fmt(xs[i])} <= x <= {fmt(xs[i + 1])}")
    return xs, pieces


def main():
    print("Cubic Spline Interpolation")
    points, kind, s0, sn = get_inputs_from_user()
    try:
        xs, pieces = cubic_spline(points, kind, s0, sn)
    except ValueError as err:
        print("Error:", err)
        return

    print("\nEvaluate g(x) (blank line to quit):")
    while True:
        text = input("  x = ").strip()
        if not text:
            break
        try:
            x = Fraction(text)
        except (ValueError, ZeroDivisionError):
            print("  Not a number.")
            continue
        val = evaluate(xs, pieces, x)
        note = "" if xs[0] <= x <= xs[-1] else "   (outside data range: extrapolated)"
        print(f"  g({float(x):.10g}) = {float(val):.10g}{note}")


if __name__ == "__main__":
    main()