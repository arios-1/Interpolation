from fractions import Fraction


# input
def read_number(prompt):
    """Ask until the user types a valid number (integer, decimal, or fraction like 1/3)."""
    while True:
        text = input(prompt).strip()
        try:
            return Fraction(text)
        except (ValueError, ZeroDivisionError):
            print("  Please enter a number, e.g. 2, -1.5, or 3/4.")


def get_points_from_user():
    """Step 1: collect the data points."""
    while True:
        try:
            m = int(input("How many data points? "))
            if m >= 1:
                break
        except ValueError:
            pass
        print("  Please enter a whole number of at least 1.")

    points = []
    for i in range(m):
        print(f"Point {i + 1}:")
        x = read_number("  x = ")
        y = read_number("  y = ")
        points.append((x, y))
    return points


# core algorithm
def prepare_points(points):
    """Step 2: sort so x0 < x1 < ... < xn, and reject repeated x-values."""
    points = sorted(points, key=lambda p: p[0])
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    for i in range(1, len(xs)):
        if xs[i] == xs[i - 1]:
            raise ValueError(f"Two points share x = {xs[i]}; x-values must be distinct "
                             "(otherwise a divided difference would divide by zero).")
    return xs, ys


def divided_difference_table(xs, ys):
    """
    Step 3: build the divided-difference table.

    table[i][j] = [y_{i-j}, ..., y_i]  (the j-th order difference ending at row i)
      column 0  : the y-values
      column j  : (table[i][j-1] - table[i-1][j-1]) / (x_i - x_{i-j})
    """
    n = len(xs)
    table = [[None] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = ys[i]
    for j in range(1, n):                 # order of the difference
        for i in range(j, n):             # rows that have a value in this column
            table[i][j] = (table[i][j - 1] - table[i - 1][j - 1]) / (xs[i] - xs[i - j])
    return table


def newton_coefficients(table):
    """Step 4: b_k is the diagonal entry table[k][k] (first filled entry of column k)."""
    return [table[k][k] for k in range(len(table))]


def evaluate(xs, b, x):
    """
    Step 5: evaluate P(x) with nested (Horner-style) multiplication:
      P(x) = b0 + (x-x0)(b1 + (x-x1)(b2 + ... + (x-x_{n-1}) bn))
    """
    x = Fraction(x)
    result = b[-1]
    for k in range(len(b) - 2, -1, -1):
        result = result * (x - xs[k]) + b[k]
    return result


def expand_to_standard_form(xs, b):
    """
    Step 6: multiply out the Newton form into a0 + a1 x + a2 x^2 + ...
    Returns the list of coefficients [a0, a1, ..., an].
    """
    n = len(b)
    poly = [Fraction(0)] * n          # running total
    basis = [Fraction(1)]             # current product (x-x0)(x-x1)... starts at 1
    for k in range(n):
        for p, c in enumerate(basis):     # add b_k * basis to the total
            poly[p] += b[k] * c
        if k < n - 1:                     # basis *= (x - x_k)
            new = [Fraction(0)] * (len(basis) + 1)
            for p, c in enumerate(basis):
                new[p + 1] += c
                new[p] -= xs[k] * c
            basis = new
    while len(poly) > 1 and poly[-1] == 0:   # drop zero leading terms
        poly.pop()
    return poly


# formatting
def fmt(v):
    """Show a Fraction as an integer or a/b."""
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def newton_form_string(xs, b):
    terms = []
    for k, bk in enumerate(b):
        if bk == 0 and k > 0:
            continue
        factors = "".join(f"(x - {fmt(xs[i])})" for i in range(k))
        coef = fmt(bk) if bk >= 0 else f"({fmt(bk)})"
        terms.append(coef + factors)
    return " + ".join(terms)


def standard_form_string(a):
    parts = []
    for p in range(len(a) - 1, -1, -1):
        c = a[p]
        if c == 0:
            continue
        sign = "-" if c < 0 else "+"
        mag = abs(c)
        if p == 0:
            body = fmt(mag)
        else:
            coef = "" if mag == 1 else fmt(mag)
            body = coef + ("x" if p == 1 else f"x^{p}")
        parts.append((sign, body))
    if not parts:
        return "0"
    first_sign, first_body = parts[0]
    out = ("-" if first_sign == "-" else "") + first_body
    for sign, body in parts[1:]:
        out += f" {sign} {body}"
    return out


# output
def newton_interpolation(points):
    """Run the whole algorithm. Returns (xs, b, standard-form coefficients)."""
    xs, ys = prepare_points(points)
    table = divided_difference_table(xs, ys)
    b = newton_coefficients(table)
    a = expand_to_standard_form(xs, b)

    assert all(evaluate(xs, b, x) == y for x, y in zip(xs, ys))

    print("\nNewton form:     P(x) = " + newton_form_string(xs, b))
    print("Standard form:   P(x) = " + standard_form_string(a))
    return xs, b, a


def main():
    print("Newton Interpolation")
    points = get_points_from_user()
    try:
        xs, b, _ = newton_interpolation(points)
    except ValueError as err:
        print("Error:", err)
        return

    print("\nEvaluate P(x) (blank line to quit):")
    while True:
        text = input("  x = ").strip()
        if not text:
            break
        try:
            x = Fraction(text)
        except (ValueError, ZeroDivisionError):
            print("  Not a number.")
            continue
        val = evaluate(xs, b, x)
        print(f"  P({float(x):.10g}) = {float(val):.10g}")


if __name__ == "__main__":
    main()