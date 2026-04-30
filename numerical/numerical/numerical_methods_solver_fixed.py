import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib


matplotlib.rcParams["axes.facecolor"] = "#F5F0E8"
matplotlib.rcParams["figure.facecolor"] = "#F5F0E8"
matplotlib.rcParams["text.color"] = "#3a3a3a"
matplotlib.rcParams["axes.labelcolor"] = "#3a3a3a"
matplotlib.rcParams["xtick.color"] = "#3a3a3a"
matplotlib.rcParams["ytick.color"] = "#3a3a3a"
matplotlib.rcParams["axes.edgecolor"] = "#A8D5C2"
matplotlib.rcParams["grid.color"] = "#D4EAE3"

st.set_page_config(
    page_title="Numerical Methods Solver",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: #F5F0E8 !important;
    color: #3a3a3a !important;
}
[data-testid="stSidebar"] {
    background-color: #D4EAE3 !important;
    border-right: 2px solid #A8D5C2;
}
[data-testid="stSidebar"] * { color: #3a3a3a !important; }

.main-title {
    font-size: 1.8rem; font-weight: 700;
    color: #3a3a3a; letter-spacing: 0.5px; margin-bottom: 2px;
}
.sub-title { font-size: 0.85rem; color: #7a7a7a; margin-bottom: 1.2rem; }

.method-badge {
    display: inline-block; background-color: #A8D5C2;
    color: #3a3a3a; padding: 4px 14px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600; margin-bottom: 1rem; letter-spacing: 0.4px;
}
.section-header {
    color: #3a3a3a; font-size: 0.95rem; font-weight: 600;
    border-bottom: 2px solid #A8D5C2; padding-bottom: 5px;
    margin-bottom: 10px; margin-top: 14px;
}
.result-box {
    background: #D4EAE3; border: 1px solid #A8D5C2;
    border-radius: 10px; padding: 0.9rem 1.2rem; margin-bottom: 1rem;
}
.result-label { color: #7a7a7a; font-size: 0.8rem; margin-bottom: 2px; }
.result-value { color: #C97D7D; font-size: 1.5rem; font-weight: 700; }

.info-box {
    background: #D4EAE3; border-left: 3px solid #A8D5C2;
    padding: 0.55rem 0.9rem; border-radius: 4px;
    font-size: 0.8rem; color: #5a5a5a; margin-bottom: 0.8rem;
}
.step-box {
    background: #F5F0E8; border: 1px solid #A8D5C2;
    border-radius: 8px; padding: 0.7rem 1rem; margin-bottom: 0.6rem;
}
.step-title { font-weight: 600; color: #3a3a3a; margin-bottom: 6px; font-size: 0.88rem; }
.step-note  { font-size: 0.8rem; color: #7a7a7a; margin-top: 4px; font-family: monospace; }

.stButton > button {
    background-color: #A8D5C2 !important; color: #3a3a3a !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; width: 100% !important;
    padding: 0.55rem 1.5rem !important; font-size: 0.95rem !important;
}
.stButton > button:hover { background-color: #8fc4af !important; }

label, .stSelectbox label, .stNumberInput label,
.stTextInput label { color: #3a3a3a !important; font-size: 0.85rem !important; }

div[data-baseweb="select"] > div {
    background-color: #F5F0E8 !important;
    border-color: #A8D5C2 !important; color: #3a3a3a !important;
}
input[type="number"], input[type="text"] {
    background-color: #F5F0E8 !important;
    border-color: #A8D5C2 !important; color: #3a3a3a !important;
}
.stDataFrame { border: 1px solid #A8D5C2; border-radius: 8px; }
hr { border-color: #A8D5C2; }
</style>
""",
    unsafe_allow_html=True,
)


# Helpers

ROOT_TOL = 1e-12
AUTO_MAX_ITERS = 300


def make_f(expr):
    expr = expr.strip()
    if not expr:
        raise ValueError("Equation cannot be empty.")

    safe = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
    }

    def f(x_val):
        local_scope = safe.copy()
        local_scope["x"] = x_val
        try:
            y = eval(expr, {"__builtins__": {}}, local_scope)
            y = float(y)
        except Exception as ex:
            raise ValueError(f"Cannot evaluate equation at x = {fmt(x_val)}: {ex}") from ex

        if not math.isfinite(y):
            raise ValueError(f"Equation returned a non-finite value at x = {fmt(x_val)}.")
        return y

    return f


def fmt(v, d=6):
    try:
        v = float(v)
    except Exception:
        return str(v)

    if not math.isfinite(v):
        return str(v)
    return f"{round(v, d):g}"


def mc(m):
    return [row[:] for row in m]


def relative_error(new, old):
    if old is None:
        return None
    if abs(new) > ROOT_TOL:
        return abs((new - old) / new) * 100
    return abs(new - old) * 100


def iter_limit(stop_mode, max_iters):
    return int(max_iters) if stop_mode == "iters" else AUTO_MAX_ITERS


def matrix_to_df(mat, n):
    cols = [f"x{i+1}" for i in range(n)] + ["b"]
    return pd.DataFrame(
        [[round(float(v), 6) for v in row] for row in mat],
        columns=cols,
        index=[f"Row {i+1}" for i in range(n)],
    )


def square_df(mat, n):
    cols = [f"x{i+1}" for i in range(n)]
    return pd.DataFrame(
        [[round(float(v), 6) for v in row] for row in mat],
        columns=cols,
        index=[f"Row {i+1}" for i in range(n)],
    )


# Stopping condition helper
# stop_mode: "eps_lte" => stop when error <= eps
#            "eps_gte" => stop when error >= eps
#            "iters"   => stop after fixed number of iterations


def should_stop(err, i, stop_mode, eps, max_iters):
    if stop_mode == "iters":
        return i >= int(max_iters) - 1
    if err is None:
        return False
    if stop_mode == "eps_lte":
        return err <= eps
    if stop_mode == "eps_gte":
        return err >= eps
    return False


def bracket_endpoint_row(f, xl, xu, xr):
    return {
        "Iter": 0,
        "xl": round(xl, 6),
        "f(xl)": round(f(xl), 6),
        "xu": round(xu, 6),
        "f(xu)": round(f(xu), 6),
        "xr": round(xr, 6),
        "f(xr)": round(f(xr), 6),
        "Error %": "---",
    }


# Root-finding methods


def bisection(f, xl, xu, stop_mode, eps, max_iters):
    rows = []
    xr_old = None
    fxl = f(xl)
    fxu = f(xu)

    if abs(fxl) < ROOT_TOL:
        return [bracket_endpoint_row(f, xl, xu, xl)], xl
    if abs(fxu) < ROOT_TOL:
        return [bracket_endpoint_row(f, xl, xu, xu)], xu
    if fxl * fxu > 0:
        raise ValueError("Bisection needs f(xl) and f(xu) to have opposite signs.")

    xr = xl
    for i in range(iter_limit(stop_mode, max_iters)):
        fxl = f(xl)
        fxu = f(xu)
        xr = (xl + xu) / 2
        fxr = f(xr)
        err = relative_error(xr, xr_old)

        rows.append(
            {
                "Iter": i,
                "xl": round(xl, 6),
                "f(xl)": round(fxl, 6),
                "xu": round(xu, 6),
                "f(xu)": round(fxu, 6),
                "xr": round(xr, 6),
                "f(xr)": round(fxr, 6),
                "Error %": "---" if err is None else round(err, 4),
            }
        )

        if abs(fxr) < ROOT_TOL or should_stop(err, i, stop_mode, eps, max_iters):
            break

        if fxl * fxr < 0:
            xu = xr
        else:
            xl = xr
        xr_old = xr

    return rows, xr


def false_position(f, xl, xu, stop_mode, eps, max_iters):
    rows = []
    xr_old = None
    fxl = f(xl)
    fxu = f(xu)

    if abs(fxl) < ROOT_TOL:
        return [bracket_endpoint_row(f, xl, xu, xl)], xl
    if abs(fxu) < ROOT_TOL:
        return [bracket_endpoint_row(f, xl, xu, xu)], xu
    if fxl * fxu > 0:
        raise ValueError("False Position needs f(xl) and f(xu) to have opposite signs.")

    xr = xl
    for i in range(iter_limit(stop_mode, max_iters)):
        fxl = f(xl)
        fxu = f(xu)
        denom = fxl - fxu
        if abs(denom) < ROOT_TOL:
            raise ZeroDivisionError("f(xl) - f(xu) is zero.")

        xr = xu - (fxu * (xl - xu)) / denom
        fxr = f(xr)
        err = relative_error(xr, xr_old)

        rows.append(
            {
                "Iter": i,
                "xl": round(xl, 6),
                "f(xl)": round(fxl, 6),
                "xu": round(xu, 6),
                "f(xu)": round(fxu, 6),
                "xr": round(xr, 6),
                "f(xr)": round(fxr, 6),
                "Error %": "---" if err is None else round(err, 4),
            }
        )

        if abs(fxr) < ROOT_TOL or should_stop(err, i, stop_mode, eps, max_iters):
            break

        if fxl * fxr < 0:
            xu = xr
        else:
            xl = xr
        xr_old = xr

    return rows, xr


def fixed_point(f, x0, stop_mode, eps, max_iters):
    rows = []
    xi = x0
    xi1 = x0

    for i in range(iter_limit(stop_mode, max_iters)):
        xi1 = f(xi)
        err = None if i == 0 else relative_error(xi1, xi)

        rows.append(
            {
                "Iter": i,
                "Xi": round(xi, 6),
                "Xi+1": round(xi1, 6),
                "Error %": "---" if err is None else round(err, 4),
            }
        )

        if should_stop(err, i, stop_mode, eps, max_iters):
            break
        xi = xi1

    return rows, xi1


def newton_raphson(f, fd, x0, stop_mode, eps, max_iters):
    rows = []
    xi = x0
    xi1 = x0

    for i in range(iter_limit(stop_mode, max_iters)):
        fxi = f(xi)
        fdxi = fd(xi)
        if abs(fdxi) < ROOT_TOL:
            raise ZeroDivisionError(f"f'(x) is zero at x = {fmt(xi)}.")

        xi1 = xi - fxi / fdxi
        err = None if i == 0 else relative_error(xi1, xi)

        rows.append(
            {
                "Iter": i,
                "Xi": round(xi, 6),
                "f(Xi)": round(fxi, 6),
                "f'(Xi)": round(fdxi, 6),
                "Xi+1": round(xi1, 6),
                "Error %": "---" if err is None else round(err, 4),
            }
        )

        if abs(f(xi1)) < ROOT_TOL or should_stop(err, i, stop_mode, eps, max_iters):
            break
        xi = xi1

    return rows, xi1


def secant(f, xm1, xi_val, stop_mode, eps, max_iters):
    rows = []
    xprev = xm1
    xi = xi_val
    xi_new = xi_val

    for i in range(iter_limit(stop_mode, max_iters)):
        fxprev = f(xprev)
        fxi = f(xi)
        denom = fxprev - fxi
        if abs(denom) < ROOT_TOL:
            raise ZeroDivisionError("f(Xi-1) - f(Xi) is zero.")

        xi_new = xi - (fxi * (xprev - xi)) / denom
        err = relative_error(xi_new, xi)

        rows.append(
            {
                "Iter": i,
                "Xi-1": round(xprev, 6),
                "f(Xi-1)": round(fxprev, 6),
                "Xi": round(xi, 6),
                "f(Xi)": round(fxi, 6),
                "Xi+1": round(xi_new, 6),
                "Error %": round(err, 4),
            }
        )

        if abs(f(xi_new)) < ROOT_TOL or should_stop(err, i, stop_mode, eps, max_iters):
            break
        xprev, xi = xi, xi_new

    return rows, xi_new


# Linear system methods


def gauss_elimination(A, pivot=False):
    n, mat, steps, mults = len(A), mc(A), [], {}

    for col in range(n - 1):
        if pivot:
            max_row = max(range(col, n), key=lambda r: abs(mat[r][col]))
            if max_row != col:
                mat[col], mat[max_row] = mat[max_row], mat[col]
                steps.append(
                    {
                        "title": f"Pivot: swap Row {col+1} <-> Row {max_row+1}",
                        "note": f"|a[{max_row+1}][{col+1}]| is largest in col {col+1}",
                        "mat": mc(mat),
                    }
                )

        pv = mat[col][col]
        if abs(pv) < ROOT_TOL:
            msg = f"Zero pivot at col {col+1}."
            if not pivot:
                msg += " Try enabling pivoting."
            raise ValueError(msg)

        for row in range(col + 1, n):
            m = mat[row][col] / pv
            key = f"m{row+1}{col+1}"
            mults[key] = round(m, 6)
            for j in range(col, n + 1):
                mat[row][j] -= m * mat[col][j]
            steps.append(
                {
                    "title": f"Eliminate col {col+1} from Row {row+1}",
                    "note": f"{key} = {round(m, 6)} => R{row+1} = R{row+1} - {round(m, 6)} * R{col+1}",
                    "mat": mc(mat),
                }
            )

    x = [0.0] * n
    bk = []
    for i in range(n - 1, -1, -1):
        if abs(mat[i][i]) < ROOT_TOL:
            raise ValueError(f"Zero diagonal at row {i+1}.")
        x[i] = (mat[i][n] - sum(mat[i][k] * x[k] for k in range(i + 1, n))) / mat[i][i]
        rhs = f"{round(mat[i][n], 6)}"
        if i < n - 1:
            terms = " - ".join(
                f"({round(mat[i][k], 6)}*{round(x[k], 6)})" for k in range(i + 1, n)
            )
            rhs = f"({round(mat[i][n], 6)} - {terms})"
        bk.append(
            {
                "title": f"x{i+1}",
                "note": f"x{i+1} = {rhs} / {round(mat[i][i], 6)} = {round(x[i], 6)}",
            }
        )

    return x, mults, steps, list(reversed(bk))


def gauss_jordan(A, pivot=False):
    n, mat, steps = len(A), mc(A), []

    for col in range(n - 1):
        if pivot:
            max_row = max(range(col, n), key=lambda r: abs(mat[r][col]))
            if max_row != col:
                mat[col], mat[max_row] = mat[max_row], mat[col]
                steps.append(
                    {
                        "phase": "pivot",
                        "title": f"Pivot: swap Row {col+1} <-> Row {max_row+1}",
                        "note": f"|a[{max_row+1}][{col+1}]| is largest in col {col+1}",
                        "mat": mc(mat),
                    }
                )

        pv = mat[col][col]
        if abs(pv) < ROOT_TOL:
            msg = f"Zero pivot at col {col+1}."
            if not pivot:
                msg += " Try enabling pivoting."
            raise ValueError(msg)

        for row in range(col + 1, n):
            if abs(mat[row][col]) < ROOT_TOL:
                continue
            m = mat[row][col] / pv
            for j in range(col, n + 1):
                mat[row][j] -= m * mat[col][j]
            steps.append(
                {
                    "phase": "forward",
                    "title": f"Forward: eliminate col {col+1} from Row {row+1}",
                    "note": f"m{row+1}{col+1} = {round(m, 6)} => R{row+1} = R{row+1} - {round(m, 6)}*R{col+1}",
                    "mat": mc(mat),
                }
            )

    for row in range(n):
        pv = mat[row][row]
        if abs(pv) < ROOT_TOL:
            raise ValueError(f"Zero diagonal at row {row+1}.")
        for j in range(row, n + 1):
            mat[row][j] /= pv
        steps.append(
            {
                "phase": "normalize",
                "title": f"Normalize Row {row+1} (divide by {round(pv, 6)})",
                "note": f"R{row+1} = R{row+1} / {round(pv, 6)}",
                "mat": mc(mat),
            }
        )

    for col in range(n - 1, 0, -1):
        for row in range(col - 1, -1, -1):
            m = mat[row][col]
            if abs(m) < ROOT_TOL:
                continue
            for j in range(col, n + 1):
                mat[row][j] -= m * mat[col][j]
            steps.append(
                {
                    "phase": "backward",
                    "title": f"Backward: eliminate col {col+1} from Row {row+1}",
                    "note": f"m{row+1}{col+1} = {round(m, 6)} => R{row+1} = R{row+1} - {round(m, 6)}*R{col+1}",
                    "mat": mc(mat),
                }
            )

    return [mat[i][n] for i in range(n)], steps


def lu_decomposition(A, pivot=False):
    n = len(A)
    b = [A[i][n] for i in range(n)]
    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    U = [[A[i][j] for j in range(n)] for i in range(n)]
    p = list(range(n))
    piv_steps, elim_steps = [], []

    for col in range(n - 1):
        if pivot:
            max_row = max(range(col, n), key=lambda r: abs(U[r][col]))
            if max_row != col:
                U[col], U[max_row] = U[max_row], U[col]
                p[col], p[max_row] = p[max_row], p[col]
                for k in range(col):
                    L[col][k], L[max_row][k] = L[max_row][k], L[col][k]
                piv_steps.append(
                    {
                        "title": f"Pivot col {col+1}: swap Row {col+1} <-> Row {max_row+1}",
                        "note": f"Permutation array p = {p}",
                    }
                )

        pv = U[col][col]
        if abs(pv) < ROOT_TOL:
            msg = f"Zero pivot at col {col+1}."
            if not pivot:
                msg += " Try enabling pivoting."
            raise ValueError(msg)

        for row in range(col + 1, n):
            m = U[row][col] / pv
            L[row][col] = m
            for j in range(col, n):
                U[row][j] -= m * U[col][j]
            elim_steps.append(
                {
                    "title": f"m{row+1}{col+1} = {round(m, 6)}",
                    "note": f"L[{row+1}][{col+1}] = {round(m, 6)} => U row{row+1} = U row{row+1} - {round(m, 6)}*U row{col+1}",
                }
            )

    pb = [b[p[i]] for i in range(n)] if pivot else b[:]

    c, fwd = [0.0] * n, []
    for i in range(n):
        c[i] = pb[i] - sum(L[i][k] * c[k] for k in range(i))
        sub = " - ".join(
            f"L[{i+1}][{k+1}]*c{k+1} ({round(L[i][k], 6)}*{round(c[k], 6)})"
            for k in range(i)
        )
        note = f"c{i+1} = {'pb' if pivot else 'b'}{i+1} ({round(pb[i], 6)})"
        if sub:
            note += f" - ({sub})"
        note += f" = {round(c[i], 6)}"
        fwd.append({"title": f"c{i+1}", "note": note})

    x, bk = [0.0] * n, []
    for i in range(n - 1, -1, -1):
        if abs(U[i][i]) < ROOT_TOL:
            raise ValueError(f"Zero diagonal in U at row {i+1}.")
        x[i] = (c[i] - sum(U[i][k] * x[k] for k in range(i + 1, n))) / U[i][i]
        rhs = f"c{i+1} ({round(c[i], 6)})"
        if i < n - 1:
            terms = " - ".join(
                f"U[{i+1}][{k+1}]*x{k+1} ({round(U[i][k], 6)}*{round(x[k], 6)})"
                for k in range(i + 1, n)
            )
            rhs = f"c{i+1} - {terms}"
        bk.append(
            {
                "title": f"x{i+1}",
                "note": f"x{i+1} = ({rhs}) / U[{i+1}][{i+1}] ({round(U[i][i], 6)}) = {round(x[i], 6)}",
            }
        )

    return x, L, U, c, pb, p, piv_steps, elim_steps, fwd, list(reversed(bk))


def cramers_rule(A):
    n = len(A)
    mat = [[A[i][j] for j in range(n)] for i in range(n)]
    b = [A[i][n] for i in range(n)]

    def det(M):
        sz = len(M)
        if sz == 1:
            return M[0][0]
        if sz == 2:
            return M[0][0] * M[1][1] - M[0][1] * M[1][0]
        d = 0.0
        for c in range(sz):
            sub = [[M[r][cc] for cc in range(sz) if cc != c] for r in range(1, sz)]
            d += ((-1) ** c) * M[0][c] * det(sub)
        return d

    det_A = det(mat)
    if abs(det_A) < ROOT_TOL:
        raise ValueError("det(A) = 0 - no unique solution.")

    xs, det_steps = [], []
    for i in range(n):
        Mi = [row[:] for row in mat]
        for r in range(n):
            Mi[r][i] = b[r]
        det_Mi = det(Mi)
        xs.append(det_Mi / det_A)
        det_steps.append(
            {
                "var": f"x{i+1}",
                "det_Mi": round(det_Mi, 6),
                "result": round(det_Mi / det_A, 6),
                "mat": Mi,
            }
        )

    return xs, det_A, det_steps


# Plot


def plot_function(f, root, x_range=None):
    a = x_range[0] if x_range else root - 3
    b = x_range[1] if x_range else root + 3
    if a == b:
        a -= 1
        b += 1
    if a > b:
        a, b = b, a

    xs = np.linspace(a, b, 500)
    ys = []
    for xv in xs:
        try:
            ys.append(f(xv))
        except Exception:
            ys.append(np.nan)

    ys = np.array(ys, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(xs, ys, color="#A8D5C2", linewidth=2.2, label="f(x)")
    ax.axhline(0, color="#aaa", linewidth=0.8)
    ax.axvline(0, color="#aaa", linewidth=0.8)
    ax.scatter([root], [0], color="#C97D7D", s=90, zorder=5, label=f"Root = {fmt(root)}")
    ax.annotate(f"  x = {fmt(root)}", xy=(root, 0), color="#C97D7D", fontsize=9, va="bottom")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Function Plot", fontsize=11, color="#3a3a3a")
    plt.tight_layout()
    return fig


# UI helpers


def section(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def result_card(label, value):
    st.markdown(
        f'<div class="result-box"><div class="result-label">{label}</div>'
        f'<div class="result-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def step_card(title, note):
    st.markdown(
        f'<div class="step-box"><div class="step-title">{title}</div>'
        f'<div class="step-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def show_matrix_step(mat, n, title, note=""):
    step_card(title, note)
    cols_lbl = [f"x{i+1}" for i in range(n)] + ["b"]
    st.dataframe(
        pd.DataFrame(
            [[round(v, 6) for v in row] for row in mat],
            columns=cols_lbl,
            index=[f"R{i+1}" for i in range(n)],
        ),
        use_container_width=True,
    )


def solution_row(sol, n):
    section("Solution")
    cols = st.columns(n)
    for i, v in enumerate(sol):
        with cols[i]:
            result_card(f"x{i+1}", fmt(v))


def show_back_steps(bk):
    section("Back Substitution")
    for s in bk:
        step_card(s["title"], s["note"])


def show_gauss_steps(elim_steps, mults, bk, n, phase_label="Forward Elimination"):
    section("Multipliers")
    st.dataframe(
        pd.DataFrame([{"Multiplier": k, "Value": v} for k, v in mults.items()]),
        use_container_width=True,
        hide_index=True,
    )
    section(phase_label)
    for s in elim_steps:
        show_matrix_step(s["mat"], n, s["title"], s["note"])
    show_back_steps(bk)


def show_gj_steps(steps, n):
    fwd = [s for s in steps if s["phase"] == "forward"]
    piv = [s for s in steps if s["phase"] == "pivot"]
    norm = [s for s in steps if s["phase"] == "normalize"]
    bwd = [s for s in steps if s["phase"] == "backward"]

    if piv:
        section("Step 1 - Pivoting + Forward Elimination")
        for s in piv + fwd:
            show_matrix_step(s["mat"], n, s["title"], s["note"])
    elif fwd:
        section("Step 1 - Forward Elimination")
        for s in fwd:
            show_matrix_step(s["mat"], n, s["title"], s["note"])
    if norm:
        section("Step 2 - Normalize Pivots to 1")
        for s in norm:
            show_matrix_step(s["mat"], n, s["title"], s["note"])
    if bwd:
        section("Step 3 - Backward Elimination")
        for s in bwd:
            show_matrix_step(s["mat"], n, s["title"], s["note"])


# Sidebar

with st.sidebar:
    st.markdown('<div class="main-title">NM Solver</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Numerical Analysis</div>', unsafe_allow_html=True)
    st.markdown("---")
    chapter = st.radio("Chapter", ["Root Finding", "Linear Systems"])
    st.markdown("---")

    if chapter == "Root Finding":
        method = st.selectbox(
            "Method",
            ["Bisection", "False Position", "Simple Fixed Point", "Newton-Raphson", "Secant"],
        )
        st.markdown(f'<div class="method-badge">{method}</div>', unsafe_allow_html=True)

        defaults = {
            "Bisection": ("2*x - 1.75*x**2 + 1.1*x**3 - 0.25*x**4", None),
            "False Position": ("2*x - 1.75*x**2 + 1.1*x**3 - 0.25*x**4", None),
            "Simple Fixed Point": ("sqrt(1.9*x + 2.8)", None),
            "Newton-Raphson": (
                "2*x - 1.75*x**2 + 1.1*x**3 - 0.25*x**4",
                "2 - 3.5*x + 3.3*x**2 - x**3",
            ),
            "Secant": ("2*x - 1.75*x**2 + 1.1*x**3 - 0.25*x**4", None),
        }

        section("Equation")
        with st.expander("Supported syntax"):
            st.code(
                "Operators : +  -  *  **  /\n"
                "Functions : sqrt  sin  cos  tan\n"
                "            log  log10  exp  abs\n"
                "Constants : pi  e\n"
                "Variable  : x",
                language="text",
            )

        label = "g(x) =" if method == "Simple Fixed Point" else "f(x) ="
        equation = st.text_input(label, value=defaults[method][0])
        fd_equation = None
        if method == "Newton-Raphson":
            fd_equation = st.text_input("f'(x) =", value=defaults[method][1])

        section("Parameters")
        xl = xu = x0 = xm1 = xi_val = None
        if method in ["Bisection", "False Position"]:
            c1, c2 = st.columns(2)
            with c1:
                xl = st.number_input("xl", value=0.0, format="%g")
            with c2:
                xu = st.number_input("xu", value=1.0, format="%g")
        elif method in ["Simple Fixed Point", "Newton-Raphson"]:
            x0 = st.number_input("x0  (initial guess)", value=0.0, format="%g")
        else:
            c1, c2 = st.columns(2)
            with c1:
                xm1 = st.number_input("Xi-1", value=0.0, format="%g")
            with c2:
                xi_val = st.number_input("Xi", value=1.0, format="%g")

        section("Stopping Condition")
        stop_mode = st.selectbox(
            "Stop when",
            ["Error %  <=  eps  (converged)", "Error %  >=  eps", "Fixed number of iterations"],
            index=0,
        )
        stop_mode_key = {
            "Error %  <=  eps  (converged)": "eps_lte",
            "Error %  >=  eps": "eps_gte",
            "Fixed number of iterations": "iters",
        }[stop_mode]

        eps = 0.5
        max_iters = 10
        if stop_mode_key in ("eps_lte", "eps_gte"):
            eps = st.number_input("eps (%)", value=0.5, min_value=0.0, format="%g")
        else:
            max_iters = st.number_input("Number of iterations", value=5, min_value=1, max_value=500, step=1)

        show_graph = st.checkbox("Show function graph", value=True)
        plot_a = plot_b = None
        if show_graph and method in ["Bisection", "False Position"]:
            c1, c2 = st.columns(2)
            with c1:
                plot_a = st.number_input("Graph from", value=float(xl) - 1, format="%g")
            with c2:
                plot_b = st.number_input("Graph to", value=float(xu) + 1, format="%g")

    else:
        ls_method = st.selectbox(
            "Method",
            ["Gauss Elimination", "Gauss-Jordan Elimination", "LU Decomposition", "Cramer's Rule"],
        )
        st.markdown(f'<div class="method-badge">{ls_method}</div>', unsafe_allow_html=True)

        use_pivot = False
        if ls_method != "Cramer's Rule":
            use_pivot = st.checkbox("Use Pivoting", value=False)

        section("Matrix Size")
        n_size = st.selectbox("System size  (n x n)", [2, 3, 4], index=1)

        section("Augmented Matrix  [A | b]")
        st.markdown(
            '<div class="info-box">Enter coefficients row by row. Last column = b.</div>',
            unsafe_allow_html=True,
        )

        matrix_input = []
        for i in range(n_size):
            cols_in = st.columns(n_size + 1)
            row = []
            for j in range(n_size):
                with cols_in[j]:
                    row.append(st.number_input(f"a{i+1}{j+1}", value=0.0, key=f"m{i}{j}", format="%g"))
            with cols_in[n_size]:
                row.append(st.number_input(f"b{i+1}", value=0.0, key=f"b{i}", format="%g"))
            matrix_input.append(row)

        show_steps = st.checkbox("Show step-by-step solution", value=True)

    st.markdown("---")
    solve_btn = st.button("Solve")


# Main

st.markdown('<div class="main-title">Numerical Methods Solver</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Root Finding  |  Linear Systems</div>', unsafe_allow_html=True)
st.markdown("---")

if not solve_btn:
    st.markdown(
        '<div class="info-box">Select a chapter and method from the sidebar, '
        "fill in the parameters, then click <b>Solve</b>.</div>",
        unsafe_allow_html=True,
    )

else:
    try:
        if chapter == "Root Finding":
            f = make_f(equation)

            if method == "Bisection":
                rows, root = bisection(f, xl, xu, stop_mode_key, eps, max_iters)
            elif method == "False Position":
                rows, root = false_position(f, xl, xu, stop_mode_key, eps, max_iters)
            elif method == "Simple Fixed Point":
                rows, root = fixed_point(f, x0, stop_mode_key, eps, max_iters)
            elif method == "Newton-Raphson":
                fd = make_f(fd_equation)
                rows, root = newton_raphson(f, fd, x0, stop_mode_key, eps, max_iters)
            elif method == "Secant":
                rows, root = secant(f, xm1, xi_val, stop_mode_key, eps, max_iters)

            c1, c2, c3 = st.columns(3)
            with c1:
                result_card("Root", fmt(root, 8))
            with c2:
                result_card("f(root)", fmt(f(root), 8))
            with c3:
                result_card("Iterations", str(len(rows)))

            if show_graph:
                xr = (plot_a, plot_b) if (plot_a is not None and plot_b is not None) else None
                st.pyplot(plot_function(f, root, xr))

            section("Iteration Table")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        else:
            A, n = [[float(v) for v in row] for row in matrix_input], int(n_size)
            section("Input Matrix  [A | b]")
            st.dataframe(matrix_to_df(A, n), use_container_width=True)

            if ls_method == "Gauss Elimination":
                sol, mults, elim_steps, bk = gauss_elimination(A, pivot=use_pivot)
                if show_steps:
                    lbl = f"{'Pivoting + ' if use_pivot else ''}Forward Elimination"
                    show_gauss_steps(elim_steps, mults, bk, n, lbl)
                solution_row(sol, n)

            elif ls_method == "Gauss-Jordan Elimination":
                sol, steps = gauss_jordan(A, pivot=use_pivot)
                if show_steps:
                    show_gj_steps(steps, n)
                section("Solution  (last column of final matrix)")
                cols = st.columns(n)
                for i, v in enumerate(sol):
                    with cols[i]:
                        result_card(f"x{i+1}", fmt(v))

            elif ls_method == "LU Decomposition":
                sol, L, U, c, pb, p, piv_steps, elim_steps, fwd, bk = lu_decomposition(A, pivot=use_pivot)
                section("L Matrix")
                st.dataframe(square_df(L, n).round(6), use_container_width=True)
                section("U Matrix")
                st.dataframe(square_df(U, n).round(6), use_container_width=True)
                if show_steps:
                    if use_pivot and piv_steps:
                        section("Step 1 - Pivoting + Elimination  (building L and U)")
                        for s in piv_steps:
                            step_card(s["title"], s["note"])
                    else:
                        section("Step 1 - Elimination  (building L and U)")
                    for s in elim_steps:
                        step_card(s["title"], s["note"])
                    if use_pivot:
                        section(f"Step 2 - Apply Permutation to b   p = {p}")
                        st.dataframe(
                            pd.DataFrame({"p[i]": p, "pb[i] = b[p[i]]": [round(v, 6) for v in pb]}),
                            use_container_width=True,
                            hide_index=False,
                        )
                        section("Step 3 - Forward Substitution  (Lc = Pb)")
                    else:
                        section("Step 2 - Forward Substitution  (Lc = b)")
                    for s in fwd:
                        step_card(s["title"], s["note"])
                    section(f"Step {'4' if use_pivot else '3'} - Back Substitution  (Ux = c)")
                    for s in bk:
                        step_card(s["title"], s["note"])
                section("c vector")
                st.dataframe(pd.DataFrame({"c": [round(v, 6) for v in c]}), use_container_width=True)
                section("Solution  (Ux = c)")
                cols = st.columns(n)
                for i, v in enumerate(sol):
                    with cols[i]:
                        result_card(f"x{i+1}", fmt(v))

            elif ls_method == "Cramer's Rule":
                sol, det_A, det_steps = cramers_rule(A)
                st.markdown(
                    f'<div class="info-box">det(A) = <b>{round(det_A, 6)}</b></div>',
                    unsafe_allow_html=True,
                )
                if show_steps:
                    section("Determinant Substitution Steps")
                    for s in det_steps:
                        cols_lbl = [f"x{i+1}" for i in range(n)]
                        step_card(
                            f"Matrix for {s['var']} - det = {s['det_Mi']}",
                            f"{s['var']} = {s['det_Mi']} / {round(det_A, 6)} = {s['result']}",
                        )
                        st.dataframe(
                            pd.DataFrame(
                                [[round(v, 6) for v in row] for row in s["mat"]],
                                columns=cols_lbl,
                                index=[f"R{i+1}" for i in range(n)],
                            ),
                            use_container_width=True,
                        )
                section("Solution")
                cols = st.columns(n)
                for i, v in enumerate(sol):
                    with cols[i]:
                        result_card(f"x{i+1}", fmt(v))

    except ZeroDivisionError as ex:
        st.error(f"Division by zero - {ex}")
    except Exception as ex:
        st.error(f"Error: {ex}")

st.markdown("---")
st.caption("Numerical Analysis  |  Ch.1 Root Finding   Ch.2 Linear Systems")
