import streamlit as st
import pandas as pd #for tables and data display.
import numpy as np #for numerical operations.
import math
import matplotlib.pyplot as plt #used for plotting graphs.
import matplotlib
#The rcParams lines customize the overall theme and colors of the plots.
matplotlib.rcParams['axes.facecolor']  = '#F5F0E8'
matplotlib.rcParams['figure.facecolor']= '#F5F0E8'
matplotlib.rcParams['text.color']      = '#3a3a3a'
matplotlib.rcParams['axes.labelcolor'] = '#3a3a3a'
matplotlib.rcParams['xtick.color']     = '#3a3a3a'
matplotlib.rcParams['ytick.color']     = '#3a3a3a'
matplotlib.rcParams['axes.edgecolor']  = '#A8D5C2'
matplotlib.rcParams['grid.color']      = '#D4EAE3'

st.set_page_config(
    page_title="Numerical Methods Solver",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ── palette ──────────────────────────────────────────────────
   mint      #A8D5C2
   light mint#D4EAE3
   cream     #F5F0E8
   rose      #C97D7D
   text      #3a3a3a
──────────────────────────────────────────────────────────── */

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
    font-size: 1.8rem;
    font-weight: 700;
    color: #3a3a3a;
    letter-spacing: 0.5px;
    margin-bottom: 2px;
}
.sub-title {
    font-size: 0.85rem;
    color: #7a7a7a;
    margin-bottom: 1.2rem;
}

.method-badge {
    display: inline-block;
    background-color: #A8D5C2;
    color: #3a3a3a;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 1rem;
    letter-spacing: 0.4px;
}

.section-header {
    color: #3a3a3a;
    font-size: 0.95rem;
    font-weight: 600;
    border-bottom: 2px solid #A8D5C2;
    padding-bottom: 5px;
    margin-bottom: 10px;
    margin-top: 14px;
}

.result-box {
    background: #D4EAE3;
    border: 1px solid #A8D5C2;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 1rem;
}
.result-label { color: #7a7a7a; font-size: 0.8rem; margin-bottom: 2px; }
.result-value { color: #C97D7D; font-size: 1.5rem; font-weight: 700; }

.info-box {
    background: #D4EAE3;
    border-left: 3px solid #A8D5C2;
    padding: 0.55rem 0.9rem;
    border-radius: 4px;
    font-size: 0.8rem;
    color: #5a5a5a;
    margin-bottom: 0.8rem;
}

.stButton > button {
    background-color: #A8D5C2 !important;
    color: #3a3a3a !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    width: 100% !important;
    padding: 0.55rem 1.5rem !important;
    font-size: 0.95rem !important;
    transition: background 0.2s;
}
.stButton > button:hover {
    background-color: #8fc4af !important;
}

label, .stSelectbox label, .stNumberInput label,
.stTextInput label { color: #3a3a3a !important; font-size: 0.85rem !important; }

div[data-baseweb="select"] > div {
    background-color: #F5F0E8 !important;
    border-color: #A8D5C2 !important;
    color: #3a3a3a !important;
}

input[type="number"], input[type="text"] {
    background-color: #F5F0E8 !important;
    border-color: #A8D5C2 !important;
    color: #3a3a3a !important;
}

.stDataFrame { border: 1px solid #A8D5C2; border-radius: 8px; }

hr { border-color: #A8D5C2; }
</style>
""", unsafe_allow_html=True)


# ── helpers ────────────────────────────────────────────────────────────────────
# This function converts a user-input string into an executable function f(x).
def make_f(expr: str):
    safe = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "log10": math.log10,
            "exp": math.exp, "pi": math.pi, "e": math.e,
            "abs": abs, "__builtins__": {}}
    def f(x_val):
        g = safe.copy(); g["x"] = x_val
        return eval(expr, g)
    return f
def df_table(dataframe):
    return st.dataframe(dataframe, width="stretch") 
# ── Root-finding methods ───────────────────────────────────────────────────────
# This function implements the Bisection method to find the root of a function. It repeatedly divides the interval [xl, xu] into two halves
def bisection(f, xl, xu, eps):
    rows, xr, xrold = [], 0.0, 0.0
    for i in range(300):
        xrold = xr
        xr    = (xl + xu) / 2
        fxr   = f(xr)
        err   = abs((xr - xrold) / xr) * 100 if (xr != 0 and i > 0) else None
        rows.append({"Iter": i,
                     "xl": round(xl,6),   "f(xl)": round(f(xl),6),
                     "xu": round(xu,6),   "f(xu)": round(f(xu),6),
                     "xr": round(xr,6),   "f(xr)": round(fxr,6),
                     "Error %": np.nan if err is None else round(err,4)
})
        if err is not None and err <= eps: break
        xl, xu = (xr, xu) if f(xl)*fxr > 0 else (xl, xr)
    return rows, xr

# This function implements the False Position (Regula Falsi) method.
# Instead of using the midpoint, it calculates the root using a linear
# interpolation between xl and xu.

def false_position(f, xl, xu, eps):
    rows, xr, xrold = [], 0.0, 0.0
    for i in range(300):
        xrold = xr
        xr    = xu - (f(xu)*(xl-xu)) / (f(xl)-f(xu))
        fxr   = f(xr)
        err   = abs((xr-xrold)/xr)*100 if (xr != 0 and i > 0) else None
        rows.append({"Iter": i,
                     "xl": round(xl,6),   "f(xl)": round(f(xl),6),
                     "xu": round(xu,6),   "f(xu)": round(f(xu),6),
                     "xr": round(xr,6),   "f(xr)": round(fxr,6),
                     "Error %": np.nan if err is None else round(err,4)})
        if err is not None and err <= eps: break
        xl, xu = (xr, xu) if f(xl)*fxr > 0 else (xl, xr)
    return rows, xr
# This function implements the Simple Fixed Point iteration method.
# The equation must be rearranged into the form x = g(x).
def fixed_point(f, x0, eps):
    rows, xi = [], x0
    for i in range(300):
        xi1 = f(xi)
        err = abs((xi1-xi)/xi1)*100 if xi1 != 0 else 0
        rows.append({"Iter": i,
                     "Xi": round(xi,6), "Xi+1": round(xi1,6),
                     "Error %": np.nan if i == 0 else round(err,4)})
        if i > 0 and err <= eps: break
        xi = xi1
    return rows, xi
# This function implements the Newton-Raphson method.
# It uses both the function f(x) and its derivative f'(x)
def newton_raphson(f, fd, x0, eps):
    rows, xi = [], x0
    for i in range(300):
        fxi, fdxi = f(xi), fd(xi)
        xi1 = xi - fxi/fdxi
        err = abs((xi1-xi)/xi1)*100 if xi1 != 0 else 0
        rows.append({"Iter": i,
                     "Xi": round(xi,6), "f(Xi)": round(fxi,6),
                     "f'(Xi)": round(fdxi,6),
                     "Error %": np.nan if i == 0 else round(err,4)})
        if i > 0 and err <= eps: break
        xi = xi1
    return rows, xi
# This function implements the Secant method.
# It approximates the derivative using two previous points instead of requiring an explicit derivative.
def secant(f, xm1, xi_val, eps):
    rows, xi, xprev = [], xi_val, xm1
    for i in range(300):
        fxi, fxprev = f(xi), f(xprev)
        err = abs((xi-xprev)/xi)*100 if xi != 0 else 0
        rows.append({"Iter": i,
                     "Xi-1": round(xprev,6), "f(Xi-1)": round(fxprev,6),
                     "Xi":   round(xi,6),    "f(Xi)":   round(fxi,6),
                     "Error %": round(err,4)})
        if i > 0 and err <= eps: break
        xi_new = xi - (fxi*(xprev-xi))/(fxprev-fxi)
        xprev, xi = xi, xi_new
    return rows, xi


# ── Linear systems ─────────────────────────────────────────────────────────────
# The matrix is transformed into reduced row echelon form (RREF) through pivoting and elimination.
def gauss_jordan(A):
    """Returns solution vector and step log."""
    n    = len(A)
    mat  = [row[:] for row in A]          # deep copy
    steps = []

    for col in range(n):
        pivot = mat[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError(f"Zero pivot at column {col+1}")
        mat[col] = [v/pivot for v in mat[col]]
        steps.append(("pivot", col, [row[:] for row in mat]))
        for row in range(n):
            if row != col:
                factor = mat[row][col]
                mat[row] = [mat[row][j] - factor*mat[col][j] for j in range(n+1)]
        steps.append(("elim", col, [row[:] for row in mat]))

    solution = [mat[i][n] for i in range(n)]
    return solution, steps
# This function performs LU Decomposition of matrix A into a lower triangular matrix L and an upper triangular matrix U.
# It then solves the system using forward substitution (Lc = b) and backward substitution (Ux = c).

def lu_decomposition(A):
    n   = len(A)
    mat = [row[:] for row in A]
    L   = [[1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]
    U   = [[0.0]*n for _ in range(n)]
    b   = [mat[i][n] for i in range(n)]

    for col in range(n):
        for row in range(col, n):
            U[col][row] = mat[col][row] - sum(L[col][k]*U[k][row] for k in range(col))
        for row in range(col+1, n):
            if abs(U[col][col]) < 1e-12:
                raise ValueError("Zero pivot encountered")
            L[row][col] = (mat[row][col] - sum(L[row][k]*U[k][col] for k in range(col))) / U[col][col]

    # forward Lc = b
    c = [0.0]*n
    for i in range(n):
        c[i] = b[i] - sum(L[i][k]*c[k] for k in range(i))

    # back Ux = c
    x = [0.0]*n
    for i in range(n-1, -1, -1):
        if abs(U[i][i]) < 1e-12:
            raise ValueError("Zero pivot in U")
        x[i] = (c[i] - sum(U[i][k]*x[k] for k in range(i+1, n))) / U[i][i]

    return x, L, U, c
# This function solves a linear system using Cramer's Rule. It computes the determinant of the main matrix and replaces columns with the constants vector to find each variable.
# This method works only when the determinant is non-zero.
def cramers_rule(A):
    n   = len(A)
    mat = [[A[i][j] for j in range(n)] for i in range(n)]
    b   = [A[i][n] for i in range(n)]

    def det3(M):
        if n == 2:
            return M[0][0]*M[1][1] - M[0][1]*M[1][0]
        # cofactor expansion (works for any n but we only have 3x3 here)
        d = 0
        for c in range(n):
            sub = [[M[r][cc] for cc in range(n) if cc != c] for r in range(1, n)]
            minor = sub[0][0]*sub[1][1] - sub[0][1]*sub[1][0] if n == 3 else sub[0][0]
            d += ((-1)**c) * M[0][c] * minor
        return d

    detA = det3(mat)
    if abs(detA) < 1e-12:
        raise ValueError("Determinant is zero — system has no unique solution")

    xs = []
    for i in range(n):
        Mi = [row[:] for row in mat]
        for r in range(n): Mi[r][i] = b[r]
        xs.append(det3(Mi) / detA)

    return xs, detA


# ── Plot ───────────────────────────────────────────────────────────────────────
# This function generates a graph of the function f(x).
# It plots the function over a specified range and highlights
# the computed root point on the graph.
# It also handles invalid values safely using NaN.
def plot_function(f, root, x_range=None):
    a = x_range[0] if x_range else root - 3
    b = x_range[1] if x_range else root + 3
    xs = np.linspace(a, b, 500)
    ys = []
    for xv in xs:
        try:    ys.append(f(xv))
        except: ys.append(np.nan)
    ys = np.array(ys, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(xs, ys, color='#A8D5C2', linewidth=2.2, label='f(x)')
    ax.axhline(0, color='#aaa', linewidth=0.8)
    ax.axvline(0, color='#aaa', linewidth=0.8)
    ax.scatter([root], [0], color='#C97D7D', s=90, zorder=5,
               label=f'Root = {round(root,6)}')
    ax.annotate(f'  x = {round(root,6)}', xy=(root, 0),
                color='#C97D7D', fontsize=9, va='bottom')
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=9)
    ax.set_xlabel('x'); ax.set_ylabel('f(x)')
    ax.set_title('Function Plot', fontsize=11, color='#3a3a3a')
    plt.tight_layout()
    return fig


def matrix_to_df(mat, n):
    cols = [f"x{i+1}" for i in range(n)] + ["b"]
    return pd.DataFrame(mat, columns=cols,
                        index=[f"Row {i+1}" for i in range(n)])


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="main-title">NM Solver</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Numerical Analysis</div>', unsafe_allow_html=True)
    st.markdown("---")

    chapter = st.radio("Chapter", ["Root Finding", "Linear Systems"])
    st.markdown("---")

    # ── Root Finding sidebar ───────────────────────────────────────────────────
    if chapter == "Root Finding":
        method = st.selectbox("Method", [
            "Bisection", "False Position",
            "Simple Fixed Point", "Newton-Raphson", "Secant"
        ])
        st.markdown(f'<div class="method-badge">{method}</div>', unsafe_allow_html=True)

        hints = {
            "Bisection":          "Example: 4*x**3 - 6*x**2 + 7*x - 2.3",
            "False Position":     "Example: -13 - 20*x + 19*x**2 - 3*x**3",
            "Simple Fixed Point": "Enter g(x) where x = g(x)\nExample: sqrt(1.8*x + 2.5)",
            "Newton-Raphson":     "Example: -2 + 6*x - 4*x**2 + 0.5*x**3",
            "Secant":             "Example: 2*x**3 - 11.7*x**2 + 17.7*x - 5",
        }
        defaults_eq = {
            "Bisection":          ("4*x**3 - 6*x**2 + 7*x - 2.3",      None),
            "False Position":     ("-13 - 20*x + 19*x**2 - 3*x**3",    None),
            "Simple Fixed Point": ("sqrt(1.8*x + 2.5)",                  None),
            "Newton-Raphson":     ("-2 + 6*x - 4*x**2 + 0.5*x**3",
                                   "6 - 8*x + 1.5*x**2"),
            "Secant":             ("2*x**3 - 11.7*x**2 + 17.7*x - 5",  None),
        }

        st.markdown('<div class="section-header">Equation</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-box">{hints[method]}</div>', unsafe_allow_html=True)
        label = "g(x) =" if method == "Simple Fixed Point" else "f(x) ="
        equation   = st.text_input(label, value=defaults_eq[method][0])
        fd_equation = None
        if method == "Newton-Raphson":
            fd_equation = st.text_input("f'(x) =", value=defaults_eq[method][1])

        st.markdown('<div class="section-header">Parameters</div>', unsafe_allow_html=True)
        xl = xu = x0 = xm1 = xi_val = None

        if method in ["Bisection", "False Position"]:
            c1, c2 = st.columns(2)
            with c1: xl = st.number_input("xl", value=0.0)
            with c2: xu = st.number_input("xu", value=1.0)
        elif method in ["Simple Fixed Point", "Newton-Raphson"]:
            x0 = st.number_input("x0 (initial guess)", value=0.5)
        else:
            c1, c2 = st.columns(2)
            with c1: xm1    = st.number_input("Xi-1", value=1.0)
            with c2: xi_val = st.number_input("Xi",   value=2.0)

        eps = st.number_input("Tolerance (eps %)", value=0.5,
                              min_value=0.0001, format="%.4f")
        show_graph = st.checkbox("Show function graph", value=True)
        plot_a = plot_b = None
        if show_graph and method in ["Bisection", "False Position"]:
            c1, c2 = st.columns(2)
            with c1: plot_a = st.number_input("Graph from", value=float(xl)-1 if xl is not None else -5.0)
            with c2: plot_b = st.number_input("Graph to",   value=float(xu)+1 if xu is not None else  5.0)

    # ── Linear Systems sidebar ─────────────────────────────────────────────────
    else:
        ls_method = st.selectbox("Method", [
            "Gauss-Jordan Elimination",
            "LU Decomposition",
            "Cramer's Rule"
        ])
        st.markdown(f'<div class="method-badge">{ls_method}</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Matrix Size</div>', unsafe_allow_html=True)
        n_size = st.selectbox("System size (n x n)", [2, 3, 4], index=1)
        st.markdown('<div class="section-header">Enter Augmented Matrix [A|b]</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="info-box">Fill each row: coefficients then constant b</div>',
                    unsafe_allow_html=True)

        matrix_input = []
        for i in range(n_size):
            cols_in = st.columns(n_size + 1)
            row = []
            for j in range(n_size):
                with cols_in[j]:
                    row.append(st.number_input(f"a{i+1}{j+1}", value=0.0,
                                               key=f"m{i}{j}", label_visibility="visible"))
            with cols_in[n_size]:
                row.append(st.number_input(f"b{i+1}", value=0.0,
                                           key=f"b{i}", label_visibility="visible"))
            matrix_input.append(row)

    st.markdown("---")
    solve_btn = st.button("Solve")


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Numerical Methods Solver</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Root Finding  |  Linear Systems</div>', unsafe_allow_html=True)
st.markdown("---")

if not solve_btn:
    st.markdown("""
    <div class="info-box">
    Select a chapter and method from the sidebar, fill in the parameters, then click <b>Solve</b>.<br><br>
    Supported functions: <code>sqrt</code>, <code>sin</code>, <code>cos</code>, <code>tan</code>,
    <code>log</code>, <code>log10</code>, <code>exp</code>, <code>pi</code>, <code>e</code>
    </div>
    """, unsafe_allow_html=True)

else:
    try:
        # ── Root Finding ───────────────────────────────────────────────────────
        if chapter == "Root Finding":
            f = make_f(equation); _ = f(1.0)
            rows = root = None

            if method == "Bisection":
                if f(xl)*f(xu) > 0:
                    st.error("f(xl) and f(xu) have the same sign — choose a different interval.")
                    st.stop()
                rows, root = bisection(f, xl, xu, eps)

            elif method == "False Position":
                if f(xl)*f(xu) > 0:
                    st.error("f(xl) and f(xu) have the same sign — choose a different interval.")
                    st.stop()
                rows, root = false_position(f, xl, xu, eps)

            elif method == "Simple Fixed Point":
                rows, root = fixed_point(f, x0, eps)

            elif method == "Newton-Raphson":
                fd = make_f(fd_equation); _ = fd(1.0)
                rows, root = newton_raphson(f, fd, x0, eps)

            elif method == "Secant":
                rows, root = secant(f, xm1, xi_val, eps)

            # result cards
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="result-box"><div class="result-label">Root</div>'
                            f'<div class="result-value">{round(root,8)}</div></div>',
                            unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="result-box"><div class="result-label">f(root)</div>'
                            f'<div class="result-value">{round(f(root),8)}</div></div>',
                            unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="result-box"><div class="result-label">Iterations</div>'
                            f'<div class="result-value">{len(rows)}</div></div>',
                            unsafe_allow_html=True)

            if show_graph:
                xr = (plot_a, plot_b) if (plot_a is not None and plot_b is not None) else None
                st.pyplot(plot_function(f, root, xr))

            st.markdown('<div class="section-header">Iteration Table</div>',
                        unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Linear Systems ─────────────────────────────────────────────────────
        else:
            A = matrix_input
            n = n_size

            st.markdown('<div class="section-header">Input Matrix [A|b]</div>',
                        unsafe_allow_html=True)
            st.dataframe(matrix_to_df(A, n), use_container_width=True)

            if ls_method == "Gauss-Jordan Elimination":
                solution, steps = gauss_jordan(A)
                c_cols = st.columns(n)
                for i, val in enumerate(solution):
                    with c_cols[i]:
                        st.markdown(
                            f'<div class="result-box"><div class="result-label">x{i+1}</div>'
                            f'<div class="result-value">{round(val,6)}</div></div>',
                            unsafe_allow_html=True)

            elif ls_method == "LU Decomposition":
                solution, L, U, c = lu_decomposition(A)
                st.markdown('<div class="section-header">L Matrix</div>',
                            unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(L,
                    columns=[f"x{i+1}" for i in range(n)],
                    index=[f"Row {i+1}" for i in range(n)]).round(6),
                    use_container_width=True)

                st.markdown('<div class="section-header">U Matrix</div>',
                            unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(U,
                    columns=[f"x{i+1}" for i in range(n)],
                    index=[f"Row {i+1}" for i in range(n)]).round(6),
                    use_container_width=True)

                st.markdown('<div class="section-header">c vector  (Lc = b)</div>',
                            unsafe_allow_html=True)
                st.dataframe(pd.DataFrame({"c": [round(v,6) for v in c]}),
                             use_container_width=True, hide_index=False)

                st.markdown('<div class="section-header">Solution  (Ux = c)</div>',
                            unsafe_allow_html=True)
                c_cols = st.columns(n)
                for i, val in enumerate(solution):
                    with c_cols[i]:
                        st.markdown(
                            f'<div class="result-box"><div class="result-label">x{i+1}</div>'
                            f'<div class="result-value">{round(val,6)}</div></div>',
                            unsafe_allow_html=True)

            elif ls_method == "Cramer's Rule":
                solution, detA = cramers_rule(A)
                st.markdown(f'<div class="info-box">det(A) = <b>{round(detA,6)}</b></div>',
                            unsafe_allow_html=True)
                c_cols = st.columns(n)
                for i, val in enumerate(solution):
                    with c_cols[i]:
                        st.markdown(
                            f'<div class="result-box"><div class="result-label">x{i+1}</div>'
                            f'<div class="result-value">{round(val,6)}</div></div>',
                            unsafe_allow_html=True)

    except ZeroDivisionError:
        st.error("Division by zero — try different values.")
    except Exception as ex:
        st.error(f"Error: {ex}")

st.markdown("---")
st.caption("Numerical Analysis | Root Finding  ·  Linear Systems")
