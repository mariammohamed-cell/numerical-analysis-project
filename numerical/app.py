import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────
st.set_page_config(page_title="Numerical Methods Dashboard", layout="wide")

# ─────────────────────────────────────────
# UI STYLE (DASHBOARD)
# ─────────────────────────────────────────
st.markdown("""
<style>

html, body, .stApp {
    background-color: #F5F0E8 !important;
    color: #3a3a3a !important;
}

header, [data-testid="stHeader"] {
    background-color: #F5F0E8 !important;
}

[data-testid="stSidebar"] {
    background-color: #D4EAE3 !important;
    border-right: 2px solid #A8D5C2;
}

.stButton > button {
    background-color: #A8D5C2 !important;
    color: #3a3a3a !important;
    border-radius: 10px !important;
    width: 100%;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #8fc4af !important;
}

.card {
    background: #D4EAE3;
    padding: 12px;
    border-radius: 12px;
    margin: 10px 0px;
    border: 1px solid #A8D5C2;
}

.title {
    font-size: 26px;
    font-weight: bold;
}

.subtitle {
    font-size: 13px;
    color: #666;
}

.result {
    font-size: 22px;
    font-weight: bold;
    color: #C97D7D;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────
st.markdown("<div class='title'>Numerical Methods Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Root Finding + Linear Systems Solver</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# FUNCTION PARSER
# ─────────────────────────────────────────
def make_f(expr):
    safe = {
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "log": math.log, "exp": math.exp,
        "pi": math.pi, "e": math.e
    }
    def f(x):
        g = safe.copy()
        g["x"] = x
        return eval(expr, g)
    return f

# ─────────────────────────────────────────
# ROOT METHODS
# ─────────────────────────────────────────
def bisection(f, xl, xu, eps):
    rows, xr, old = [], 0, 0
    for i in range(100):
        old = xr
        xr = (xl + xu)/2
        err = abs((xr-old)/xr)*100 if i>0 else None
        rows.append({"Iter":i,"xr":xr,"Error %":err})
        if err and err<=eps: break
        if f(xl)*f(xr)>0: xl=xr
        else: xu=xr
    return rows, xr

def false_position(f, xl, xu, eps):
    rows, xr, old = [], 0, 0
    for i in range(100):
        old = xr
        xr = xu - f(xu)*(xl-xu)/(f(xl)-f(xu))
        err = abs((xr-old)/xr)*100 if i>0 else None
        rows.append({"Iter":i,"xr":xr,"Error %":err})
        if err and err<=eps: break
        if f(xl)*f(xr)>0: xl=xr
        else: xu=xr
    return rows, xr

def fixed_point(f, x0, eps):
    rows=[]
    for i in range(100):
        x1 = f(x0)
        err = abs((x1-x0)/x1)*100 if i>0 else None
        rows.append({"Iter":i,"x":x1,"Error %":err})
        if err and err<=eps: break
        x0=x1
    return rows, x1

def newton(f, df, x0, eps):
    rows=[]
    for i in range(100):
        x1 = x0 - f(x0)/df(x0)
        err = abs((x1-x0)/x1)*100 if i>0 else None
        rows.append({"Iter":i,"x":x1,"Error %":err})
        if err and err<=eps: break
        x0=x1
    return rows, x1

def secant(f, x0, x1, eps):
    rows=[]
    for i in range(100):
        x2 = x1 - f(x1)*(x0-x1)/(f(x0)-f(x1))
        err = abs((x2-x1)/x2)*100 if i>0 else None
        rows.append({"Iter":i,"x":x2,"Error %":err})
        if err and err<=eps: break
        x0,x1=x1,x2
    return rows, x2

# ─────────────────────────────────────────
# LINEAR METHODS (SAFE)
# ─────────────────────────────────────────
def gauss_elimination(A):
    a = np.array(A,float)
    n=len(a)

    for i in range(n):
        for j in range(i+1,n):
            if abs(a[i][i]) < 1e-12:
                continue
            factor=a[j][i]/a[i][i]
            a[j]=a[j]-factor*a[i]

    x=np.zeros(n)

    for i in range(n-1,-1,-1):
        if abs(a[i][i]) < 1e-12:
            x[i]=0
            continue
        sum_val = sum(a[i][j]*x[j] for j in range(i+1,n))
        x[i]=(a[i][-1]-sum_val)/a[i][i]

    return x

def gauss_jordan(A):
    a=np.array(A,float)
    n=len(a)

    for i in range(n):
        if abs(a[i][i]) < 1e-12:
            continue
        a[i]=a[i]/a[i][i]
        for j in range(n):
            if i!=j:
                a[j]-=a[j][i]*a[i]

    return a[:,-1]

def lu(A):
    A=np.array(A,float)
    n=len(A)

    L=np.eye(n)
    U=np.zeros_like(A)

    for i in range(n):
        for j in range(i,n):
            U[i][j]=A[i][j]-sum(L[i][k]*U[k][j] for k in range(i))
        for j in range(i+1,n):
            if abs(U[i][i]) < 1e-12:
                continue
            L[j][i]=(A[j][i]-sum(L[j][k]*U[k][i] for k in range(i)))/U[i][i]

    return L,U

def cramer(A):
    A=np.array(A,float)
    coef=A[:,:-1]
    b=A[:,-1]

    detA=np.linalg.det(coef)
    if abs(detA)<1e-12:
        return [0]*len(coef)

    x=[]
    for i in range(len(coef)):
        temp=coef.copy()
        temp[:,i]=b
        x.append(np.linalg.det(temp)/detA)

    return x

# ─────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────
def plot_function(f, root):
    xs=np.linspace(root-3,root+3,400)
    ys=[f(x) for x in xs]

    fig,ax=plt.subplots()
    ax.plot(xs,ys)
    ax.axhline(0)
    ax.scatter(root,0)

    return fig

# ─────────────────────────────────────────
# UI NAVIGATION
# ─────────────────────────────────────────
chapter = st.sidebar.selectbox("Choose Chapter", ["Root Finding","Linear Systems"])

# ─────────────────────────────────────────
# ROOT FINDING UI
# ─────────────────────────────────────────
if chapter=="Root Finding":

    method = st.selectbox("Method",
        ["Bisection","False Position","Fixed Point","Newton","Secant"])

    eq = st.text_input("f(x) =", "x**3 - x - 1")
    f = make_f(eq)

    if method in ["Bisection","False Position"]:
        xl = st.number_input("xl",1.0)
        xu = st.number_input("xu",2.0)

    elif method=="Fixed Point":
        x0 = st.number_input("x0",1.0)

    elif method=="Newton":
        x0 = st.number_input("x0",1.0)
        df = make_f(st.text_input("f'(x)"))

    else:
        x0 = st.number_input("x0",1.0)
        x1 = st.number_input("x1",2.0)

    eps = st.number_input("Tolerance %",0.01)

    if st.button("Solve"):

        if method=="Bisection":
            rows,root=bisection(f,xl,xu,eps)
        elif method=="False Position":
            rows,root=false_position(f,xl,xu,eps)
        elif method=="Fixed Point":
            rows,root=fixed_point(f,x0,eps)
        elif method=="Newton":
            rows,root=newton(f,df,x0,eps)
        else:
            rows,root=secant(f,x0,x1,eps)

        st.markdown(f"<div class='card'>Root = <div class='result'>{round(root,6)}</div></div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><b>Graph</b></div>", unsafe_allow_html=True)
        st.pyplot(plot_function(f,root))

        st.markdown("<div class='card'><b>Iteration Table</b></div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows),use_container_width=True)

# ─────────────────────────────────────────
# LINEAR SYSTEMS UI
# ─────────────────────────────────────────
else:

    method = st.selectbox("Method",
        ["Gauss Elimination","Gauss Jordan","LU","Cramer"])

    n=st.selectbox("Size",[2,3])
    A=[]

    for i in range(n):
        row=[]
        cols=st.columns(n+1)
        for j in range(n):
            row.append(cols[j].number_input(f"a{i}{j}",key=f"a{i}{j}"))
        row.append(cols[n].number_input(f"b{i}",key=f"b{i}"))
        A.append(row)

    if st.button("Solve"):

        if method=="Gauss Elimination":
            sol=gauss_elimination(A)
        elif method=="Gauss Jordan":
            sol=gauss_jordan(A)
        elif method=="LU":
            L,U=lu(A)
            st.write("L",L)
            st.write("U",U)
            sol=None
        else:
            sol=cramer(A)

        if sol is not None:
            for i,v in enumerate(sol):
                st.markdown(f"<div class='card'>x{i+1} = <b>{round(v,6)}</b></div>", unsafe_allow_html=True)