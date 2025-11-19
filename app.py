import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional, List, Dict, Any

app = FastAPI(title="Zero of Functions (ZOF) Solver")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -------------------------- Numerical Methods --------------------------

def bisection(f, a: float, b: float, tol: float = 1e-6, max_iter: int = 100):
    if f(a) * f(b) >= 0:
        return None, "f(a) and f(b) must have opposite signs"
    
    iterations = []
    iter_count = 0
    while (b - a) > tol and iter_count < max_iter:
        c = (a + b) / 2
        fc = f(c)
        error = abs(b - a) / 2
        iterations.append({"n": iter_count+1, "a": a, "b": b, "c": c, "f(c)": fc, "error": error})
        if abs(fc) < tol:
            return c, iterations
        if f(a) * fc < 0:
            b = c
        else:
            a = c
        iter_count += 1
    return (a + b)/2, iterations

def regula_falsi(f, a: float, b: float, tol: float = 1e-6, max_iter: int = 100):
    if f(a) * f(b) >= 0:
        return None, "f(a) and f(b) must have opposite signs"
    
    iterations = []
    iter_count = 0
    while iter_count < max_iter:
        c = b - f(b) * (b - a) / (f(b) - f(a))
        fc = f(c)
        error = abs(f(c))
        iterations.append({"n": iter_count+1, "a": a, "b": b, "c": c, "f(c)": fc, "error": error})
        if error < tol:
            return c, iterations
        if f(a) * fc < 0:
            b = c
        else:
            a = c
        iter_count += 1
    return c, iterations

def secant(f, x0: float, x1: float, tol: float = 1e-6, max_iter: int = 100):
    iterations = []
    iter_count = 0
    while iter_count < max_iter:
        if f(x1) - f(x0) == 0:
            return None, "Division by zero"
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        error = abs(x2 - x1)
        iterations.append({"n": iter_count+1, "x0": x0, "x1": x1, "x2": x2, "f(x2)": f(x2), "error": error})
        if error < tol:
            return x2, iterations
        x0, x1 = x1, x2
        iter_count += 1
    return x2, iterations

def newton_raphson(f, df, x0: float, tol: float = 1e-6, max_iter: int = 100):
    iterations = []
    x = x0
    iter_count = 0
    while iter_count < max_iter:
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            return None, "Derivative is zero"
        x_new = x - fx / dfx
        error = abs(x_new - x)
        iterations.append({"n": iter_count+1, "x": x, "f(x)": fx, "f'(x)": dfx, "x_new": x_new, "error": error})
        if error < tol:
            return x_new, iterations
        x = x_new
        iter_count += 1
    return x, iterations

def fixed_point(g, x0: float, tol: float = 1e-6, max_iter: int = 100):
    iterations = []
    x = x0
    iter_count = 0
    while iter_count < max_iter:
        x_new = g(x)
        error = abs(x_new - x)
        iterations.append({"n": iter_count+1, "x": x, "g(x)": x_new, "error": error})
        if error < tol:
            return x_new, iterations
        x = x_new
        iter_count += 1
    return x, iterations

def modified_secant(f, x0: float, delta: float = 0.01, tol: float = 1e-6, max_iter: int = 100):
    iterations = []
    x = x0
    iter_count = 0
    while iter_count < max_iter:
        f_x = f(x)
        f_xd = f(x + delta * x)
        if (f_xd - f_x) == 0:
            return None, "Division by zero"
        x_new = x - delta * x * f_x / (f_xd - f_x)
        error = abs(x_new - x)
        iterations.append({"n": iter_count+1, "x": x, "f(x)": f_x, "x_new": x_new, "error": error})
        if error < tol:
            return x_new, iterations
        x = x_new
        iter_count += 1
    return x, iterations

# -------------------------- Helper to build function --------------------------

def build_function(coeffs: List[float]) -> callable:
    coeffs = [float(c) for c in coeffs[::-1]]  # reverse to match poly format
    def f(x):
        return np.polyval(coeffs, x)
    return f

def build_derivative(coeffs: List[float]) -> callable:
    coeffs = [float(c) for c in coeffs[::-1]]
    deriv_coeffs = np.polyder(coeffs)
    def df(x):
        return np.polyval(deriv_coeffs, x)
    return df

# -------------------------- Routes --------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/solve", response_class=HTMLResponse)
async def solve(
    request: Request,
    method: str = Form(...),
    coeffs: str = Form(...),           # e.g., "1, -5, 6" for x² -5x +6
    a: Optional[float] = Form(None),
    b: Optional[float] = Form(None),
    x0: Optional[float] = Form(None),
    x1: Optional[float] = Form(None),
    delta: Optional[float] = Form(0.01),
    tol: float = Form(1e-6),
    max_iter: int = Form(100)
):
    try:
        coeff_list = [float(c.strip()) for c in coeffs.split(",")]
        f = build_function(coeff_list)
        
        result = None
        iterations = []
        error_msg = None

        if method == "bisection":
            if a is None or b is None:
                error_msg = "Bisection requires a and b"
            else:
                result, iterations = bisection(f, a, b, tol, max_iter)
                if isinstance(iterations, str):
                    error_msg = iterations
                    iterations = []

        elif method == "regula_falsi":
            if a is None or b is None:
                error_msg = "Regula Falsi requires a and b"
            else:
                result, iterations = regula_falsi(f, a, b, tol, max_iter)
                if isinstance(iterations, str):
                    error_msg = iterations

        elif method == "secant":
            if x0 is None or x1 is None:
                error_msg = "Secant requires x0 and x1"
            else:
                result, iterations = secant(f, x0, x1, tol, max_iter)
                if isinstance(iterations, str):
                    error_msg = iterations

        elif method == "newton":
            df = build_derivative(coeff_list)
            if x0 is None:
                error_msg = "Newton-Raphson requires x0"
            else:
                result, iterations = newton_raphson(f, df, x0, tol, max_iter)
                if isinstance(iterations, str):
                    error_msg = iterations

        elif method == "fixed_point":
            # For fixed-point, we convert f(x)=0 → x = g(x)
            # Here we use g(x) = x - k*f(x) with small k (Steffensen-like)
            k = 0.01
            def g(x): return x - k * f(x)
            if x0 is None:
                error_msg = "Fixed Point requires x0"
            else:
                result, iterations = fixed_point(g, x0, tol, max_iter)

        elif method == "modified_secant":
            if x0 is None:
                error_msg = "Modified Secant requires x0"
            else:
                result, iterations = modified_secant(f, x0, delta, tol, max_iter)
                if isinstance(iterations, str):
                    error_msg = iterations

        return templates.TemplateResponse("index.html", {
            "request": request,
            "method": method,
            "coeffs": coeffs,
            "result": result,
            "iterations": iterations,
            "error_msg": error_msg,
            "final_error": iterations[-1]["error"] if iterations and "error" in iterations[-1] else None,
            "num_iters": len(iterations)
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error_msg": f"Error: {str(e)}"
        })