import math
import sympy as sp
import pandas as pd
import sys

# --- Core Numerical Methods ---

def evaluate_function(func_str, x_val):
    """Evaluates a mathematical function string at a given x."""
    try:
        x = sp.symbols('x')
        # Use sympy to parse and evaluate for better safety and power
        expr = sp.sympify(func_str)
        return float(expr.subs(x, x_val))
    except Exception as e:
        raise ValueError(f"Error evaluating function: {e}")

def get_derivative(func_str):
    """Calculates the derivative of the function string."""
    x = sp.symbols('x')
    expr = sp.sympify(func_str)
    diff_expr = sp.diff(expr, x)
    return str(diff_expr)

def bisection_method(func_str, a, b, tol, max_iter):
    results = []
    
    fa = evaluate_function(func_str, a)
    fb = evaluate_function(func_str, b)
    
    if fa * fb >= 0:
        return {"error": "Bisection method fails: f(a) and f(b) must have opposite signs."}

    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = evaluate_function(func_str, c)
        error = abs(b - a) # Or abs(c - old_c) if we tracked it, but interval width is standard for bisection
        
        results.append({
            "Iteration": i,
            "a": a,
            "b": b,
            "c (Root Est)": c,
            "f(c)": fc,
            "Error": error
        })
        
        if abs(fc) < tol or error < tol:
            break
        
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
            
    return {"results": results, "root": c, "final_error": error, "iterations": i}

def regula_falsi_method(func_str, a, b, tol, max_iter):
    results = []
    
    fa = evaluate_function(func_str, a)
    fb = evaluate_function(func_str, b)
    
    if fa * fb >= 0:
        return {"error": "Regula Falsi method fails: f(a) and f(b) must have opposite signs."}

    c = a # Initialize c
    for i in range(1, max_iter + 1):
        # c = b - (f(b) * (a - b)) / (f(a) - f(b))
        if abs(fa - fb) < 1e-12: # Avoid division by zero
             return {"error": "Division by zero in Regula Falsi."}
             
        c = (a * fb - b * fa) / (fb - fa)
        fc = evaluate_function(func_str, c)
        error = abs(fc) # For RF, error is often estimated by |f(c)| or |c_new - c_old|
        
        results.append({
            "Iteration": i,
            "a": a,
            "b": b,
            "c (Root Est)": c,
            "f(c)": fc,
            "Error": error
        })
        
        if abs(fc) < tol:
            break
        
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
            
    return {"results": results, "root": c, "final_error": error, "iterations": i}

def secant_method(func_str, x0, x1, tol, max_iter):
    results = []
    
    for i in range(1, max_iter + 1):
        f0 = evaluate_function(func_str, x0)
        f1 = evaluate_function(func_str, x1)
        
        if abs(f1 - f0) < 1e-12:
            return {"error": "Division by zero in Secant Method."}
            
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        error = abs(x2 - x1)
        
        results.append({
            "Iteration": i,
            "x(i-1)": x0,
            "x(i)": x1,
            "x(i+1) (Root Est)": x2,
            "f(x2)": evaluate_function(func_str, x2),
            "Error": error
        })
        
        if error < tol:
            return {"results": results, "root": x2, "final_error": error, "iterations": i}
            
        x0 = x1
        x1 = x2
        
    return {"results": results, "root": x2, "final_error": error, "iterations": max_iter}

def newton_raphson_method(func_str, x0, tol, max_iter):
    results = []
    deriv_str = get_derivative(func_str)
    
    for i in range(1, max_iter + 1):
        f0 = evaluate_function(func_str, x0)
        df0 = evaluate_function(deriv_str, x0)
        
        if abs(df0) < 1e-12:
            return {"error": "Derivative is zero. Newton-Raphson fails."}
            
        x1 = x0 - f0 / df0
        error = abs(x1 - x0)
        
        results.append({
            "Iteration": i,
            "x(i)": x0,
            "f(x)": f0,
            "f'(x)": df0,
            "x(i+1) (Root Est)": x1,
            "Error": error
        })
        
        if error < tol:
            return {"results": results, "root": x1, "final_error": error, "iterations": i}
            
        x0 = x1
        
    return {"results": results, "root": x1, "final_error": error, "iterations": max_iter}

def fixed_point_iteration(func_str, x0, tol, max_iter):
    # Note: User must provide g(x) such that x = g(x)
    # Or we assume func_str is f(x) and we try x + f(x)? 
    # Standard requirement usually implies user gives g(x).
    # However, for a general solver, usually we ask for f(x)=0.
    # But FPI specifically needs x = g(x).
    # I will assume the input 'func_str' IS g(x).
    results = []
    
    for i in range(1, max_iter + 1):
        x1 = evaluate_function(func_str, x0)
        error = abs(x1 - x0)
        
        results.append({
            "Iteration": i,
            "x(i)": x0,
            "g(x)": x1,
            "Error": error
        })
        
        if error < tol:
            return {"results": results, "root": x1, "final_error": error, "iterations": i}
            
        x0 = x1
        
    return {"results": results, "root": x1, "final_error": error, "iterations": max_iter}

def modified_secant_method(func_str, x0, delta, tol, max_iter):
    results = []
    
    for i in range(1, max_iter + 1):
        f0 = evaluate_function(func_str, x0)
        f_delta = evaluate_function(func_str, x0 + delta * x0)
        
        denom = f_delta - f0
        if abs(denom) < 1e-12:
             return {"error": "Division by zero in Modified Secant Method."}

        x1 = x0 - (delta * x0 * f0) / denom
        error = abs(x1 - x0)
        
        results.append({
            "Iteration": i,
            "x(i)": x0,
            "x(i+1) (Root Est)": x1,
            "f(x)": f0,
            "Error": error
        })
        
        if error < tol:
             return {"results": results, "root": x1, "final_error": error, "iterations": i}
        
        x0 = x1
        
    return {"results": results, "root": x1, "final_error": error, "iterations": max_iter}


# --- CLI Interface ---

def main():
    print("========================================")
    print("   ZOF Solver - Zero of Functions CLI   ")
    print("========================================")
    
    while True:
        print("\nSelect Method:")
        print("1. Bisection Method")
        print("2. Regula Falsi Method")
        print("3. Secant Method")
        print("4. Newton-Raphson Method")
        print("5. Fixed Point Iteration")
        print("6. Modified Secant Method")
        print("0. Exit")
        
        choice = input("Enter choice (0-6): ").strip()
        
        if choice == '0':
            print("Exiting...")
            break
            
        if choice not in ['1', '2', '3', '4', '5', '6']:
            print("Invalid choice. Please try again.")
            continue
            
        try:
            func_str = input("Enter function (e.g., x**2 - 4): ").strip()
            tol = float(input("Enter tolerance (e.g., 1e-6): ").strip())
            max_iter = int(input("Enter max iterations (e.g., 100): ").strip())
            
            result = None
            
            if choice == '1': # Bisection
                a = float(input("Enter start of interval (a): "))
                b = float(input("Enter end of interval (b): "))
                result = bisection_method(func_str, a, b, tol, max_iter)
                
            elif choice == '2': # Regula Falsi
                a = float(input("Enter start of interval (a): "))
                b = float(input("Enter end of interval (b): "))
                result = regula_falsi_method(func_str, a, b, tol, max_iter)
                
            elif choice == '3': # Secant
                x0 = float(input("Enter first guess (x0): "))
                x1 = float(input("Enter second guess (x1): "))
                result = secant_method(func_str, x0, x1, tol, max_iter)
                
            elif choice == '4': # Newton-Raphson
                x0 = float(input("Enter initial guess (x0): "))
                result = newton_raphson_method(func_str, x0, tol, max_iter)
                
            elif choice == '5': # Fixed Point
                print("NOTE: For Fixed Point, enter g(x) such that x = g(x).")
                x0 = float(input("Enter initial guess (x0): "))
                result = fixed_point_iteration(func_str, x0, tol, max_iter)
                
            elif choice == '6': # Modified Secant
                x0 = float(input("Enter initial guess (x0): "))
                delta = float(input("Enter perturbation delta (e.g., 0.01): "))
                result = modified_secant_method(func_str, x0, delta, tol, max_iter)
            
            # Output
            if "error" in result:
                print(f"\nERROR: {result['error']}")
            else:
                print("\n--- Iteration Results ---")
                df = pd.DataFrame(result['results'])
                print(df.to_string(index=False))
                print("\n--- Final Result ---")
                print(f"Root: {result['root']}")
                print(f"Final Error: {result['final_error']}")
                print(f"Iterations: {result['iterations']}")
                
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
