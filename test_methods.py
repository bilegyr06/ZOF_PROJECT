import ZOF_CLI as solver
import math

def test_methods():
    print("Running Tests...")
    
    # Test Case: x^2 - 4 = 0, Root = 2
    func = "x**2 - 4"
    
    print("\n--- Bisection ---")
    res = solver.bisection_method(func, 0, 3, 1e-5, 100)
    print(f"Root: {res.get('root')}, Error: {res.get('final_error')}")
    
    print("\n--- Regula Falsi ---")
    res = solver.regula_falsi_method(func, 0, 3, 1e-5, 100)
    print(f"Root: {res.get('root')}, Error: {res.get('final_error')}")
    
    print("\n--- Secant ---")
    res = solver.secant_method(func, 0, 3, 1e-5, 100)
    print(f"Root: {res.get('root')}, Error: {res.get('final_error')}")
    
    print("\n--- Newton-Raphson ---")
    res = solver.newton_raphson_method(func, 1, 1e-5, 100)
    print(f"Root: {res.get('root')}, Error: {res.get('final_error')}")
    
    print("\n--- Modified Secant ---")
    res = solver.modified_secant_method(func, 1, 0.01, 1e-5, 100)
    print(f"Root: {res.get('root')}, Error: {res.get('final_error')}")

    # Fixed Point: x = g(x) -> x = sqrt(4) -> g(x) = 4/x ? No, x^2=4 -> x = 4/x is unstable.
    # Try g(x) = 2 + 0.5*(x - 2) ? 
    # Let's try a simple one: x^2 - x - 2 = 0 -> x = x^2 - 2 (diverges). x = sqrt(x+2).
    print("\n--- Fixed Point (x = sqrt(x+2), Root=2) ---")
    res = solver.fixed_point_iteration("sqrt(x+2)", 1, 1e-5, 100)
    print(f"Root: {res.get('root')}, Error: {res.get('final_error')}")

if __name__ == "__main__":
    test_methods()
