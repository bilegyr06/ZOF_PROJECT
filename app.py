from flask import Flask, render_template, request, jsonify
import ZOF_CLI as solver
import traceback

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    try:
        data = request.json
        method = data.get('method')
        func_str = data.get('function')
        tol = float(data.get('tolerance'))
        max_iter = int(data.get('max_iter'))
        
        result = {}
        
        if method == 'bisection':
            a = float(data.get('a'))
            b = float(data.get('b'))
            result = solver.bisection_method(func_str, a, b, tol, max_iter)
            
        elif method == 'regula_falsi':
            a = float(data.get('a'))
            b = float(data.get('b'))
            result = solver.regula_falsi_method(func_str, a, b, tol, max_iter)
            
        elif method == 'secant':
            x0 = float(data.get('x0'))
            x1 = float(data.get('x1'))
            result = solver.secant_method(func_str, x0, x1, tol, max_iter)
            
        elif method == 'newton':
            x0 = float(data.get('x0'))
            result = solver.newton_raphson_method(func_str, x0, tol, max_iter)
            
        elif method == 'fixed_point':
            x0 = float(data.get('x0'))
            result = solver.fixed_point_iteration(func_str, x0, tol, max_iter)
            
        elif method == 'modified_secant':
            x0 = float(data.get('x0'))
            delta = float(data.get('delta'))
            result = solver.modified_secant_method(func_str, x0, delta, tol, max_iter)
            
        else:
            return jsonify({"error": "Invalid method selected"}), 400

        if "error" in result:
            return jsonify({"error": result["error"]}), 400
            
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
