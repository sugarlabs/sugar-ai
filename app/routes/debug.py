"""
Lightweight Python debugging logic for Sugar-AI.
Performs static analysis using AST and restricted execution using a custom environment.
"""
import ast
import traceback
import sys
import io
import time
import signal
from typing import Dict, Any, Optional, List

class Debugger:
    """Hardened debugger that analyzes and executes Python code safely."""

    # Allowed AST node types (Safe subset of Python)
    ALLOWED_NODES = {
        ast.Module, ast.Expr, ast.Assign, ast.AugAssign,
        ast.Name, ast.Load, ast.Store,
        ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
        ast.List, ast.Tuple, ast.Dict, ast.Set,
        ast.Constant, ast.Call, ast.Attribute,
        ast.Subscript, ast.Index, ast.Slice,
        ast.For, ast.While, ast.If,
        ast.Pass, ast.Break, ast.Continue,
        # Comprehensions and modern string formatting
        ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp,
        ast.comprehension, ast.JoinedStr, ast.FormattedValue,
        # Loaders / Contexts
        ast.Load, ast.Store, ast.Del,
        # Operators
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
        ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd, ast.FloorDiv,
        ast.UAdd, ast.USub, ast.Not, ast.Invert,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.Is, ast.IsNot, ast.In, ast.NotIn,
        ast.And, ast.Or
    }

    # Safe built-in functions
    SAFE_BUILTINS = {
        'print': print, 'len': len, 'range': range,
        'str': str, 'int': int, 'float': float, 'bool': bool,
        'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
        'abs': abs, 'sum': sum, 'min': min, 'max': max,
        'enumerate': enumerate, 'zip': zip, 'any': any, 'all': all,
        'sorted': sorted, 'reversed': reversed, 'round': round,
        'Exception': Exception, 'ValueError': ValueError, 
        'TypeError': TypeError, 'IndexError': IndexError, 'KeyError': KeyError,
    }

    @staticmethod
    def analyze_security(code: str) -> List[str]:
        """Strict whitelist-based AST analysis."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [f"Syntax Error at line {e.lineno}: {e.msg}"]

        dangers = []
        for node in ast.walk(tree):
            # 1. Check if node type is allowed
            if type(node) not in Debugger.ALLOWED_NODES:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    dangers.append("Imports are strictly prohibited.")
                else:
                    dangers.append(f"Disallowed code structure detected: {type(node).__name__}")
                continue

            # 2. Block private/magic attribute access
            if isinstance(node, ast.Attribute):
                if node.attr.startswith('_'):
                    dangers.append(f"Access to private/magic attribute '{node.attr}' is blocked.")

            # 3. Block access to dangerous names
            if isinstance(node, ast.Name):
                name_id = getattr(node, 'id', '')
                if name_id.startswith('_') or name_id in ['eval', 'exec', 'open', 'getattr', 'setattr', 'help', 'copyright']:
                    dangers.append(f"Access to restricted name '{name_id}' is blocked.")

            # 4. Deep check for Call nodes
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    func_id = getattr(func, 'id', '')
                    if func_id not in Debugger.SAFE_BUILTINS:
                        dangers.append(f"Function call to '{func_id}' is blocked.")
                elif isinstance(func, ast.Attribute):
                    if func.attr.startswith('_'):
                        dangers.append(f"Method call to '{func.attr}' is blocked.")
        
        return list(dict.fromkeys(dangers))

    @staticmethod
    def execute_safely(code: str, timeout: int = 2) -> Dict[str, Any]:
        """Execute code in a heavily restricted environment."""
        restricted_globals = {
            "__builtins__": Debugger.SAFE_BUILTINS
        }
        
        stdout_capture = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = stdout_capture
        
        # Explicit typing for result dict to satisfy linter
        result: Dict[str, Any] = {
            "success": True, 
            "output": "", 
            "error": None, 
            "suggestions": []
        }

        try:
            def handle_timeout(signum: Any, frame: Any) -> None:
                raise TimeoutError("Execution exceeded the 2-second security limit.")

            is_unix = hasattr(signal, 'SIGALRM')
            if is_unix:
                signal.signal(signal.SIGALRM, handle_timeout)
                signal.alarm(timeout)

            restricted_locals: Dict[str, Any] = {}
            exec(code, restricted_globals, restricted_locals)
            
            if is_unix:
                signal.alarm(0)

        except TimeoutError as e:
            result["success"] = False
            result["error"] = str(e)
            result["suggestions"] = ["Optimization tip: Avoid infinite loops or large data processing."]
        except Exception as e:
            result["success"] = False
            result["error"] = f"{type(e).__name__}: {str(e)}"
            result["suggestions"] = ["Verify your logic and data types."]
        finally:
            sys.stdout = old_stdout
            result["output"] = stdout_capture.getvalue()
        
        return result

def debug_code(code: str) -> Dict[str, Any]:
    """Secure public interface for code debugging."""
    # 1. Security check (includes syntax check via ast.parse)
    dangers = Debugger.analyze_security(code)
    if dangers:
        return {
            "answer": "Security/Validation Block:\n- " + "\n- ".join(dangers),
            "status": "blocked",
            "type": "security"
        }

    # 2. Execution in sandbox
    execution_result = Debugger.execute_safely(code)
    
    if execution_result["success"]:
        return {
            "answer": f"Code executed successfully.\n\nOutput:\n{execution_result['output'] or '[No output]'}",
            "status": "success",
            "details": execution_result
        }
    else:
        return {
            "answer": f"Runtime Error:\n{execution_result['error']}",
            "status": "error",
            "type": "runtime",
            "details": execution_result
        }
