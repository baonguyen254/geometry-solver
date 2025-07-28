"""
Clean Triangle Solver - Refactored version with better structure
Preserves all original logic from geometry_problem_solver_9_NA_2.py
"""

from sympy import symbols, Eq, solve, sqrt, sin, cos, pi, simplify, sympify, tan, atan, asin, acos, atan2
from sympy.abc import _clash
import math
import re
import time
from typing import Dict, List, Tuple, Optional, Any, Union
import tkinter as tk
from tkinter import scrolledtext
import threading


class TriangleSolver:
    """Main triangle solver class with clean structure."""
    
    def __init__(self):
        self._init_variables()
        self._init_formulas()
        self._init_math_context()
    
    def _init_variables(self):
        """Initialize variables and descriptions."""
        self.variables = ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'S', 'R', 'p', 'h_a', 'h_b', 'h_c', 
                         'm_a', 'm_b', 'm_c', 'p_a', 'p_b', 'p_c', 'r', 'r_a', 'r_b', 'r_c']
        
        self.descriptions = {
            'a': 'Cạnh a',
            'b': 'Cạnh b', 
            'c': 'Cạnh c',
            'alpha': 'Góc đối diện cạnh a',
            'beta': 'Góc đối diện cạnh b',
            'gamma': 'Góc đối diện cạnh c',
            'S': 'Diện tích',
            'R': 'Bán kính đường tròn ngoại tiếp tam giác',
            'p': 'Nửa chu vi',
            'h_a': 'Đường cao từ đỉnh A',
            'h_b': 'Đường cao từ đỉnh B',
            'h_c': 'Đường cao từ đỉnh C',
            'm_a': 'Đường trung tuyến từ đỉnh A',
            'm_b': 'Đường trung tuyến từ đỉnh B',
            'm_c': 'Đường trung tuyến từ đỉnh C',
            'p_a': 'Đường phân giác từ đỉnh A',
            'p_b': 'Đường phân giác từ đỉnh B',
            'p_c': 'Đường phân giác từ đỉnh C',
            'r': 'Bán kính đường tròn nội tiếp tam giác',
            'r_a': 'Bán kính đường tròn nội tiếp từ đỉnh A',
            'r_b': 'Bán kính đường tròn nội tiếp từ đỉnh B',
            'r_c': 'Bán kính đường tròn nội tiếp từ đỉnh C',
        }
        
        self.sym_vars = {var: symbols(var) for var in self.variables}
    
    def _init_math_context(self):
        """Initialize math context for symbolic operations."""
        self.math_context = {
            **self.sym_vars, 'sin': sin, 'cos': cos, 'sqrt': sqrt, 'tan': tan, 'atan': atan, 
            'pi': pi, 'simplify': simplify, 'math': math, 'asin': asin, 'acos': acos, 'atan2': atan2
        }
    
    def _generate_base_formulas(self) -> List[Dict]:
        """Generate base triangle formulas."""
        formulas = []
        
        # Cosine law (cạnh)
        for x, y, z, X in [('a', 'b', 'c', 'alpha'), ('b', 'a', 'c', 'beta'), ('c', 'a', 'b', 'gamma')]:
            formulas.append({
                'eq': f'{x}**2 = ({y}**2 + {z}**2 - 2*{y}*{z}*cos({X}))', 
                'output': x, 
                'inputs': [y, z, X]
            })
        
        # Cosine law (góc)
        for x, y, z, X in [('a', 'b', 'c', 'alpha'), ('b', 'a', 'c', 'beta'), ('c', 'a', 'b', 'gamma')]:
            formulas.append({
                'eq': f'cos({X}) = ({y}**2 + {z}**2 - {x}**2)/(2*{y}*{z})', 
                'output': X, 
                'inputs': [x, y, z]
            })
        
        # Sine law (cạnh)
        for x, X, y, Y in [('a', 'alpha', 'b', 'beta'), ('b', 'beta', 'c', 'gamma'), ('c', 'gamma', 'a', 'alpha')]:
            formulas.append({
                'eq': f'{x}/sin({X}) = {y}/sin({Y})', 
                'output': x, 
                'inputs': [y, X, Y]
            })
        
        # Sine law (góc)
        for x, X, y, Y in [('a', 'alpha', 'b', 'beta'), ('b', 'beta', 'c', 'gamma'), ('c', 'gamma', 'a', 'alpha')]:
            formulas.append({
                'eq': f'sin({X})/{x} = sin({Y})/{y}', 
                'output': X, 
                'inputs': [x, y, Y]
            })
        
        # Area (2 cạnh 1 góc xen giữa)
        for x, y, Z in [('a', 'b', 'gamma'), ('b', 'c', 'alpha'), ('c', 'a', 'beta')]:
            formulas.append({
                'eq': f'S = ({x}*{y}*sin({Z})/2)', 
                'output': 'S', 
                'inputs': [x, y, Z]
            })
        
        # Heron and perimeter formulas
        formulas.extend([
            {'eq': 'S = sqrt(p*(p-a)*(p-b)*(p-c))', 'output': 'S', 'inputs': ['p', 'a', 'b', 'c']},
            {'eq': 'p = (a+b+c)/2', 'output': 'p', 'inputs': ['a', 'b', 'c']},
            {'eq': 'c = 2*p-a-b', 'output': 'c', 'inputs': ['a', 'b', 'p']},
            {'eq': 'a = 2*p-c-b', 'output': 'a', 'inputs': ['c', 'b', 'p']},
            {'eq': 'b = 2*p-c-a', 'output': 'b', 'inputs': ['a', 'c', 'p']},
        ])
        
        # Sum of angles
        formulas.extend([
            {'eq': 'alpha + beta + gamma = pi', 'output': 'alpha', 'inputs': ['beta', 'gamma']},
            {'eq': 'alpha + beta + gamma = pi', 'output': 'beta', 'inputs': ['alpha', 'gamma']},
            {'eq': 'alpha + beta + gamma = pi', 'output': 'gamma', 'inputs': ['alpha', 'beta']},
        ])
        
        # Heights
        for x, X, h in [('a', 'alpha', 'h_a'), ('b', 'beta', 'h_b'), ('c', 'gamma', 'h_c')]:
            formulas.append({
                'eq': f'{h} = (2*S/{x})', 
                'output': h, 
                'inputs': ['S', x]
            })
        
        # Medians
        for x, y, z, m in [('a', 'b', 'c', 'm_a'), ('b', 'a', 'c', 'm_b'), ('c', 'a', 'b', 'm_c')]:
            formulas.append({
                'eq': f'{m} = (sqrt(2*{y}**2 + 2*{z}**2 - {x}**2)/2)', 
                'output': m, 
                'inputs': [x, y, z]
            })
        
        # Angle bisectors
        for x, y, z, p in [('alpha', 'b', 'c', 'p_a'), ('beta', 'a', 'c', 'p_b'), ('gamma', 'a', 'b', 'p_c')]:
            formulas.append({
                'eq': f'{p} = (2*{y}*{z}*cos({x}/2)/({y}+{z}))', 
                'output': p, 
                'inputs': [y, z, x]
            })
        
        # Circumradius and inradius
        formulas.extend([
            {'eq': 'R = (a*b*c)/(4*S)', 'output': 'R', 'inputs': ['a', 'b', 'c', 'S']},
            {'eq': 'r = S/p', 'output': 'r', 'inputs': ['S', 'p']},
        ])
        
        return formulas
    
    def _extract_inverse_rules(self, eq_str: str) -> List[Dict]:
        """Extract inverse formulas using algebra rules."""
        valid_vars = self.variables
        
        try:
            eq_str = eq_str.replace(' ', '')
            lhs_str, rhs_str = eq_str.split('=')
            lhs = sympify(lhs_str, locals=_clash)
            rhs = sympify(rhs_str, locals=_clash)

            A = lhs
            A_str = str(A)

            # Only process if A is a valid variable
            if A_str not in valid_vars:
                return []

            if rhs.is_Mul:
                args = list(rhs.args)

                # Special form A = (k * B) / C
                pow_arg = next((arg for arg in args if arg.is_Pow and arg.exp == -1), None)
                if pow_arg:
                    C = pow_arg.base
                    numerator_terms = [arg for arg in args if arg != pow_arg]
                    if len(numerator_terms) == 2 and numerator_terms[0].is_number:
                        k, B = numerator_terms
                        B_str, C_str = str(B), str(C)
                        if B_str in valid_vars and C_str in valid_vars:
                            return [
                                {'eq': f'{B_str} = {A_str}*{C_str}/{str(k)}', 'output': B_str, 'inputs': [A_str, C_str]},
                                {'eq': f'{C_str} = {str(k)}*{B_str}/{A_str}', 'output': C_str, 'inputs': [B_str, A_str]},
                            ]

                # A = B / C (detected via Pow)
                for i in range(len(args)):
                    if args[i].is_Pow and args[i].exp == -1:
                        B = simplify(rhs / args[i])
                        C = args[i].base
                        B_str, C_str = str(B), str(C)
                        if B_str in valid_vars and C_str in valid_vars:
                            return [
                                {'eq': f'{C_str} = {B_str}/{A_str}', 'output': C_str, 'inputs': [B_str, A_str]},
                                {'eq': f'{B_str} = {A_str}*{C_str}', 'output': B_str, 'inputs': [A_str, C_str]},
                            ]

                # Standard multiplication A = B * C
                if len(args) == 2:
                    B, C = args
                    B_str, C_str = str(B), str(C)
                    if B_str in valid_vars and C_str in valid_vars:
                        return [
                            {'eq': f'{B_str} = {A_str}/{C_str}', 'output': B_str, 'inputs': [A_str, C_str]},
                            {'eq': f'{C_str} = {A_str}/{B_str}', 'output': C_str, 'inputs': [A_str, B_str]},
                        ]

            elif rhs.is_Add:
                args = rhs.args
                if len(args) == 2:
                    B, C = args
                    B_str, C_str = str(B), str(C)
                    if B_str in valid_vars and C_str in valid_vars:
                        return [
                            {'eq': f'{B_str} = {A_str} - {C_str}', 'output': B_str, 'inputs': [A_str, C_str]},
                            {'eq': f'{C_str} = {A_str} - {B_str}', 'output': C_str, 'inputs': [A_str, B_str]},
                        ]

            elif rhs.is_Pow:
                B, C = rhs.args
                B_str, C_str = str(B), str(C)
                if B_str in valid_vars and C_str in valid_vars:
                    return [
                        {'eq': f'{B_str} = {A_str}**(1/{C_str})', 'output': B_str, 'inputs': [A_str, C_str]},
                        {'eq': f'{C_str} = log({A_str})/log({B_str})', 'output': C_str, 'inputs': [A_str, B_str]},
                    ]

        except Exception as e:
            print(f"[Skip] Could not process {eq_str}: {e}")
            return []

        return []
    
    def _simplify_formula_inputs(self, formula: Dict) -> List[Dict]:
        """Simplify complex inputs in formulas by extracting individual variables."""
        valid_vars = self.variables
        
        # Get the original formula
        eq_str = formula['eq']
        output = formula['output']
        inputs = formula['inputs']
        
        # Check if output is valid
        if output not in valid_vars:
            return []  # Skip formulas with invalid outputs
        
        # Check if any input is a complex expression
        complex_inputs = []
        simple_inputs = []
        
        for inp in inputs:
            # Check if input is a complex expression (contains operators)
            if any(op in inp for op in ['+', '-', '*', '/', '**', '(', ')']):
                complex_inputs.append(inp)
            else:
                simple_inputs.append(inp)
        
        if not complex_inputs:
            # All inputs are simple, check if they are all valid variables
            if all(inp in valid_vars for inp in inputs):
                return [formula]
            else:
                return []  # Skip if any input is not a valid variable
        
        # For complex inputs, we'll skip these formulas for now
        # as they create too many intermediate variables
        return []
    
    def _infer_general_formulas(self, formulas: List[Dict]) -> List[Dict]:
        """Infer general formulas from base formulas."""
        inferred = []
        seen = set()

        for f in formulas:
            original_eq = f['eq']
            inverse_forms = self._extract_inverse_rules(original_eq)
            for inv in inverse_forms:
                if inv['eq'] not in seen:
                    rule_used = inv.get('rule', 'Unknown')
                    print(f"\n🔁 From: {original_eq} → Rule applied: {rule_used}")
                    print(f"  → {inv['eq']}   [Output: {inv['output']}, Inputs: {', '.join(inv['inputs'])}]")
                    inferred.append(inv)
                    seen.add(inv['eq'])

        return inferred
    
    def _init_formulas(self):
        """Initialize all formulas including inferred ones."""
        base_formulas = self._generate_base_formulas()
        inferred = self._infer_general_formulas(base_formulas)
        
        # Simplify complex inputs in all formulas
        all_formulas = base_formulas + inferred
        simplified_formulas = []
        
        for formula in all_formulas:
            simplified = self._simplify_formula_inputs(formula)
            simplified_formulas.extend(simplified)
        
        # Convert to dictionary format
        self.formulas = {f'formula_{i}': f for i, f in enumerate(simplified_formulas)}
        
        # Print formulas for debugging
        for i, f in enumerate(simplified_formulas):
            print(f)
    
    def parse_input(self, input_str: str) -> Tuple[Optional[Dict], List, List]:
        """Parse input string into known values and relations."""
        known = {}
        relations = []
        steps = []
        
        for part in input_str.split(','):
            part = part.strip()
            if '=' in part:
                left, right = part.split('=')
                left, right = left.strip(), right.strip()
                
                if left not in self.variables:
                    return None, None, [f"Biến {left} không hợp lệ"]
                
                # Nếu right là số
                if right.isdigit() or right.replace('.', '').isdigit():
                    known[left] = float(right)
                else:
                    # Kiểm tra tất cả biến trong biểu thức right đã có trong known chưa
                    tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', right)
                    if all(tok in known for tok in tokens):
                        try:
                            # giải phương trình right với các biến trong known
                            val = eval(right, {**known})
                            if isinstance(val, (int, float)):
                                known[left] = float(val)
                                steps.append(f"Giải phương trình: {left} = {right} = {val}")
                            else:
                                relations.append(Eq(eval(left, self.sym_vars), eval(right, self.sym_vars)))
                                steps.append(f"Thêm quan hệ: {left} = {right}")
                        except Exception:
                            relations.append(Eq(eval(left, self.sym_vars), eval(right, self.sym_vars)))
                            steps.append(f"Thêm quan hệ: {left} = {right}")
                    else:
                        # Chưa đủ biến, thêm vào relations
                        relations.append(Eq(eval(left, self.sym_vars), eval(right, self.sym_vars)))
                        steps.append(f"Thêm quan hệ: {left} = {right}")
        
        known, relations, steps = self._try_update_known_from_relations(known, relations, steps)
        return known, relations, steps
    
    def _try_update_known_from_relations(self, known: Dict, relations: List, steps: List) -> Tuple[Dict, List, List]:
        """Update known values from relations."""
        updated = True
        while updated:
            updated = False
            to_remove = []
            
            for rel in relations:
                if isinstance(rel, Eq):
                    left, right = rel.lhs, rel.rhs
                    left_str, right_str = str(left), str(right)
                    
                    # Nếu left là biến đã biết, giải cho right (nếu right là biến chưa biết)
                    if left_str in known and right_str not in known and len(right.free_symbols) == 1:
                        try:
                            # Giải phương trình left_val = right cho biến right
                            var = list(right.free_symbols)[0]
                            sol = solve(Eq(known[left_str], right), var)
                            if sol:
                                val = float(sol[0].evalf())
                                known[str(var)] = val
                                steps.append(f"Giải phương trình: {left_str} = {right} => {var} = {val}")
                                to_remove.append(rel)
                                updated = True
                                continue
                        except Exception:
                            continue
                    
                    # Nếu right là biến đã biết, giải cho left (nếu left là biến chưa biết)
                    if right_str in known and left_str not in known and len(left.free_symbols) == 1:
                        try:
                            var = list(left.free_symbols)[0]
                            sol = solve(Eq(left, known[right_str]), var)
                            if sol:
                                val = float(sol[0].evalf())
                                known[str(var)] = val
                                steps.append(f"Giải phương trình: {left} = {right_str} => {var} = {val}")
                                to_remove.append(rel)
                                updated = True
                                continue
                        except Exception:
                            continue
                    
                    # Trường hợp left là biến chưa biết, right chỉ chứa các biến đã biết
                    right_vars = [str(s) for s in right.free_symbols]
                    if left_str not in known and all(v in known for v in right_vars):
                        try:
                            val = float(right.evalf(subs=known))
                            known[left_str] = val
                            steps.append(f"Giải phương trình: {left} = {right} => {left_str} = {val}")
                            to_remove.append(rel)
                            updated = True
                        except Exception:
                            continue
            
            for rel in to_remove:
                relations.remove(rel)
        
        return known, relations, steps
    
    def check_triangle_type(self, known: Dict) -> str:
        """Check triangle type."""
        if 'a' in known and 'b' in known and 'c' in known:
            a, b, c = known['a'], known['b'], known['c']
            if a == b == c:
                return 'equilateral'
            elif a == b or b == c or a == c:
                return 'isosceles'
        
        if ('alpha' in known and known['alpha'] == math.pi/2) or ('beta' in known and known['beta'] == math.pi/2) or ('gamma' in known and known['gamma'] == math.pi/2):
            return 'right'
        
        return 'general'
    
    def is_valid_triangle(self, known: Dict) -> Tuple[bool, str]:
        """Validate triangle data."""
        # Kiểm tra các cạnh phải dương
        for k in ['a', 'b', 'c']:
            if k in known and isinstance(known[k], (int, float)) and known[k] <= 0:
                return False, f"Cạnh {k} phải lớn hơn 0."
        
        # Lưu lại giá trị độ gốc của các góc để kiểm tra
        deg_angles = {}
        for k in ['alpha', 'beta', 'gamma']:
            # Nếu có cả 3 góc thì kiểm tra tổng 3 góc phải bằng 180 độ
            if all(k in known for k in ['alpha', 'beta', 'gamma']):
                if known['alpha'] + known['beta'] + known['gamma'] != 180:
                    return False, "Tổng 3 góc phải bằng 180 độ."
            if k in known:
                # Chỉ kiểm tra nếu là số thực (int, float), tuyệt đối bỏ qua nếu là biểu thức sympy hoặc kiểu khác
                if type(known[k]) in (int, float):
                    deg_angles[k] = known[k]
                    if known[k] <= 0 or known[k] >= 180:
                        return False, f"Góc {k} phải trong khoảng (0, 180) độ."
        
        # Kiểm tra tam giác vuông: cạnh huyền phải là lớn nhất (trên giá trị độ gốc)
        for angle, side in [('alpha', 'a'), ('beta', 'b'), ('gamma', 'c')]:
            if angle in deg_angles and abs(deg_angles[angle] - 90) < 1e-3:
                # Góc vuông
                if side in known and type(known[side]) in (int, float):
                    others = [k for k in ['a', 'b', 'c'] if k != side and k in known and type(known[k]) in (int, float)]
                    if any(known[side] < known[o] for o in others):
                        return False, f"Cạnh {side} đối diện góc vuông phải là cạnh lớn nhất (cạnh huyền)."
        
        # Sau khi kiểm tra xong, chuyển đổi sang radian nếu là số thực
        for k in ['alpha', 'beta', 'gamma']:
            if k in known and type(known[k]) in (int, float):
                known[k] = math.radians(known[k])
        
        # Kiểm tra bất đẳng thức tam giác
        if all(k in known for k in ['a', 'b', 'c']):
            a, b, c = known['a'], known['b'], known['c']
            if a + b <= c or a + c <= b or b + c <= a:
                return False, "Tổng hai cạnh bất kỳ phải lớn hơn cạnh còn lại."
        
        return True, ""
    
    def _backward_chain(self, goal: str, known: Dict, relations: List = None, visited: set = None, depth: int = 0, max_depth: int = 5) -> Tuple[Optional[Any], List, Dict]:
        """Backward chaining algorithm to find unknown variables."""
        if visited is None:
            visited = set()
        if relations is None:
            relations = []
        if goal in visited:
            return None, [f"Vòng lặp phát hiện cho {goal}"], known
        if depth > max_depth:
            return None, [f"Đạt đến độ sâu tối đa khi tìm {goal}"], known
        visited.add(goal)
        if goal in known:
            return known[goal], [f"{goal} đã biết: {known[goal]}"], known
        
        best_result = None
        best_steps = []
        
        # Tìm công thức phù hợp dựa trên known và goal
        formulas_search = []
        scored_formulas = []
        
        for key, info in self.formulas.items():
            all_vars = set(info['inputs'] + [info['output']])
            if goal in all_vars:
                num_known = sum(1 for v in all_vars if v in known)
                num_missing = len(all_vars) - num_known
                # Chỉ chọn công thức mà chỉ còn 1 biến chưa biết (chính là goal)
                if num_missing <= 1 and goal not in known:
                    formulas_search.append((num_known, -num_missing, key, info))
        
        # Ưu tiên công thức có nhiều biến đã biết nhất
        formulas_search.sort(reverse=True)
        if not formulas_search:
            # search formula theo output
            formulas_search = [(key, info) for key, info in self.formulas.items() if info['output'] == goal]
            # Tính score cho từng công thức
            for fname, formula in formulas_search:
                inputs = formula['inputs']
                num_known = sum(1 for inp in inputs if inp in known)
                num_missing = len(inputs) - num_known
                scored_formulas.append((num_known, -num_missing, fname, formula))
            # Sắp xếp: nhiều input đã biết nhất, ít input thiếu nhất
            scored_formulas.sort(reverse=True)
            formulas_search = scored_formulas

        # Duyệt theo thứ tự phù hợp nhất
        for _, _, fname, formula in formulas_search:
            # Cách 1: Tìm công thức phù hợp dựa trên goal, hoặc phải tìm thêm biến chưa biết
            if scored_formulas:
                inputs = formula['inputs']
                condition = formula.get('condition', '')
                if condition:
                    cond_met = all(c.split('=')[0].strip() in known and str(known[c.split('=')[0].strip()]) == c.split('=')[1].strip() for c in condition.split(' and '))
                    if not cond_met and (self.check_triangle_type(known) != 'right' or 'gamma=90' not in condition) and \
                    (self.check_triangle_type(known) != 'isosceles' or not condition.startswith('a = b')) and \
                    (self.check_triangle_type(known) != 'equilateral' or not condition.startswith('a = b = c')):
                        continue
                missing = [inp for inp in inputs if inp not in known]
                sub_steps = []
                temp_known = known.copy()
                can_solve = True
                for miss in missing:
                    if miss in temp_known:
                        continue
                    val, sub, temp_known = self._backward_chain(miss, temp_known, relations, visited, depth+1, max_depth)
                    if val is not None:
                        temp_known[miss] = val
                        sub_steps.extend(sub)
                        if goal in temp_known:
                            return temp_known[goal], sub_steps, temp_known
                    else:
                        can_solve = False
                        break
                if can_solve:
                    eq_str = formula['eq']
                    if '=' in eq_str:
                        left, right = eq_str.split('=')
                        eq = Eq(eval(left.strip(), self.math_context), eval(right.strip(), self.math_context))
                    else:
                        eq = Eq(eval(eq_str, self.math_context), 0)
                    try:
                        exprs = [left.strip(), right.strip()] if '=' in eq_str else [eq_str]
                        for expr in exprs:
                            if 'sqrt' in expr:
                                val_expr = eval(expr, {**self.math_context, **temp_known})
                                if hasattr(val_expr, 'is_real') and not val_expr.is_real:
                                    return None, [f"Biểu thức {expr} không cho giá trị thực (có thể do căn số âm)."], temp_known
                                if hasattr(val_expr, 'is_real') and val_expr.is_real and hasattr(val_expr, 'is_number') and val_expr.is_number and val_expr < 0:
                                    return None, [f"Biểu thức {expr} có giá trị âm dưới căn."], temp_known
                        if '/0' in eq_str.replace(' ', ''):
                            return None, [f"Biểu thức {eq_str} có phép chia cho 0."], temp_known
                        sol = solve(eq.subs(temp_known), self.sym_vars[goal])
                        if sol:
                            sol_val = sol[0]
                            # Nếu nghiệm là số phức, bỏ qua hoặc báo lỗi
                            if hasattr(sol_val, 'is_real') and not sol_val.is_real:
                                steps = [f"Nghiệm {goal} = {sol_val} là số phức, không hợp lệ."]
                                return None, steps, temp_known
                            if hasattr(sol_val, 'free_symbols') and len(sol_val.free_symbols) == 0:
                                try:
                                    val = float(sol_val.evalf())
                                except Exception:
                                    val = sol_val
                            else:
                                val = sol_val
                            if goal in ['a', 'b', 'c', 'h_a', 'h_b', 'h_c', 'm_a', 'm_b', 'm_c', 'R', 'r', 'r_a', 'r_b', 'r_c', 'S', 'p', 'p_a', 'p_b', 'p_c']:
                                val = abs(val)
                            temp_known[goal] = val
                            result = temp_known[goal]
                            steps = sub_steps + [f"Áp dụng {fname}: {eq_str} => {goal} = {result}"]
                            if best_result is None or len(steps) < len(best_steps):
                                best_result = result
                                best_steps = steps
                                break
                    except Exception as e:
                        steps.append(f"Lỗi khi áp dụng {fname}: {str(e)}")
            # Cách 2: với công thức phù hợp dựa trên known và goal, có thể giải trực tiếp
            else:
                # Lúc này chỉ còn 1 biến chưa biết là goal, các biến còn lại đều đã biết
                eq_str = formula['eq']
                try:
                    if '=' in eq_str:
                        left, right = eq_str.split('=')
                        eq = Eq(eval(left.strip(), self.math_context), eval(right.strip(), self.math_context))
                    else:
                        eq = Eq(eval(eq_str, self.math_context), 0)
                    # Thay thế các biến đã biết vào phương trình
                    eq = eq.subs(known)
                    sol = solve(eq, self.sym_vars[goal])
                    if sol:
                        sol_val = sol[0]
                        # Nếu nghiệm là số phức, bỏ qua hoặc báo lỗi
                        if hasattr(sol_val, 'is_real') and not sol_val.is_real:
                            steps = [f"Nghiệm {goal} = {sol_val} là số phức, không hợp lệ."]
                            return None, steps, known
                        if hasattr(sol_val, 'free_symbols') and len(sol_val.free_symbols) == 0:
                            try:
                                val = float(sol_val.evalf())
                            except Exception:
                                val = sol_val
                        else:
                            val = sol_val
                        if goal in ['a', 'b', 'c', 'h_a', 'h_b', 'h_c', 'm_a', 'm_b', 'm_c', 'R', 'r', 'r_a', 'r_b', 'r_c', 'S', 'p', 'p_a', 'p_b', 'p_c']:
                            val = abs(val)
                        known[goal] = val
                        steps = [f"Áp dụng {fname}: {eq_str} => {goal} = {val}"]
                        return val, steps, known
                except Exception as e:
                    steps = [f"Lỗi khi áp dụng {fname}: {str(e)}"]
                    return None, steps, known
        if best_result is not None:
            known.update(temp_known)
            known[goal] = best_result
            # update known from temp_known
            return best_result, best_steps, known
        # Nếu không giải được bằng các công thức từng bước, mới thử giải hệ phương trình
        # cách 3: giải hệ phương trình
        # Tập hợp các biến chưa biết
        unknown_vars = set([goal])
        for rel in relations:
            unknown_vars.update(str(s) for s in rel.free_symbols if str(s) not in known)

        eqs = []
        eqs_info = []  # Lưu thông tin (fname hoặc 'relation', phương trình)
        for rel in relations:
            # Chỉ thêm nếu có ít nhất một biến chưa biết
            if any(str(s) in unknown_vars for s in rel.free_symbols):
                eqs.append(rel.subs(known))
                eqs_info.append(('relation', str(rel.subs(known))))

        for fname, formula in self.formulas.items():
            # Chỉ thêm nếu có ít nhất một biến chưa biết trong inputs hoặc output
            all_vars = set(formula['inputs'] + [formula['output']])
            if any(v not in known for v in all_vars) and (formula['output'] in unknown_vars or any(inp in unknown_vars for inp in formula['inputs'])):
                eq_str = formula['eq']
                if '=' in eq_str:
                    left, right = eq_str.split('=')
                    eq_obj = Eq(eval(left.strip(), self.math_context).subs(known), eval(right.strip(), self.math_context).subs(known))
                else:
                    eq_obj = Eq(eval(eq_str, self.math_context).subs(known), 0)
                eqs.append(eq_obj)
                eqs_info.append((fname, str(eq_obj)))
        # Tìm các biến thực sự liên quan
        related_vars = set([goal])
        for eq in eqs:
            related_vars.update(str(s) for s in eq.free_symbols)
        unknowns = [self.sym_vars[v] for v in self.variables if v not in known and v in related_vars]
        if not unknowns:
            unknowns = [self.sym_vars[goal]]
        try:
            # Show hệ phương trình kèm fname
            eqs_str = [f"{info[0]}: {info[1]}" for info in eqs_info]
            eqs_str = "\n".join(eqs_str)
            show_steps = [f"Giải hệ phương trình với các phương trình: \n{eqs_str}"]
            sol = solve(eqs, unknowns, dict=True)
            if sol:
                for k, v in sol[0].items():
                    if hasattr(v, 'free_symbols') and len(v.free_symbols) == 0:
                        try:
                            v = float(v.evalf())
                        except Exception:
                            pass
                        known[str(k)] = abs(v)
                    else:
                        # Nếu vẫn còn free_symbols, lưu lại như một relation mới
                        relations.append(Eq(self.sym_vars[str(k)], v))
                if goal in known:
                    show_steps.extend([f'Giải hệ phương trình: {goal} = {known[goal]}'])
                    return known[goal], show_steps, known
                else:
                    return None, show_steps.extend([f'Không thể tìm {goal}']), known
            else:
                return None, [f'Không thể tìm {goal}'], known
        except Exception as e:
            return None, [f'Lỗi khi giải hệ phương trình cho {goal}: {str(e)}'], known
        return None, [f'Không thể tìm {goal}'], known
    
    def solve_triangle(self, hypotheses_str: str, conclusions_str: str) -> Tuple[Dict, List]:
        """Main solve function for triangle problems."""
        print("hypotheses_str: ", hypotheses_str)
        print("conclusions_str: ", conclusions_str)
        
        known_values, relations, steps = self.parse_input(hypotheses_str)
        if not known_values:
            return {"error": "Không thể phân tích đầu vào"}, steps
        
        # Kiểm tra hợp lệ đầu vào
        valid, msg = self.is_valid_triangle(known_values)
        if not valid:
            return {"error": msg}, [msg]
        
        goals = [goal.strip() for goal in conclusions_str.split(',')]
        triangle_type = self.check_triangle_type(known_values)

        results = {}
        known = known_values.copy()
        
        if relations:
            # Ưu tiên giải các quan hệ đơn giản trước
            simple_relations = []
            complex_relations = []
            
            for rel in relations:
                # Kiểm tra xem có phải quan hệ đơn giản không (chỉ chứa các biến góc)
                rel_str = str(rel)
                if all(var in ['alpha', 'beta', 'gamma'] for var in ['alpha', 'beta', 'gamma'] if var in rel_str):
                    simple_relations.append(rel)
                else:
                    complex_relations.append(rel)
            
            # Giải quan hệ đơn giản trước
            if simple_relations:
                try:
                    # Thêm tổng góc tam giác
                    angle_eqs = simple_relations + [Eq(self.sym_vars['alpha'] + self.sym_vars['beta'] + self.sym_vars['gamma'], pi)]
                    angle_vars = [self.sym_vars[v] for v in ['alpha', 'beta', 'gamma'] if v not in known]
                    
                    if angle_vars:
                        sol = solve(angle_eqs, angle_vars, dict=True)
                        if sol:
                            for k, v in sol[0].items():
                                if hasattr(v, 'free_symbols') and len(v.free_symbols) == 0:
                                    try:
                                        v = float(v.evalf())
                                    except Exception:
                                        pass
                                    known[str(k)] = v
                                else:
                                    # Nếu vẫn còn free_symbols, lưu lại như một relation mới
                                    relations.append(Eq(self.sym_vars[str(k)], v))
                            steps.append(f"Giải quan hệ góc: {sol[0]}")
                except Exception as e:
                    steps.append(f"Lỗi khi giải quan hệ góc: {str(e)}")
            
            # Sau đó giải từng biến còn lại
            for goal in goals:
                if goal in known and known[goal] is not None:
                    results[goal] = known[goal]
                    steps.append(f"{goal} đã biết: {known[goal]}")
                else:
                    value, step, known = self._backward_chain(goal, known, relations)
                    if value is not None:
                        results[goal] = value
                        if step is not None:
                            steps.extend(step)
                    else:
                        if step is not None:
                            steps.extend(step)
                        else:
                            steps.append(f"Không thể tìm {goal}")
        else:
            for goal in goals:
                if goal in known and known[goal] is not None:
                    results[goal] = known[goal]
                    steps.append(f"{goal} đã biết: {known[goal]}")
                else:
                    value, step, known = self._backward_chain(goal, known, relations)
                    if value is not None:
                        results[goal] = value
                        if step is not None:
                            steps.extend(step)
                    else:
                        if step is not None:
                            steps.extend(step)
                        else:
                            steps.append(f"Không thể tìm {goal}")
        
        return results, steps


# def main():
#     """Main function to run the triangle solver."""
#     solver = TriangleSolver()
    
#     print("Các biến và giải thích các biến:")
#     for var, val in solver.descriptions.items():
#         print(f"{var}: {val}")
#     print("--------------------------------")
#     print("Lưu ý: alpha, beta, gamma nhập theo ĐỘ (degree), hệ thống sẽ tự động quy đổi sang radian khi tính toán!")
#     print("--- Công cụ giải bài toán tam giác ---")
#     print("Nhập giả thiết dưới dạng: a=3, b=4, c=5 hoặc a=5, b=5, alpha=60")
#     print("Nhập kết luận dưới dạng: S, R hoặc gamma")
    
#     # Test cases
#     test_cases = [
#         ("a=3, b=4, c=5", "S, R"),
#         ("a=3, b=4, gamma=90", "c"),
#         ("a=3, b=4, alpha=90", "c"),
#         ("a=3, b=4, beta=90", "c"),
#         ("a=3, b=4, gamma=90", "c,S"),
#         ("a=5, b=3, alpha=beta+gamma", "c,alpha,beta,gamma"),
#         ("a=3, b=4, c=2*a", "c, S, R"),
#         ("a=3, b=4, c=b", "S, r, h_a"),
#         ("b=5, alpha=90, beta=alpha/2", "c, a, S"),
#     ]
    
#     additional_test_cases = [
#         ("a=5, b=5, c=5", "S, R, r, h_a, h_b, h_c, m_a, m_b, m_c, p_a, p_b, p_c"),
#         ("a=6, b=8, c=10", "S, R, r, h_a, h_b, h_c, m_a, m_b, m_c, p_a, p_b, p_c"),
#         ("a=7, b=7, gamma=60", "c, S, R, p"),
#         ("b=5, c=7, alpha=45", "a, S, h_b"),
#         ("a=9, c=13, beta=90", "b, S, h_c"),
#         ("alpha=50, beta=60, gamma=70", "a, b, c"),
#         ("alpha=30, beta=60, c=10", "a, b, S, R"),
#         ("a=4, b=4, alpha=45, beta=45", "c, S, R, r"),
#         ("a=3, b=4, alpha=90, gamma=60", "a, b, c, alpha, beta, gamma"),
#         ("alpha=beta, a=5, b=5, gamma=60", "c, alpha, beta, S, R, r"),
#     ]

#     expression_test_cases = [
#         ("a=4, b=3, c=2*a", "S, R"),
#         ("a=5, b=5, p=8", "c, S, R, r"),
#         ("a=7, b=8, alpha=2*beta, alpha=60", "beta, gamma, c, S"),
#         ("a=5, b=5, alpha=beta+10, beta=50", "alpha, gamma, c"),
#         ("a=3, b=4, p=5", "c, S, R, r"),
#         ("alpha=30, beta=2*alpha, a=5, b=6", "gamma, c, S"),
#         ("alpha=45, beta=45, gamma=180-alpha-beta, a=7", "b, c, S"),
#         ("a=5, b=5, m_a=7", "c, S, R, r"),
#         ("a=6, b=7, alpha=beta+20, beta=40", "alpha, gamma, c"),
#         ("a=5, b=3, alpha=beta+gamma", "c,alpha,beta,gamma"),
#         ("S=12, r=2, a=3, b=5", "c, R"),
#     ]

#     for i, (hypotheses, conclusions) in enumerate(expression_test_cases, 1):
#         start_time = time.time()
#         print(f"\nTest case {i}:")
#         print(f"  Giả thiết: {hypotheses}")
#         print(f"  Kết luận: {conclusions}")
#         print("--- Quá trình suy luận ---")
        
#         results, steps = solver.solve_triangle(hypotheses, conclusions)
        
#         for step in steps:
#             print(step)
        
#         print("--- Kết quả ---")
#         for var, val in results.items():
#             if isinstance(val, complex) or (hasattr(val, 'is_real') and not val.is_real):
#                 print(f"{var} = Không tồn tại nghiệm thực hợp lệ")
#             elif var in ['alpha', 'beta', 'gamma']:
#                 if isinstance(val, (int, float)):
#                     print(f"{var} = {round(math.degrees(val), 4)}")
#                 else:
#                     print(f"{var} = {val}")
#             else:
#                 print(f"{var} = {val}")
        
#         print("--- Thời gian ---")
#         print(f"Thời gian chạy: {time.time() - start_time} giây")


# if __name__ == "__main__":
#     main()


# --- Tkinter GUI Integration ---
class TriangleSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Triangle Geometry Problem Solver")

        # --- Guide Section ---
        guide_frame = tk.Frame(root)
        guide_frame.pack(pady=10)

        guide_label = tk.Label(guide_frame, text="📘 Hướng dẫn sử dụng & Giải thích biến", font=("Arial", 10, "bold"))
        guide_label.pack(anchor="w")

        self.guide_text = scrolledtext.ScrolledText(guide_frame, width=100, height=15, wrap=tk.WORD)
        self.guide_text.pack()
        self.guide_text.insert(tk.END, self.get_variable_guide())
        self.guide_text.config(state='disabled')  # Make read-only

        # --- Input Frame ---
        input_frame = tk.Frame(root)
        input_frame.pack(pady=10)

        tk.Label(input_frame, text="Giả thiết:").grid(row=0, column=0, sticky='w')
        self.hypotheses_entry = tk.Entry(input_frame, width=80)
        self.hypotheses_entry.grid(row=0, column=1, padx=5)

        tk.Label(input_frame, text="Kết luận:").grid(row=1, column=0, sticky='w')
        self.conclusions_entry = tk.Entry(input_frame, width=80)
        self.conclusions_entry.grid(row=1, column=1, padx=5)

        # Solve Button
        self.solve_button = tk.Button(root, text="Giải", command=self.solve_problem)
        self.solve_button.pack(pady=5)

        # --- Thinking Process Output ---
        tk.Label(root, text="--- Quá trình suy luận ---").pack()
        self.process_text = scrolledtext.ScrolledText(root, width=100, height=10)
        self.process_text.pack(pady=5)

        # --- Result Output ---
        tk.Label(root, text="--- Kết quả ---").pack()
        self.result_text = scrolledtext.ScrolledText(root, width=100, height=10)
        self.result_text.pack(pady=5)

    def get_variable_guide(self):
        return """\
Các biến và giải thích các biến:
a: Cạnh a
b: Cạnh b
c: Cạnh c
alpha: Góc đối diện cạnh a
beta: Góc đối diện cạnh b
gamma: Góc đối diện cạnh c
S: Diện tích
R: Bán kính đường tròn ngoại tiếp tam giác
p: Nửa chu vi
h_a: Đường cao từ đỉnh A
h_b: Đường cao từ đỉnh B
h_c: Đường cao từ đỉnh C
m_a: Đường trung tuyến từ đỉnh A
m_b: Đường trung tuyến từ đỉnh B
m_c: Đường trung tuyến từ đỉnh C
p_a: Đường phân giác từ đỉnh A
p_b: Đường phân giác từ đỉnh B
p_c: Đường phân giác từ đỉnh C
r: Bán kính đường tròn nội tiếp tam giác
r_a: Bán kính đường tròn nội tiếp từ đỉnh A
r_b: Bán kính đường tròn nội tiếp từ đỉnh B
r_c: Bán kính đường tròn nội tiếp từ đỉnh C
--------------------------------
Lưu ý: alpha, beta, gamma nhập theo ĐỘ (degree), hệ thống sẽ tự động quy đổi sang radian khi tính toán!
--- Công cụ giải bài toán tam giác ---
Nhập giả thiết dưới dạng: a=3, b=4, c=5 hoặc a=5, b=5, alpha=60
Nhập kết luận dưới dạng: S, R hoặc gamma
"""

    def solve_problem(self):
        hypotheses = self.hypotheses_entry.get()
        conclusions = self.conclusions_entry.get()

        self.process_text.delete(1.0, tk.END)
        self.result_text.delete(1.0, tk.END)
        solver = TriangleSolver()
        start_time = time.time()
        results, steps = solver.solve_triangle(hypotheses, conclusions)

        for step in steps:
            self.process_text.insert(tk.END, step + '\n')

        for var, val in results.items():
            if isinstance(val, complex) or (hasattr(val, 'is_real') and not val.is_real):
                self.result_text.insert(tk.END, f"{var} = Không tồn tại nghiệm thực hợp lệ\n")
            elif var in ['alpha', 'beta', 'gamma']:
                if isinstance(val, (int, float)):
                    self.result_text.insert(tk.END, f"{var} = {round(math.degrees(val), 4)}°\n")
                else:
                    self.result_text.insert(tk.END, f"{var} = {val}\n")
            else:
                self.result_text.insert(tk.END, f"{var} = {val}\n")

        self.result_text.insert(tk.END, f"\nThời gian chạy: {time.time() - start_time:.4f} giây")

def run_gui():
    root = tk.Tk()
    app = TriangleSolverApp(root)
    root.mainloop()


# --- Entry point ---
if __name__ == "__main__":
    # Choose one of the following:
    # main()     # Run CLI version
    run_gui()    # Run GUI version