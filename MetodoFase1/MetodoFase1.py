# -*- coding: utf-8 -*-
"""
FASE 1 DEL MÉTODO DE DOS FASES
=================================================
Esta implementación muestra paso a paso cómo se determina la factibilidad
de un problema de programación lineal antes de intentar optimizarlo.

ENTRADA: Problema de programación lineal completo
    - Función objetivo: min/max c^T x
    - Restricciones: Ax {<=, >=, =} b
    - No negatividad: x >= 0

SALIDA: Determinación de factibilidad + solución básica inicial si existe
"""

import numpy as np

TOL = 1e-10  # Tolerancia numérica

class Phase1Result:
    """Resultado completo de la Fase 1"""
    def __init__(self, status, x_original, x_extended, basis_idx, tableau, var_names,
                 problem_type, original_c):
        self.status = status
        self.x_original = x_original      # Solución solo en variables originales
        self.x_extended = x_extended      # Solución completa (con auxiliares)
        self.basis_idx = basis_idx
        self.tableau = tableau
        self.var_names = var_names
        self.problem_type = problem_type  # "min" o "max"
        self.original_c = original_c      # Vector de costos original

    def __str__(self):
        result = f"\n{'='*60}\n"
        result += f"RESULTADO DE LA FASE 1\n"
        result += f"{'='*60}\n"
        result += f"Estado del problema: {self.status.upper()}\n"

        if self.status == "feasible":
            result += f"El problema original ES FACTIBLE\n"
            result += f"Valor de W* = {self.tableau[-1, -1]:.6f}\n"
            result += f"\nSolución básica factible encontrada:\n"
            for i, val in enumerate(self.x_original):
                result += f"  x{i+1} = {val:.6f}\n"
            result += f"\nVariables en la base: {[self.var_names[i] for i in self.basis_idx]}\n"
            result += f"\nEste problema puede continuar a la Fase 2 para encontrar el óptimo.\n"
        else:
            result += f"El problema original ES INFACTIBLE\n"
            result += f"Valor de W* = {self.tableau[-1, -1]:.6f} > 0\n"
            result += f"\nNo existe solución que satisfaga todas las restricciones.\n"

        return result


# ---------------------- Impresión del tableau (didáctica) ----------------------
def _print_tableau(tableau, var_names, basis_idx, title="TABLEAU", step_num=None):
    if step_num is not None:
        print(f"\n{'='*50}")
        print(f"PASO {step_num}: {title}")
        print(f"{'='*50}")
    else:
        print(f"\n=== {title} ===")

    m, n = tableau.shape
    total_vars = n - 1

    # Encabezados
    print("Base".ljust(8), end="")
    for name in var_names:
        print(name.rjust(8), end="")
    print("RHS".rjust(8))
    print("-" * (8 + 8 * len(var_names) + 8))

    # Filas de restricciones
    for i in range(m-1):
        basic_var = var_names[basis_idx[i]] if i < len(basis_idx) else "?"
        print(basic_var.ljust(8), end="")
        for j in range(total_vars):
            print(f"{tableau[i,j]:8.3f}", end="")
        print(f"{tableau[i,-1]:8.3f}")

    # Fila objetivo W
    print("W".ljust(8), end="")
    for j in range(total_vars):
        print(f"{tableau[-1,j]:8.3f}", end="")
    print(f"{tableau[-1,-1]:8.3f}")
    print()

# ---------------------- NUEVO: impresión explícita de B -----------------------
def _print_B(extended_A, basis_idx, var_names, title="MATRIZ B"):
    """
    Muestra la matriz B (columnas básicas de A extendida) de forma explícita.
    - extended_A: matriz A tras agregar s/e/a (m x total_vars)
    - basis_idx: índices de las columnas básicas
    - var_names: nombres de todas las columnas
    """
    B = extended_A[:, basis_idx]  # (m x m)
    print("\n" + "-"*50)
    print(title)
    print("-"*50)
    # Encabezados con nombres de variables básicas
    header = "      " + " ".join([f"{var_names[j]:>10}" for j in basis_idx])
    print(header)
    # Cuerpo
    for i in range(B.shape[0]):
        row = "fila%02d:" % (i+1)
        row += "".join([f"{B[i, j]:10.3f}" for j in range(B.shape[1])])
        print(row)
    print("-"*50)

# ------------------- Reglas de selección (Bland) y pivoteo --------------------
def _bland_rule_entering(cost_row, verbose=False):
    """Selecciona variable entrante según Bland (menor índice con costo < 0)."""
    idx = np.where(cost_row < -TOL)[0]
    if idx.size == 0:
        return None
    entering = int(idx.min())
    if verbose:
        print(f"Variables con costo negativo: {idx}")
        print(f"Regla de Bland selecciona: columna {entering}")
    return entering

def _ratio_test_bland(col, rhs, var_names, basis_idx, verbose=False):
    """Prueba de razón mínima con explicación detallada."""
    mask = col > TOL
    if not np.any(mask):
        return None

    if verbose:
        print(f"\nPrueba de razón mínima:")
        print("Fila | Variable básica | Coef. | RHS | Razón")
        print("-" * 45)

    ratios = np.full(len(col), np.inf)
    valid_rows = []

    for i in range(len(col)):
        if mask[i]:
            ratio = rhs[i] / col[i] if col[i] > TOL else np.inf
            ratios[i] = ratio
            valid_rows.append(i)

            if verbose:
                var_name = var_names[basis_idx[i]] if i < len(basis_idx) else "?"
                print(f"{i+1:4d} | {var_name:14s} | {col[i]:5.3f} | {rhs[i]:3.0f} | {ratio:5.3f}")

    if not valid_rows:
        return None

    # Encontrar mínimo
    min_ratio = np.min(ratios[ratios < np.inf])
    tied_rows = [i for i in valid_rows if abs(ratios[i] - min_ratio) <= TOL]
    leaving_row = min(tied_rows)  # Bland

    if verbose:
        print(f"\nRazón mínima: {min_ratio:.3f}")
        if len(tied_rows) > 1:
            print(f"Empate en filas: {[i+1 for i in tied_rows]}")
            print(f"Regla de Bland selecciona: fila {leaving_row+1}")
        else:
            print(f"Variable saliente: {var_names[basis_idx[leaving_row]]} (fila {leaving_row+1})")

    return leaving_row

def _pivot(T, pivot_row, pivot_col, verbose=False):
    """Operación de pivoteo con explicación."""
    if verbose:
        pivot_val = T[pivot_row, pivot_col]
        print(f"\nPivoteo en elemento ({pivot_row+1}, {pivot_col+1}) = {pivot_val:.3f}")

    # Normalizar fila pivote
    T[pivot_row, :] = T[pivot_row, :] / T[pivot_row, pivot_col]

    # Eliminar columna en otras filas
    m = T.shape[0]
    for r in range(m):
        if r == pivot_row:
            continue
        factor = T[r, pivot_col]
        if abs(factor) > TOL:
            T[r, :] -= factor * T[pivot_row, :]

def solve_phase1_complete(c, A, b, sense, problem_type="min", verbose=True, show_tableaus=True):
    """
    FUNCIÓN PRINCIPAL: Resuelve la Fase 1 de un problema completo de PL.
    """
    # Validación y normalización
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    m, n = A.shape
    sense = list(sense)

    if verbose:
        print("="*60)
        print("RESOLUCIÓN DE FASE 1 - DETERMINACIÓN DE FACTIBILIDAD")
        print("="*60)
        print(f"\nPROBLEMA ORIGINAL:")
        print(f"{'Minimizar' if problem_type == 'min' else 'Maximizar'}:")
        obj_str = " + ".join([f"{c[i]:.1f}*x{i+1}" for i in range(n) if abs(c[i]) > TOL])
        print(f"  {obj_str}")
        print(f"\nSujeto a:")
        for i in range(m):
            const_str = " + ".join([f"{A[i,j]:.1f}*x{j+1}" for j in range(n) if abs(A[i,j]) > TOL])
            print(f"  {const_str} {sense[i]} {b[i]:.1f}")
        print(f"  x1, x2, ..., x{n} >= 0")

        print(f"\nNOTA: En la Fase 1 NO usamos la función objetivo original.")
        print(f"En su lugar, construimos un problema auxiliar para encontrar factibilidad.")

    # PASO 1: Normalización (b >= 0)
    if verbose:
        print(f"\n" + "="*50)
        print("PASO 1: NORMALIZACIÓN (Asegurar b >= 0)")
        print("="*50)

    changes_made = False
    for i in range(m):
        if b[i] < 0:
            if verbose:
                print(f"b[{i+1}] = {b[i]:.1f} < 0, multiplicando restricción {i+1} por -1")
            A[i, :] *= -1
            b[i] *= -1
            if sense[i] == "<=":
                sense[i] = ">="
            elif sense[i] == ">=":
                sense[i] = "<="
            changes_made = True

    if verbose and not changes_made:
        print("No se requieren cambios, b >= 0 ya está satisfecho.")

    # PASO 2: Construcción del problema auxiliar
    if verbose:
        print(f"\n" + "="*50)
        print("PASO 2: CONSTRUCCIÓN DEL PROBLEMA AUXILIAR")
        print("="*50)

    extended_A = A.copy()
    var_names = [f"x{j+1}" for j in range(n)]
    art_cols = []
    basis_idx = np.full(m, -1, dtype=int)
    current_col = n

    if verbose:
        print("Agregando variables auxiliares:")

    for i in range(m):
        if sense[i] == "<=":
            s_col = np.zeros((m, 1)); s_col[i, 0] = 1.0
            extended_A = np.hstack([extended_A, s_col])
            var_names.append(f"s{i+1}")
            basis_idx[i] = current_col
            if verbose: print(f"  Restricción {i+1}: + s{i+1} (holgura)")
            current_col += 1

        elif sense[i] == ">=":
            e_col = np.zeros((m, 1)); e_col[i, 0] = -1.0
            a_col = np.zeros((m, 1)); a_col[i, 0] =  1.0
            extended_A = np.hstack([extended_A, e_col, a_col])
            var_names.extend([f"e{i+1}", f"a{i+1}"])
            art_cols.append(current_col + 1)
            basis_idx[i] = current_col + 1
            if verbose: print(f"  Restricción {i+1}: - e{i+1} + a{i+1} (exceso + artificial)")
            current_col += 2

        elif sense[i] == "=":
            a_col = np.zeros((m, 1)); a_col[i, 0] = 1.0
            extended_A = np.hstack([extended_A, a_col])
            var_names.append(f"a{i+1}")
            art_cols.append(current_col)
            basis_idx[i] = current_col
            if verbose: print(f"  Restricción {i+1}: + a{i+1} (artificial)")
            current_col += 1

    if verbose:
        print(f"\nVariables artificiales: {[var_names[j] for j in art_cols]}")
        print(f"Función objetivo auxiliar: W = {' + '.join([var_names[j] for j in art_cols])}")

    # >>> NUEVO: Imprimir B (base inicial)
    _print_B(extended_A, basis_idx, var_names, title="MATRIZ B (BASE INICIAL)")

    # PASO 3: Tableau inicial y reducción
    if verbose:
        print(f"\n" + "="*50)
        print("PASO 3: TABLEAU INICIAL Y REDUCCIÓN")
        print("="*50)

    total_vars = extended_A.shape[1]
    T = np.zeros((m + 1, total_vars + 1))
    T[:m, :total_vars] = extended_A
    T[:m, -1] = b

    cost = np.zeros(total_vars)
    if art_cols:
        cost[art_cols] = 1.0
    T[-1, :total_vars] = cost

    if show_tableaus:
        _print_tableau(T, var_names, basis_idx, "INICIAL (antes de reducción)")

    # Reducción inicial
    for i in range(m):
        cb = cost[basis_idx[i]]
        if abs(cb) > TOL:
            if verbose: print(f"Reduciendo fila objetivo: restando {cb:.0f} × (fila {i+1})")
            T[-1, :] -= cb * T[i, :]

    if show_tableaus:
        _print_tableau(T, var_names, basis_idx, "INICIAL (después de reducción)")

    # PASO 4: Iteraciones del método Simplex
    if verbose:
        print(f"\n" + "="*50)
        print("PASO 4: ITERACIONES DEL MÉTODO SIMPLEX")
        print("="*50)

    iteration = 0
    MAX_ITERS = 1000

    while iteration < MAX_ITERS:
        iteration += 1

        if verbose:
            print(f"\n--- ITERACIÓN {iteration} ---")

        # Variable entrante
        entering = _bland_rule_entering(T[-1, :total_vars], verbose)
        if entering is None:
            if verbose:
                print("ÓPTIMO ALCANZADO: No hay costos negativos")
            break

        if verbose:
            print(f"Variable entrante: {var_names[entering]}")

        # Variable saliente
        leaving_row = _ratio_test_bland(T[:m, entering], T[:m, -1],
                                        var_names, basis_idx, verbose)
        if leaving_row is None:
            raise RuntimeError("Problema auxiliar no acotado")

        # Pivoteo
        _pivot(T, leaving_row, entering, verbose)
        basis_idx[leaving_row] = entering

        # >>> NUEVO: Imprimir B en cada iteración (base actual)
        _print_B(extended_A, basis_idx, var_names,
                 title=f"MATRIZ B (DESPUÉS DE ITERACIÓN {iteration})")

        if show_tableaus:
            _print_tableau(T, var_names, basis_idx, f"DESPUÉS DE ITERACIÓN {iteration}")

        if verbose:
            current_W = T[-1, -1]
            print(f"Nuevo valor de W: {current_W:.6f}")

    # Extraer resultados
    W_star = T[-1, -1]
    x_extended = np.zeros(total_vars)
    for i in range(m):
        x_extended[basis_idx[i]] = T[i, -1]

    x_original = x_extended[:n].copy()
    status = "feasible" if abs(W_star) <= TOL else "infeasible"

    if verbose:
        print(f"\n" + "="*60)
        print("PASO 5: INTERPRETACIÓN DEL RESULTADO")
        print("="*60)
        print(f"Valor óptimo de W* = {W_star:.10f}")

        if status == "feasible":
            print("Como W* ≈ 0, el problema original ES FACTIBLE")
            print("Existe al menos una solución que satisface todas las restricciones")
            print("\nSolución básica factible encontrada:")
            for i in range(n):
                print(f"  x{i+1} = {x_original[i]:.6f}")
        else:
            print("Como W* > 0, el problema original ES INFACTIBLE")
            print("No existe solución que satisfaga todas las restricciones")

    return Phase1Result(
        status=status,
        x_original=x_original,
        x_extended=x_extended,
        basis_idx=basis_idx.copy(),
        tableau=T.copy(),
        var_names=var_names,
        problem_type=problem_type,
        original_c=c
    )

# ======================================================================
# ENTRADA INTERACTIVA
# ======================================================================
if __name__ == "__main__":
    print("="*60)
    print("INGRESO DEL PROBLEMA DE PROGRAMACIÓN LINEAL")
    print("="*60)

    # Número de variables y restricciones
    n = int(input("Ingrese el número de variables de decisión: "))
    m = int(input("Ingrese el número de restricciones: "))

    # Función objetivo
    print("\nIngrese los coeficientes de la función objetivo separados por espacio:")
    c = list(map(float, input(f"c[1..{n}]: ").split()))

    # Tipo de problema
    problem_type = input("\n¿Desea 'min' o 'max'? ").strip().lower()

    # Restricciones
    A, b, sense = [], [], []
    print("\nIngrese cada restricción en el formato:")
    print("coeficientes separados por espacio   signo   valor")
    print("Ejemplo: 20 50 <= 3000")
    print("-"*50)
    for i in range(m):
        restr = input(f"Restricción {i+1}: ").split()
        coef = list(map(float, restr[:-2]))
        signo = restr[-2]
        val = float(restr[-1])
        A.append(coef)
        sense.append(signo)
        b.append(val)

    # Resolver fase 1
    resultado = solve_phase1_complete(
        c, A, b, sense, problem_type,
        verbose=True, show_tableaus=True
    )

    # Mostrar resultado final
    print(resultado)
