# Proyecto_Fase1_Simplex
Implementación en **Python** de la **Fase 1 del Método de Dos Fases** en Programación Lineal.  
El programa permite ingresar un problema de PL, construir el problema auxiliar e identificar si es **factible o infactible** mediante el método **Simplex**, mostrando el proceso paso a paso.
---
## Características
- Ingreso de función objetivo (min o max).
- Definición de restricciones en formato estándar.
- Construcción automática del problema auxiliar (Fase 1).
- Resolución mediante Simplex con **regla de Bland**.
- Impresión de **tableaux**, **matriz B** y pasos de pivoteo.
- Informe final sobre factibilidad y solución básica inicial.

##  Ejemplo de Entrada
Ingrese el número de variables de decisión: 2
Ingrese el número de restricciones: 3

Ingrese los coeficientes de la función objetivo separados por espacio:
c[1..2]: 10000 6000

¿Desea 'min' o 'max'? max

Ingrese cada restricción en el formato:
coeficientes separados por espacio   signo   valor
Ejemplo: 20 50 <= 3000
--------------------------------------------------
Restricción 1: 20 50 <= 3000
Restricción 2: 1 1 <= 90
Restricción 3: 0 1 >= 10
