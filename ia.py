import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.manager import CoreAI

# Crear instancia del core
ai = CoreAI()

# Cargar módulos desde modules/
ai.load_modules()

# Ejecutar módulo de ejemplo
resultado = ai.run_module("ejemplo", "Hola mundo")
print(resultado)
