import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.manager import GripenAI

ai = GripenAI()
ai.load_modules()

print("Gripen_v1\n Selecciona el modulo a utilizar:\n 1. Lenguaje")

modeloautilizar = input("Seleccione un numero: ")

if modeloautilizar == "1":
    while True:
        entrada = input("Tú: ")
        if entrada.lower() in ["salir", "exit", "quit"]:
            print("Gripen: Hasta luego 👋")
            break

        respuesta = ai.run_module("language", entrada)
        print(f"Gripen_L1: {respuesta}")
else:
    print("No se selecciono un numero valido")