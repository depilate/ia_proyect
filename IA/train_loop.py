import subprocess
import os
import sys

# ==========================
# ⚙️ CONFIGURACIÓN DEL LOOP
# ==========================
TRAIN_SCRIPT = "C:\\Users\\Thorrija\\Desktop\\IA v2\\IA\\train.py"
MAX_RUNS = 80                  # Máximo de veces que entrenará
CHECK_EVERY = 10               # Cada cuántas ejecuciones revisa mejora
VAL_LOG = "training_data/val_acc.txt"  # Archivo temporal donde guardaremos la métrica de validación

# Crear carpeta si no existe
os.makedirs(os.path.dirname(VAL_LOG), exist_ok=True)

# ==========================
# 🔁 LOOP DE ENTRENAMIENTO
# ==========================
prev_val_acc = 0.0
stop_loop = False

for run in range(1, MAX_RUNS + 1):
    print(f"\n🚀 Entrenamiento {run}/{MAX_RUNS}...")
    
    # Ejecuta train.py usando el mismo Python que este script
    subprocess.run([sys.executable, TRAIN_SCRIPT])
    
    # Cada CHECK_EVERY iteraciones, verifica mejora
    if run % CHECK_EVERY == 0:
        if os.path.exists(VAL_LOG):
            with open(VAL_LOG, "r") as f:
                try:
                    val_acc = float(f.read().strip())
                    print(f"📊 Validación actual: {val_acc:.4f} | Anterior: {prev_val_acc:.4f}")
                    if val_acc <= prev_val_acc:
                        print("⚠️ No hay mejora significativa, deteniendo entrenamiento.")
                        stop_loop = True
                        break
                    prev_val_acc = val_acc
                except:
                    print("❌ No se pudo leer la métrica de validación.")
        else:
            print("⚠️ Archivo de validación no encontrado, continuando...")

if not stop_loop:
    print(f"\n✅ Se completaron los {MAX_RUNS} entrenamientos.")
