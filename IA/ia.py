import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.manager import GripenAI

def mostrar_menu():
    print("\n" + "="*50)
    print("🤖 GRIPEN AI v1.0 - Sistema Avanzado")
    print("="*50)
    print("1. 💬 Modo Conversación (Lenguaje)")
    print("2. 📊 Ver Estadísticas del Modelo")
    print("3. 🧪 Modo Prueba Rápida")
    print("4. 🎓 Entrenar con Nuevos Datos")
    print("5. ❌ Salir")
    print("="*50)

def modo_conversacion(ai):
    print("\n💬 Modo Conversación Activado")
    print("Escribe 'menu' para volver al menú principal\n")
    
    while True:
        entrada = input("Tú: ").strip()
        
        if not entrada:
            continue
            
        if entrada.lower() == "menu":
            break
            
        if entrada.lower() in ["salir", "exit", "quit"]:
            print("Gripen: Hasta luego 👋")
            return False
            
        respuesta = ai.run_module("language", entrada)
        print(f"🤖 Gripen: {respuesta}\n")
    
    return True

def ver_estadisticas(ai):
    print("\n📊 Estadísticas del Modelo")
    print("-" * 50)
    
    if ai.model:
        print(f"✅ Modelo cargado: Sí")
        print(f"📝 Capas del modelo: {len(ai.model.layers)}")
        
        # Contar parámetros
        total_params = ai.model.count_params()
        print(f"🔢 Parámetros totales: {total_params:,}")
        
        if ai.tokenizer:
            vocab_size = len(ai.tokenizer.word_index)
            print(f"📚 Tamaño del vocabulario: {vocab_size} palabras")
            
            # Mostrar algunas palabras del vocabulario
            palabras = list(ai.tokenizer.word_index.keys())[:10]
            print(f"🔤 Primeras palabras: {', '.join(palabras)}")
    else:
        print("❌ No hay modelo cargado")
    
    print("-" * 50)
    input("\nPresiona ENTER para continuar...")

def modo_prueba_rapida(ai):
    print("\n🧪 Modo Prueba Rápida")
    print("Probando el modelo con frases predefinidas...\n")
    
    frases_prueba = [
        "¿el sol es amarillo?",
        "¿los peces vuelan?",
        "¿puedes ayudarme?",
        "¿eres un humano?",
        "¿el agua es líquida?",
        "¿la luna es cuadrada?",
        "¿estás funcionando?",
        "¿puedes comer pizza?"
    ]
    
    print("Resultados:")
    print("-" * 50)
    for frase in frases_prueba:
        respuesta = ai.run_module("language", frase)
        print(f"❓ {frase}")
        print(f"   → {respuesta}\n")
    
    print("-" * 50)
    input("\nPresiona ENTER para continuar...")

def entrenar_interactivo(ai):
    print("\n🎓 Entrenamiento Interactivo")
    print("Añade nuevos ejemplos de entrenamiento")
    print("(Esta función requiere modificar train.py manualmente)")
    print("-" * 50)
    
    print("\n💡 Para entrenar con nuevos datos:")
    print("1. Abre el archivo 'train.py'")
    print("2. Añade más ejemplos en 'training_data'")
    print("3. Ejecuta: python train.py")
    print("4. Reinicia esta aplicación")
    
    print("\nEjemplo de formato:")
    print('  ("¿pregunta aquí?", 1),  # 1 = Sí')
    print('  ("¿otra pregunta?", 0),  # 0 = No')
    
    input("\nPresiona ENTER para continuar...")

def main():
    # Inicializar IA
    ai = GripenAI()
    ai.load_modules()
    
    continuar = True
    
    while continuar:
        mostrar_menu()
        opcion = input("Selecciona una opción: ").strip()
        
        if opcion == "1":
            continuar = modo_conversacion(ai)
        elif opcion == "2":
            ver_estadisticas(ai)
        elif opcion == "3":
            modo_prueba_rapida(ai)
        elif opcion == "4":
            entrenar_interactivo(ai)
        elif opcion == "5":
            print("\n👋 ¡Hasta luego!")
            continuar = False
        else:
            print("\n❌ Opción no válida. Intenta de nuevo.")
            input("Presiona ENTER para continuar...")

if __name__ == "__main__":
    main()