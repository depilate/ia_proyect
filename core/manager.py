import importlib
import os
from tensorflow.keras.models import load_model

class CoreAI:
    def __init__(self, model_path="core_model.h5"):
        """
        Inicializa el core de la IA.
        Carga el modelo de TensorFlow y prepara el diccionario de módulos.
        """
        # Cargar modelo de TensorFlow si existe
        if os.path.exists(model_path):
            self.model = load_model(model_path)
        else:
            self.model = None
        self.modules = {}

    def load_modules(self, modules_path="modules"):
        """
        Carga todos los módulos .py que estén en la carpeta modules/
        y los guarda en self.modules con su nombre de archivo como clave.
        """
        for file in os.listdir(modules_path):
            if file.endswith(".py") and not file.startswith("__"):
                name = file[:-3]  # quitar .py
                self.modules[name] = importlib.import_module(f"modules.{name}")

    def run_module(self, name, *args, **kwargs):
        """
        Ejecuta la función 'run' del módulo indicado.
        Devuelve lo que el módulo retorne.
        """
        if name in self.modules:
            return self.modules[name].run(*args, **kwargs)
        else:
            raise ValueError(f"Módulo {name} no encontrado")
