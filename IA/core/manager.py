import importlib.util
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np

class GripenAI:
    def __init__(self):
        self.modules = {}
        self.model = None
        self.prepare_brain()

    def prepare_brain(self):
        """Prepara una red neuronal básica."""
        self.tokenizer = keras.preprocessing.text.Tokenizer(num_words=5000)
        self.model = keras.Sequential([
            keras.layers.Input(shape=(10,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("🧠 Red neuronal lista.")

    def load_modules(self):
        """Carga directamente modules/language/model.py"""
        path = os.path.join(os.path.dirname(__file__), "..", "modules", "language", "model.py")
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("language", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.modules["language"] = module
            print(f"✅ Módulo cargado: language -> {path}")
        else:
            print(f"❌ No se encontró model.py en modules/language/")

    def run_module(self, module_name, *args, **kwargs):
        """Ejecuta un módulo específico."""
        module = self.modules.get(module_name)
        if not module:
            return f"Módulo '{module_name}' no encontrado."
        if hasattr(module, "run"):
            return module.run(self, *args, **kwargs)
        else:
            return f"El módulo '{module_name}' no tiene función 'run'."