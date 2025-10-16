import os
import tensorflow as tf
from tensorflow import keras
import pickle

class GripenAI:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path = os.path.join(os.getcwd(), "trained_models", "gripen_model.h5")
        self.tokenizer_path = os.path.join(os.getcwd(), "trained_models", "tokenizer.pkl")
        self.modules_loaded = False

        # Intentar cargar modelo al iniciar
        self.load_brain()

    def load_brain(self):
        """Carga el modelo y tokenizer entrenados."""
        if os.path.exists(self.model_path) and os.path.exists(self.tokenizer_path):
            try:
                print(f"🔄 Cargando modelo desde {self.model_path}")
                self.model = keras.models.load_model(self.model_path)

                with open(self.tokenizer_path, "rb") as f:
                    self.tokenizer = pickle.load(f)

                print(f"✅ Modelo cargado con {self.model.count_params():,} parámetros")
            except Exception as e:
                print(f"⚠️ Error cargando modelo o tokenizer: {e}")
                self.model = None
                self.tokenizer = None
        else:
            print("❌ Modelo o tokenizer no encontrado. Ejecuta 'train.py' primero")
            self.model = None
            self.tokenizer = None

    def load_modules(self):
        """Simula la carga de módulos para compatibilidad con ia.py"""
        self.modules_loaded = True
        print("🔧 Módulos cargados correctamente")

    def run_module(self, module_name, input_text):
        """Ejecuta un módulo por nombre. Por ahora solo 'language'."""
        if module_name.lower() == "language":
            return self.predict(input_text)
        else:
            return f"❌ Módulo '{module_name}' no encontrado"

    def predict(self, pregunta):
        """Predice 0 o 1 para la pregunta dada usando el modelo entrenado."""
        if self.model is None or self.tokenizer is None:
            return "❌ Modelo no cargado"

        try:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            seq = self.tokenizer.texts_to_sequences([pregunta])
            pad = pad_sequences(seq, maxlen=16, padding="post", truncating="post")
            pred = self.model.predict(pad, verbose=0)[0][0]
            return "Sí" if pred > 0.5 else "No"
        except Exception as e:
            return f"⚠️ Error prediciendo: {e}"
