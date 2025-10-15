import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def run(ai, text):
    """
    Procesa el texto usando la red neuronal de Gripen.
    Devuelve un número como predicción, usando la red neuronal que ya está en ai.model.
    """
    # --- Paso 1: Convertir texto a secuencias numéricas ---
    # Entrenamos el tokenizer con la frase (solo ejemplo)
    ai.tokenizer.fit_on_texts([text])
    seq = ai.tokenizer.texts_to_sequences([text])
    
    # Rellenamos a longitud 10 (igual que la red espera)
    seq_padded = pad_sequences(seq, maxlen=10)
    
    # --- Paso 2: Predicción con la red ---
    pred = ai.model.predict(seq_padded, verbose=0)
    
    # --- Paso 3: Convertir resultado en respuesta ---
    # Por ahora, un ejemplo simple: si pred>0.5 "Sí", si <0.5 "No"
    respuesta = "Sí" if pred[0][0] > 0.5 else "No"
    
    return f"Gripen predice: {respuesta} (valor: {pred[0][0]:.2f})"
