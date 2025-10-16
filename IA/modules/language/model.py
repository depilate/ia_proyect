import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def run(ai, text):
    """
    Procesa texto con la red neuronal de Gripen.
    """
    MAX_LEN = 15  # Debe coincidir con train.py
    
    # Convertir texto a secuencia
    try:
        seq = ai.tokenizer.texts_to_sequences([text.lower().strip()])
        seq_padded = pad_sequences(seq, maxlen=MAX_LEN)
        
        # Predicción
        pred = ai.model.predict(seq_padded, verbose=0)
        valor = pred[0][0]
        
        # Interpretar resultado
        if valor > 0.65:  # Confianza alta en SÍ
            respuesta = "Sí"
            confianza = valor
        elif valor < 0.35:  # Confianza alta en NO
            respuesta = "No"
            confianza = 1 - valor
        else:  # Zona gris (indeciso)
            respuesta = "No estoy seguro"
            confianza = 0.5
        
        return f"{respuesta} (confianza: {confianza:.1%})"
        
    except Exception as e:
        return f"Error al procesar: {str(e)}"