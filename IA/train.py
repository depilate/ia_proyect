import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import pickle

# ==========================
# ⚙️ CONFIGURACIÓN
# ==========================
MAX_WORDS = 3000
MAX_LEN = 16
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 0.001
MODEL_PATH = "trained_models/gripen_model.h5"
ERROR_LOG = "training_data/errores.csv"
VAL_LOG = "training_data/val_acc.txt"

os.makedirs(os.path.dirname(ERROR_LOG), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(VAL_LOG), exist_ok=True)

# ==========================
# 📚 DATOS DE ENTRENAMIENTO
# ==========================
dataset = [
    # Aquí van tus frases de entrenamiento (pregunta, etiqueta 0 o 1)
    # DEJADO VACÍO PARA QUE AÑADAS DESPUÉS
    # POSITIVOS
    ("¿el agua moja?", 1),
    ("¿el agua es húmeda?", 1),
    ("¿el fuego quema?", 1),
    ("¿el hielo es frío?", 1),
    ("¿las piedras son duras?", 1),
    ("¿el sol brilla?", 1),
    ("¿la noche es oscura?", 1),
    ("¿el cielo es azul?", 1),
    ("¿las nubes flotan?", 1),
    ("¿los ríos fluyen con agua?", 1),
    ("¿la lluvia cae del cielo?", 1),
    ("¿el vapor de agua es gas?", 1),
    ("¿el hielo se derrite con calor?", 1),
    ("¿el agua hierve a 100 grados?", 1),
    ("¿el agua puede evaporarse?", 1),
    ("¿el agua puede congelarse?", 1),
    ("¿el agua apaga el fuego?", 1),
    ("¿los océanos contienen agua salada?", 1),
    ("¿las nubes están hechas de vapor de agua?", 1),
    ("¿los peces viven en el agua?", 1),
    ("¿los humanos beben agua para vivir?", 1),
    ("¿el agua es líquida?", 1),
    ("¿el agua no tiene sabor?", 1),
    ("¿el agua conduce electricidad si tiene sales?", 1),
    ("¿el agua se convierte en hielo al enfriarse?", 1),
    ("¿el fuego calienta?", 1),
    ("¿el fuego da calor?", 1),
    ("¿los pájaros vuelan?", 1),
    ("¿los perros ladran?", 1),
    ("¿las vacas mugen?", 1),
    ("¿las plantas hacen fotosíntesis?", 1),
    ("¿dos más dos es cuatro?", 1),
    ("¿el oxígeno es necesario para respirar?", 1),
    ("¿el cerebro controla el cuerpo?", 1),
    ("¿python es un lenguaje de programación?", 1),
    ("el agua puede ser sólida", 1),
    ("¿el agua es incolora?", 1),
("¿el fuego da calor?", 1),
("¿el hielo está frío?", 1),
("¿los peces nadan en el agua?", 1),
("¿los pájaros vuelan en el cielo?", 1),
("¿las vacas producen leche?", 1),
("¿el sol es amarillo?", 1),
("¿la luna refleja la luz del sol?", 1),
("¿las plantas realizan fotosíntesis?", 1),
("¿los árboles producen oxígeno?", 1),
("¿el agua se congela a 0 grados?", 1),
("¿el vapor de agua es gaseoso?", 1),
("¿la lluvia cae desde las nubes?", 1),
("¿los ríos fluyen hacia el mar?", 1),
("¿el océano contiene agua salada?", 1),
("¿el fuego se apaga con agua?", 1),
("¿las piedras son duras?", 1),
("¿el cielo es azul durante el día?", 1),
("¿los unicornios no existen?", 1),
("¿el aire tiene peso?", 1),
("¿los peces viven en el agua?", 1),
("¿el sol es más grande que la tierra?", 1),
("¿las nubes están hechas de agua?", 1),
("¿las montañas existen?", 1),
("¿la nieve es fría?", 1),
("¿el agua hierve a 100 grados?", 1),
("¿los humanos tienen una cabeza?", 1),
("¿los perros ladran?", 1),
("¿los gatos maúllan?", 1),
("¿las vacas mugen?", 1),
("¿las plantas requieren luz para crecer?", 1),
("¿dos más dos es cuatro?", 1),
("¿el oxígeno es necesario para respirar?", 1),
("¿el cerebro controla el cuerpo?", 1),
("¿Python es un lenguaje de programación?", 1),
("¿los árboles tienen raíces?", 1),
("¿el sol sale por el este?", 1),
("¿las plantas crecen hacia la luz?", 1),
("¿los humanos respiran aire?", 1),
("¿los coches circulan por carreteras?", 1),
("¿el hielo flota en el agua?", 1),
("¿el vapor de agua se condensa en gotas?", 1),
("¿los peces tienen branquias para respirar?", 1),
("¿el sol calienta la tierra durante el día?", 1),
("¿las vacas comen pasto?", 1),
("¿el agua hidrata a los humanos?", 1),
("¿los perros mueven la cola?", 1),
("¿los gatos cazan ratones?", 1),
("¿el fuego produce luz y calor?", 1),
("¿las estrellas brillan en el cielo nocturno?", 1),
("¿la tierra es redonda?", 1),
("¿los planetas giran alrededor del sol?", 1),
("¿el agua es segura para beber?", 1),
("¿las montañas están formadas por rocas?", 1),
("¿los árboles producen frutos?", 1),
("¿el sol sale cada mañana?", 1),
("¿las nubes se mueven con el viento?", 1),
("¿los ríos transportan agua dulce?", 1),
("¿los humanos tienen cinco sentidos?", 1),
("¿el hielo se derrite con calor?", 1),
("¿la luna orbita la tierra?", 1),
("¿las estrellas existen en el espacio?", 1),
("¿los árboles tienen hojas?", 1),
("¿el aire es necesario para la vida?", 1),
("¿el fuego necesita oxígeno para arder?", 1),
("¿el hielo se forma cuando el agua se congela?", 1),
("¿los humanos necesitan agua para vivir?", 1),
("¿los gatos maúllan para comunicarse?", 1),
("¿los perros pueden oler muy bien?", 1),
("¿las vacas tienen cuatro estómagos?", 1),
("¿los peces nadan usando aletas y cola?", 1),
("¿el sol es una estrella?", 1),
("¿la luna orbita alrededor de la tierra?", 1),
("¿las plantas convierten CO2 en oxígeno?", 1),
("¿el agua es esencial para la vida?", 1),
("¿los ríos transportan agua desde las montañas?", 1),
("¿los humanos tienen corazón?", 1),
("¿el sol proporciona luz y energía?", 1),
("¿los peces dependen del agua para vivir?", 1),
("¿los gatos cazan ratones?", 1),
("¿los perros protegen a sus dueños?", 1),
("¿el fuego se puede controlar?", 1),
("¿el hielo se derrite al aumentar la temperatura?", 1),
("las plantas requieren luz solar para crecer?", 1),
("¿los árboles tienen troncos?", 1),
("¿el aire contiene oxígeno?", 1),
("¿el agua se puede beber para hidratarse?", 1),
("los humanos tienen cerebro?", 1),
("¿las montañas son altas?", 1),
("¿los ríos llevan agua dulce?", 1),
("¿el sol sale cada día?", 1),
("¿la luna cambia de forma en sus fases?", 1),
("¿los gatos respiran aire?", 1),
("¿los perros necesitan comida y agua?", 1),
("¿el hielo flota en el agua?", 1),
("¿los peces nadan en ríos y océanos?", 1),
("¿las plantas tienen raíces?", 1),
("¿los árboles producen frutos y semillas?", 1),
("¿el fuego necesita oxígeno para arder?", 1),
("¿el agua es líquida a temperatura ambiente?", 1),
("¿los humanos necesitan dormir y comer?", 1),
("¿el sol proporciona luz durante el día?", 1),
("la luna refleja luz solar?", 1),
("¿los gatos cazan y maúllan?", 1),
("¿los perros ladran y mueven la cola?", 1),
("¿los peces viven bajo el agua?", 1),
("¿el hielo se derrite?", 1),
("¿el hielo se derrite?", 1),
("¿el hielo se derrite?", 1),
("¿el hielo se derrite?", 1),
("¿los peces viven bajo el agua?", 1),
("¿los peces viven bajo el agua?", 1),
("¿los peces viven bajo el agua?", 1),
("¿los peces viven bajo el agua?", 1),

    # NEGATIVOS
    ("¿el agua es seca?", 0),
    ("¿el agua quema como el fuego?", 0),
    ("¿el agua es sólida por naturaleza?", 0),
    ("¿los peces viven fuera del agua?", 0),
    ("¿las nubes son de algodón?", 0),
    ("¿la lluvia sube hacia arriba?", 0),
    ("¿el hielo está caliente?", 0),
    ("¿el vapor de agua es sólido?", 0),
    ("¿el agua tiene color rojo?", 0),
    ("¿los ríos fluyen hacia el cielo?", 0),
    ("¿el océano es seco?", 0),
    ("¿los unicornios existen?", 0),
    ("¿el sol es negro?", 0),
    ("¿las vacas vuelan al espacio?", 0),
    ("¿el fuego es frío?", 0),
    ("¿las plantas caminan?", 0),
    ("¿los árboles bailan salsa?", 0),
    ("¿los peces caminan por la calle?", 0),
    ("¿el fuego se congela?", 0),
    ("¿el agua hierve a 50 grados?", 0),
    ("¿los peces vuelan en el agua?", 0),
    ("¿el vapor de agua es líquido?", 0),
    ("¿el agua es venenosa?", 0),
    ("¿el agua nunca se congela?", 0),
    ("¿el vapor de agua moja?", 0),
    ("¿el agua es roja?", 0),
("¿el fuego es frío?", 0),
("¿el hielo está caliente?", 0),
("¿los peces caminan por la tierra?", 0),
("¿los pájaros nadan en el suelo?", 0),
("¿las vacas vuelan al espacio?", 0),
("¿el sol es negro?", 0),
("¿la luna produce luz propia?", 0),
("¿las plantas caminan por la casa?", 0),
("¿los árboles bailan salsa?", 0),
("¿el agua nunca se congela?", 0),
("¿el vapor de agua es sólido?", 0),
("¿la lluvia sube hacia arriba?", 0),
("¿los ríos fluyen hacia el cielo?", 0),
("¿los océanos son de color rosa?", 0),
("¿el fuego se convierte en hielo?", 0),
("¿las piedras vuelan?", 0),
("¿el cielo es naranja por la noche?", 0),
("¿los unicornios existen?", 0),
("¿el aire pesa 0 gramos?", 0),
("¿los peces vuelan por el espacio?", 0),
("¿el sol es azul?", 0),
("¿las nubes son de algodón?", 0),
("¿las montañas desaparecen de noche?", 0),
("¿la nieve es caliente?", 0),
("¿el agua hierve a 50 grados?", 0),
("¿los humanos tienen 10 cabezas?", 0),
("¿los perros hablan como humanos?", 0),
("¿los gatos pueden volar?", 0),
("¿los peces vuelan en el cielo?", 0),
("¿el hielo se derrite a -10 grados?", 0),
("¿el fuego no produce calor?", 0),
("¿las estrellas caen cada minuto?", 0),
("¿la tierra es plana?", 0),
("¿el oxígeno no es necesario para respirar?", 0),
("¿los planetas se chocan todos los días?", 0),
("¿el agua es venenosa?", 0),
("¿las montañas flotan en el aire?", 0),
("¿los árboles caminan de noche?", 0),
("¿el sol se apaga en segundos?", 0),
("¿las plantas caminan hacia el sol?", 0),
("¿los humanos respiran agua?", 0),
("¿los coches vuelan solos por la ciudad?", 0),
("¿el hielo se convierte en fuego al tocarlo?", 0),
("¿el vapor de agua moja como aceite?", 0),
("¿el cielo es de color verde por la mañana?", 0),
("¿los peces cantan canciones?", 0),
("¿el sol es más pequeño que una manzana?", 0),
("¿la luna es más grande que la tierra?", 0),
("¿el agua nunca hierve?", 0),
("¿los ríos corren hacia arriba?", 0),
("¿las nubes son sólidas?", 0),
("¿el fuego se apaga con calor?", 0),
("¿las piedras son líquidas?", 0),
("¿los humanos pueden respirar bajo el sol?", 0),
("¿el aire es comestible?", 0),
("¿el hielo es amarillo?", 0),
("¿los árboles se mueven de lugar?", 0),
("¿el sol es más frío que la luna?", 0),
("¿las vacas caminan en el cielo?", 0),
("¿los unicornios viven en la ciudad?", 0),
("¿el agua es dulce por naturaleza?", 0),
("¿los peces vuelan en la luna?", 0),
("¿las montañas desaparecen de día?", 0),
("¿el vapor de agua es comestible?", 0),
("¿el hielo flota sobre el fuego?", 0),
("¿los gatos conducen coches?", 0),
("¿el sol se esconde en el océano?", 0),
("¿la luna tiene luz propia?", 0),
("¿los árboles producen fuego?", 0),
("¿el fuego es sólido?", 0),
("¿las nubes son rosas por la noche?", 0),
("¿los ríos fluyen hacia el cielo?", 0),
("¿los humanos vuelan cuando duermen?", 0),
("¿el agua hierve a -5 grados?", 0),
("¿el hielo puede encender fuego?", 0),
("¿los peces viven en el desierto?", 0),
("¿el sol es cuadrado?", 0),
("¿el cielo es de color negro brillante?", 0),
("¿las plantas caminan hacia la luna?", 0),
("¿el aire se puede beber?", 0),
("¿las montañas son líquidas?", 0),
("¿los árboles caminan de día?", 0),
("¿los perros vuelan cuando ladran?", 0),
("¿el hielo arde al contacto con agua?", 0),
("¿el agua es sólida a temperatura ambiente?", 0),
("¿los gatos vuelan por la ventana?", 0),
("¿el sol desaparece de repente?", 0),
("¿las estrellas explotan cada segundo?", 0),
("¿los unicornios respiran fuego?", 0),
("¿el fuego se convierte en agua?", 0),
("¿las nubes son rojas?", 0),
("¿el hielo es de color negro?", 0),
("¿los ríos fluyen hacia la luna?", 0),
("¿el cielo se convierte en líquido por la noche?", 0),
("¿los humanos pueden volar sin alas?", 0),
("¿el agua quema como fuego?", 0),
("¿los peces bailan salsa?", 0),
("¿el sol es pequeño como una pelota?", 0),
("¿el vapor de agua es sólido como piedra?", 0),
("¿los árboles caminan en el invierno?", 0),
("¿las montañas son líquidas por naturaleza?", 0),
("¿el hielo flota en el aire?", 0),
("¿los perros respiran bajo el agua?", 0),
("¿el agua nunca se evapora?", 0),
("¿los gatos vuelan en el cielo?", 0),
("¿la luna está hecha de queso?", 0),
("¿el sol es frío por la noche?", 0),
("¿las nubes caminan por la tierra?", 0),
("¿los ríos corren hacia arriba siempre?", 0),
("¿el fuego no da calor?", 0),
("¿el hielo es gaseoso?", 0),
("¿los humanos vuelan al espacio sin cohete?", 0),
("¿el aire es líquido?", 0),
("¿los peces caminan sobre el hielo?", 0),
("¿el agua se convierte en fuego por sí sola?", 0),
("¿los árboles vuelan con el viento?", 0),
("¿las plantas se mueven a grandes distancias?", 0),
("¿el sol desaparece cada mañana?", 0),
("¿el cielo es rosa por la noche?", 0),
("¿los unicornios producen leche?", 0),
("¿el fuego se convierte en hielo al tocarlo?", 0),
("¿el hielo camina por sí solo?", 0),
("¿las nubes vuelan hacia abajo?", 0),
("¿los ríos corren hacia la luna?", 0),
("¿los gatos vuelan a la luna?", 0),
("¿los perros bailan salsa?", 0),
("¿el agua es gaseosa a temperatura ambiente?", 0),
("¿los peces cantan canciones?", 0),
("¿el sol se apaga en segundos?", 0),
("¿los árboles bailan tango?", 0),
("¿las montañas vuelan?", 0),
("¿el hielo quema al tocarlo?", 0),
("¿los humanos respiran fuego?", 0),
("¿el vapor de agua es venenoso?", 0),
("¿el aire es sólido?", 0),
("¿las nubes son líquidas?", 0),
("¿el sol se mueve solo cada segundo?", 0),
("¿el agua se convierte en piedra instantáneamente?", 0),
("¿los peces vuelan fuera del agua?", 0),
("¿el hielo es naranja?", 0),
("¿los gatos nadan en el cielo?", 0),
("¿las plantas producen fuego?", 0),
("¿el fuego es sólido como piedra?", 0),
("¿los perros vuelan al espacio?", 0),
("¿las vacas vuelan por la noche?", 0),
("¿el agua nunca se congela?", 0),
("¿los árboles respiran agua?", 0),
("¿los ríos corren hacia el sol?", 0),
("¿el hielo es líquido a -10 grados?", 0),
("¿el sol se esconde bajo la tierra?", 0),
("¿los peces vuelan hacia la luna?", 0),
("¿las nubes están hechas de vidrio?", 0),
("¿los humanos vuelan solos?", 0),
("¿el vapor de agua quema?", 0),
("¿los unicornios respiran agua?", 0),
("¿el fuego se apaga con hielo?", 0),
("¿las plantas vuelan por la ciudad?", 0),
("¿los árboles bailan hip hop?", 0),
("¿el agua es sólida y líquida al mismo tiempo?", 0),
("¿los gatos vuelan en la lluvia?", 0),
("¿el sol es cuadrado?", 0),
("¿los perros respiran fuego?", 0),
("¿las vacas vuelan por el cielo?", 0),
("¿el hielo se convierte en fuego sin calor?", 0),
("¿los peces vuelan sobre la nieve?", 0),
("¿las nubes son de plástico?", 0),
("¿el fuego es líquido?", 0),
("¿el agua respira?", 0),
("¿los humanos vuelan como pájaros?", 0),
("¿los árboles cantan canciones?", 0),
("¿el hielo se mueve solo?", 0),
("¿las montañas caminan por la tierra?", 0),
("¿los ríos corren hacia el desierto?", 0),
("¿el sol se convierte en hielo?", 0),
("¿los peces vuelan sobre las montañas?", 0),
("¿las nubes bajan a la tierra?", 0),
("¿los gatos vuelan al espacio?", 0),
("¿el agua se convierte en hielo sin frío?", 0),
("¿los perros vuelan a la luna?", 0),
("¿los árboles respiran fuego?", 0),
("¿el fuego se convierte en agua sin enfriarse?", 0),
("¿los unicornios vuelan sobre la ciudad?", 0),
("¿las plantas respiran fuego?", 0),
("¿el hielo flota en el espacio?", 0),
("¿los ríos corren hacia Marte?", 0),
("¿el sol se derrite?", 0),
("¿los gatos vuelan sobre el sol?", 0),
("¿los perros respiran agua?", 0),
("¿el agua es cuadrada?", 0),
("¿las nubes caminan por la ciudad?", 0),
("¿los árboles vuelan por el viento?", 0),
("¿el hielo canta canciones?", 0),
("¿los peces vuelan al espacio?", 0),
("¿los humanos respiran lava?", 0),
("¿el sol se enfría en segundos?", 0),
("¿los perros bailan rock?", 0),
("¿los gatos vuelan al sol?", 0),
("¿las montañas desaparecen de noche?", 0),
("¿el fuego respira?", 0),
("¿los ríos corren hacia Saturno?", 0),
("¿las nubes flotan sobre fuego?", 0),
("¿los árboles bailan reggaetón?", 0),
("¿el hielo es fuego?", 0),
("¿los unicornios vuelan?", 0),
("¿las plantas bailan salsa?", 0),
("¿los gatos respiran fuego?", 0),
("¿los perros vuelan como pájaros?", 0),
("¿el agua desaparece de noche?", 0),
("¿el sol canta canciones?", 0),
("¿los humanos caminan sobre el aire?", 0),
("¿el vapor de agua es sólido como piedra?", 0),
("¿las nubes son fuego?", 0),
("¿el hielo camina sobre la tierra?", 0),
("¿los peces vuelan por la ciudad?", 0),
("¿el fuego respira agua?", 0),
("¿las montañas vuelan sobre los ríos?", 0),
("¿los árboles vuelan al espacio?", 0),
("¿el agua es fuego?", 0),
("¿los gatos vuelan sobre los ríos?", 0),
("¿los perros respiran fuego?", 0),
("¿el hielo baila salsa?", 0),
("¿los peces respiran fuego?", 0),
("¿los unicornios vuelan sobre el sol?", 0),
("¿las plantas respiran agua?", 0),
("¿el vapor de agua se convierte en fuego?", 0),
("¿los humanos vuelan sobre el agua?", 0),
("¿el sol respira?", 0),
("¿los ríos vuelan por la ciudad?", 0),
("¿los gatos vuelan sobre las montañas?", 0),
("¿los perros respiran aire?", 0),
("¿el hielo respira?", 0),
("¿las nubes respiran fuego?", 0),
("¿los árboles vuelan sobre los ríos?", 0),
("¿el agua camina?", 0),
("¿los peces vuelan sobre la luna?", 0),
("¿los unicornios respiran aire?", 0),
("¿las plantas vuelan sobre los ríos?", 0),
("¿el vapor de agua camina?", 0),
("¿los humanos vuelan sobre el sol?", 0),
("¿los perros vuelan sobre la luna?", 0),
("¿los gatos bailan tango?", 0),
("¿el hielo vuela sobre los árboles?", 0),
("¿las montañas vuelan sobre la luna?", 0),
("¿el sol baila salsa?", 0),
("¿los ríos vuelan hacia la luna?", 0),
("¿el agua respira fuego?", 0),
("¿los peces vuelan sobre la ciudad?", 0),
("¿los árboles bailan rock?", 0),
("¿los humanos vuelan sobre los árboles?", 0),
("¿los unicornios caminan sobre el agua?", 0),
("¿las nubes vuelan sobre el sol?", 0),
("¿los gatos vuelan sobre el sol?", 0),
("¿el fuego respira aire?", 0),
("¿los perros vuelan sobre el sol?", 0),
("¿el hielo respira fuego?", 0),
("¿el agua baila salsa?", 0),
("¿los peces respiran aire?", 0),
("¿las plantas vuelan sobre el sol?", 0),
("¿los humanos vuelan sobre Marte?", 0),
("¿los peces no viven bajo el agua?", 0),
("¿el hielo no se derrite?", 0),
("¿los peces no viven bajo el agua?", 0),
("¿los peces no viven bajo el agua?", 0),
("¿los peces no viven bajo el agua?", 0),
]

# ==========================
# 🔁 INCLUIR ERRORES PREVIOS
# ==========================
if os.path.exists(ERROR_LOG):
    errores_df = pd.read_csv(ERROR_LOG)
    errores_extra = list(zip(errores_df["pregunta"], errores_df["respuesta"]))
    dataset.extend(errores_extra)
    print(f"🩹 Se han añadido {len(errores_extra)} ejemplos de errores previos.")

# ==========================
# 🧩 TOKENIZACIÓN
# ==========================
tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_WORDS, oov_token="<UNK>")
texts = [t for t, _ in dataset]
labels = [l for _, l in dataset]
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
labels = np.array(labels)

# ==========================
# 🧠 CARGAR O CREAR MODELO
# ==========================
if os.path.exists(MODEL_PATH):
    print("🔁 Modelo existente encontrado, cargando...")
    model = load_model(MODEL_PATH)
else:
    print("✨ No hay modelo previo, creando uno nuevo...")
    model = keras.Sequential([
        keras.layers.Input(shape=(MAX_LEN,)),
        keras.layers.Embedding(MAX_WORDS, 64, mask_zero=True),
        keras.layers.Bidirectional(
            keras.layers.LSTM(32, return_sequences=False, kernel_regularizer=keras.regularizers.l2(0.01))
        ),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(1, activation='sigmoid')
    ])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("✅ Modelo listo (cargado o creado).")

# ==========================
# 🧷 CALLBACKS DE SEGURIDAD
# ==========================
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1,
    min_delta=0.0001
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# ==========================
# 🏆 CREAR PESOS AUTOMÁTICOS (RECOMPENSA)
# ==========================
sample_weight = np.ones(len(labels))
if 'errores_extra' in locals() and errores_extra:
    for i, (pregunta, _) in enumerate(dataset):
        if pregunta in [e[0] for e in errores_extra]:
            sample_weight[i] = 0.5  # penalizamos errores previos
        else:
            sample_weight[i] = 1.5  # recompensamos aciertos previos

# ==========================
# 🚀 ENTRENAMIENTO
# ==========================
history = model.fit(
    padded,
    labels,
    sample_weight=sample_weight,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1,
    callbacks=[early_stop, checkpoint]
)

# ==========================
# 💡 GUARDAR MÉTRICA PARA EL LOOP
# ==========================
val_acc = history.history.get('val_accuracy', [0])[-1]
with open(VAL_LOG, "w") as f:
    f.write(str(val_acc))

print(f"💾 Modelo guardado en {MODEL_PATH}")
print(f"📊 Validación guardada en {VAL_LOG}")

# ==========================
# 🧪 PRUEBAS DE GENERALIZACIÓN
# ==========================
test_data = [
    # Aquí puedes poner frases de prueba rápidas para validar
     ("¿el agua es húmeda?", 1),
    ("¿el vapor de agua es líquido?", 0),
    ("¿el hielo se derrite?", 1),
    ("¿el agua hierve a 50 grados?", 0),
    ("¿los peces viven bajo el agua?", 1),
    ("¿el agua puede ser sólida?", 1),
    ("¿el agua está seca?", 0),
    ("¿el fuego calienta?", 1),
]

errores = []
for pregunta, esperado in test_data:
    seq = tokenizer.texts_to_sequences([pregunta])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    pred = model.predict(pad, verbose=0)[0][0]
    resultado = 1 if pred > 0.5 else 0
    icono = "✅" if resultado == esperado else "❌"
    print(f"{icono} '{pregunta}' → {'Sí' if resultado else 'No'} ({pred:.2f}) [{'Sí' if esperado else 'No'} esperado]")
    if resultado != esperado:
        errores.append((pregunta, esperado))
        dataset.append((pregunta, esperado))  # reincorporar automáticamente al dataset

# ==========================
# 📓 GUARDAR ERRORES
# ==========================
if errores:
    errores_df = pd.DataFrame(errores, columns=["pregunta", "respuesta"])
    if os.path.exists(ERROR_LOG):
        errores_df_ant = pd.read_csv(ERROR_LOG)
        errores_df = pd.concat([errores_df_ant, errores_df]).drop_duplicates()
    errores_df.to_csv(ERROR_LOG, index=False)
    print(f"\n⚠️ Se guardaron {len(errores)} errores en '{ERROR_LOG}' para mejorar la próxima vez.")
else:
    print("\n🎯 Sin errores detectados. El modelo va fino.")

# ==========================
# 💾 GUARDAR TOKENIZER
# ==========================
tokenizer_path = "trained_models/tokenizer.pkl"
with open(tokenizer_path, "wb") as f:
    pickle.dump(tokenizer, f)
print(f"✅ Tokenizer guardado en {tokenizer_path}")

# ==========================
# 💬 LUGAR PARA NUEVAS FRASES
# ==========================
# dataset.append(("Aquí tu nueva frase", 1 o 0))
# Añade aquí nuevas frases de entrenamiento para que sean incluidas en la próxima ronda
