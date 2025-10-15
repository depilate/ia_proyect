import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# ==========================
# CONFIGURACIÓN
# ==========================
MAX_WORDS = 5000  # Vocabulario máximo
MAX_LEN = 10     # Longitud de secuencia
EPOCHS = 300      # Número de épocas de entrenamiento
BATCH_SIZE = 4    # Tamaño de lote

# ==========================
# DATOS DE ENTRENAMIENTO
# ==========================
# Añade aquí tus datos de entrenamiento
# Formato: (texto, etiqueta)
# etiqueta: 1 = "Sí", 0 = "No"

training_data = [
    # ==========================================
    # PREGUNTAS OBVIAS - SÍ (1)
    # ==========================================
    ("¿el agua moja?", 1),
    ("¿el fuego quema?", 1),
    ("¿el hielo es frío?", 1),
    ("¿las piedras son duras?", 1),
    ("¿el sol brilla?", 1),
    ("¿la noche es oscura?", 1),
    ("¿el cielo es azul?", 1),
    ("¿las nubes flotan?", 1),
    ("¿los pájaros vuelan?", 1),
    ("¿los peces nadan?", 1),
    ("¿los gatos maúllan?", 1),
    ("¿los perros ladran?", 1),
    ("¿las vacas mugen?", 1),
    ("¿las abejas hacen miel?", 1),
    ("¿los árboles tienen hojas?", 1),
    ("¿las flores tienen pétalos?", 1),
    ("¿el café tiene cafeína?", 1),
    ("¿el azúcar es dulce?", 1),
    ("¿el limón es ácido?", 1),
    ("¿la sal es salada?", 1),
    
    # ==========================================
    # VERDADES CIENTÍFICAS - SÍ (1)
    # ==========================================
    ("¿la tierra es redonda?", 1),
    ("¿la gravedad existe?", 1),
    ("¿el oxígeno es necesario para respirar?", 1),
    ("¿dos más dos es cuatro?", 1),
    ("¿el agua hierve a 100 grados?", 1),
    ("¿la luz viaja rápido?", 1),
    ("¿el corazón bombea sangre?", 1),
    ("¿los humanos necesitan dormir?", 1),
    ("¿las plantas hacen fotosíntesis?", 1),
    ("¿los huesos son duros?", 1),
    ("¿el cerebro controla el cuerpo?", 1),
    ("¿los dientes sirven para masticar?", 1),
    ("¿los ojos sirven para ver?", 1),
    ("¿las orejas sirven para oír?", 1),
    ("¿la nariz sirve para oler?", 1),
    
    # ==========================================
    # SOBRE LA IA - SÍ (1)
    # ==========================================
    ("¿eres una inteligencia artificial?", 1),
    ("¿puedes procesar texto?", 1),
    ("¿estás funcionando ahora?", 1),
    ("¿puedes responder preguntas?", 1),
    ("¿me estás entendiendo?", 1),
    ("¿eres un programa?", 1),
    ("¿usas electricidad?", 1),
    ("¿tienes un modelo entrenado?", 1),
    ("¿estás disponible?", 1),
    ("¿puedes ayudarme?", 1),
    ("¿procesas información?", 1),
    ("¿eres digital?", 1),
    ("¿estás en una computadora?", 1),
    ("¿usas redes neuronales?", 1),
    ("¿puedes aprender?", 1),
    
    # ==========================================
    # TECNOLOGÍA Y COMPUTACIÓN - SÍ (1)
    # ==========================================
    ("¿las computadoras usan electricidad?", 1),
    ("¿internet conecta el mundo?", 1),
    ("¿los teléfonos pueden hacer llamadas?", 1),
    ("¿las tablets tienen pantallas táctiles?", 1),
    ("¿python es un lenguaje de programación?", 1),
    ("¿los videojuegos son digitales?", 1),
    ("¿wifi es conexión inalámbrica?", 1),
    ("¿los emails se envían por internet?", 1),
    ("¿los datos se guardan en servidores?", 1),
    ("¿las contraseñas protegen cuentas?", 1),
    
    # ==========================================
    # COSAS COTIDIANAS - SÍ (1)
    # ==========================================
    ("¿las sillas sirven para sentarse?", 1),
    ("¿las camas son para dormir?", 1),
    ("¿los zapatos se usan en los pies?", 1),
    ("¿los sombreros van en la cabeza?", 1),
    ("¿los libros tienen páginas?", 1),
    ("¿los lápices sirven para escribir?", 1),
    ("¿las tijeras cortan papel?", 1),
    ("¿los relojes marcan la hora?", 1),
    ("¿las llaves abren cerraduras?", 1),
    ("¿los espejos reflejan?", 1),
    ("¿las ventanas dejan pasar luz?", 1),
    ("¿las puertas se abren y cierran?", 1),
    ("¿las lámparas dan luz?", 1),
    ("¿los refrigeradores enfrían?", 1),
    ("¿los hornos calientan?", 1),
    
    # ==========================================
    # COMIDA - SÍ (1)
    # ==========================================
    ("¿la pizza es comida?", 1),
    ("¿las hamburguesas tienen carne?", 1),
    ("¿el pan se hace con harina?", 1),
    ("¿la pasta es italiana?", 1),
    ("¿el sushi lleva arroz?", 1),
    ("¿las frutas son saludables?", 1),
    ("¿las verduras son nutritivas?", 1),
    ("¿el chocolate es dulce?", 1),
    ("¿el helado es frío?", 1),
    ("¿la sopa es líquida?", 1),
    
    # ==========================================
    # CULTURA Y ENTRETENIMIENTO - SÍ (1)
    # ==========================================
    ("¿la música tiene sonidos?", 1),
    ("¿las películas cuentan historias?", 1),
    ("¿los deportes requieren actividad física?", 1),
    ("¿el fútbol se juega con pelota?", 1),
    ("¿el baloncesto usa un balón?", 1),
    ("¿los libros tienen historias?", 1),
    ("¿las pinturas son arte?", 1),
    ("¿las esculturas son tridimensionales?", 1),
    ("¿la danza involucra movimiento?", 1),
    ("¿el teatro tiene actores?", 1),
    
    # ==========================================
    # ABSURDOS TOTALES - NO (0)
    # ==========================================
    ("¿las vacas vuelan al espacio?", 0),
    ("¿los peces caminan por la calle?", 0),
    ("¿las piedras cantan ópera?", 0),
    ("¿los árboles bailan salsa?", 0),
    ("¿las nubes son de algodón de azúcar?", 0),
    ("¿la luna es de queso?", 0),
    ("¿el sol sale por el oeste?", 0),
    ("¿las estrellas son diamantes?", 0),
    ("¿los unicornios existen?", 0),
    ("¿los dragones viven en mi garaje?", 0),
    ("¿las hadas tienen alas de verdad?", 0),
    ("¿los zombies caminan entre nosotros?", 0),
    ("¿los vampiros beben jugo de tomate?", 0),
    ("¿los fantasmas tocan el piano?", 0),
    ("¿los extraterrestres venden tacos?", 0),
    
    # ==========================================
    # CONTRADICCIONES FÍSICAS - NO (0)
    # ==========================================
    ("¿el fuego es frío?", 0),
    ("¿el hielo quema?", 0),
    ("¿el agua es seca?", 0),
    ("¿las piedras flotan naturalmente?", 0),
    ("¿el aire es sólido?", 0),
    ("¿la noche es brillante?", 0),
    ("¿el día es oscuro?", 0),
    ("¿la gravedad empuja hacia arriba?", 0),
    ("¿los pájaros nadan bajo el agua?", 0),
    ("¿los peces respiran aire directamente?", 0),
    ("¿las plantas caminan?", 0),
    ("¿los árboles corren maratones?", 0),
    ("¿el sol es negro?", 0),
    ("¿el cielo es verde fluorescente?", 0),
    ("¿la nieve es caliente?", 0),
    
    # ==========================================
    # MATEMÁTICAS INCORRECTAS - NO (0)
    # ==========================================
    ("¿dos más dos es cinco?", 0),
    ("¿tres por tres es diez?", 0),
    ("¿uno más uno es tres?", 0),
    ("¿cinco menos dos es uno?", 0),
    ("¿diez dividido dos es tres?", 0),
    ("¿cero es mayor que uno?", 0),
    ("¿los números negativos son positivos?", 0),
    
    # ==========================================
    # SOBRE LA IA - NO (0)
    # ==========================================
    ("¿eres un humano?", 0),
    ("¿tienes cuerpo físico?", 0),
    ("¿puedes tocar objetos?", 0),
    ("¿puedes comer pizza?", 0),
    ("¿puedes dormir?", 0),
    ("¿tienes sentimientos reales?", 0),
    ("¿puedes respirar?", 0),
    ("¿tienes cinco dedos?", 0),
    ("¿puedes volar físicamente?", 0),
    ("¿vives en una casa?", 0),
    ("¿tienes mascotas?", 0),
    ("¿vas al supermercado?", 0),
    ("¿conduces un coche?", 0),
    ("¿puedes nadar en la piscina?", 0),
    ("¿necesitas comer para vivir?", 0),
    
    # ==========================================
    # TECNOLOGÍA IMPOSIBLE - NO (0)
    # ==========================================
    ("¿las computadoras funcionan sin electricidad?", 0),
    ("¿internet existe sin cables ni señales?", 0),
    ("¿los teléfonos pueden leer mentes?", 0),
    ("¿las tablets crecen en árboles?", 0),
    ("¿los videojuegos son reales?", 0),
    ("¿puedes descargar una pizza?", 0),
    ("¿wifi viaja por tuberías de agua?", 0),
    ("¿los emails se envían por palomas?", 0),
    
    # ==========================================
    # ANIMALES IMPOSIBLES - NO (0)
    # ==========================================
    ("¿los gatos ladran?", 0),
    ("¿los perros maúllan?", 0),
    ("¿las vacas ponen huevos?", 0),
    ("¿los pollitos dan leche?", 0),
    ("¿los elefantes vuelan naturalmente?", 0),
    ("¿las jirafas viven bajo el agua?", 0),
    ("¿los pingüinos viven en el desierto?", 0),
    ("¿los camellos viven en el ártico?", 0),
    ("¿las serpientes tienen patas?", 0),
    ("¿los peces respiran fuera del agua?", 0),
    ("¿las ballenas viven en la montaña?", 0),
    ("¿los tiburones hacen la fotosíntesis?", 0),
    
    # ==========================================
    # COMIDA ABSURDA - NO (0)
    # ==========================================
    ("¿la pizza crece en árboles?", 0),
    ("¿las hamburguesas vuelan?", 0),
    ("¿el pan es un animal?", 0),
    ("¿la pasta es un metal?", 0),
    ("¿el sushi es una roca?", 0),
    ("¿las frutas son venenosas siempre?", 0),
    ("¿el chocolate es salado?", 0),
    ("¿el helado está caliente?", 0),
    ("¿la sopa es sólida?", 0),
    ("¿el agua es un alimento sólido?", 0),
    
    # ==========================================
    # COSAS COTIDIANAS IMPOSIBLES - NO (0)
    # ==========================================
    ("¿las sillas vuelan solas?", 0),
    ("¿las camas caminan por la noche?", 0),
    ("¿los zapatos se usan en las manos?", 0),
    ("¿los sombreros van en los pies?", 0),
    ("¿los libros se comen?", 0),
    ("¿los lápices son comida?", 0),
    ("¿las tijeras sirven para peinar?", 0),
    ("¿los relojes pueden cocinar?", 0),
    ("¿las llaves son dulces?", 0),
    ("¿los espejos absorben luz?", 0),
    
    # ==========================================
    # CULTURA IMPOSIBLE - NO (0)
    # ==========================================
    ("¿la música es invisible y muda?", 0),
    ("¿las películas son reales?", 0),
    ("¿el fútbol se juega con una sandía?", 0),
    ("¿el baloncesto usa una pelota cuadrada?", 0),
    ("¿los libros no tienen palabras?", 0),
    ("¿las pinturas son invisibles?", 0),
    ("¿las esculturas son líquidas?", 0),
    
    # ==========================================
    # HISTORIA Y GEOGRAFÍA - SÍ (1)
    # ==========================================
    ("¿españa está en europa?", 1),
    ("¿parís es la capital de francia?", 1),
    ("¿el everest es una montaña?", 1),
    ("¿el amazonas es un río?", 1),
    ("¿el sahara es un desierto?", 1),
    ("¿roma está en italia?", 1),
    ("¿el pacífico es un océano?", 1),
    
    # ==========================================
    # HISTORIA Y GEOGRAFÍA - NO (0)
    # ==========================================
    ("¿españa está en áfrica?", 0),
    ("¿parís es la capital de españa?", 0),
    ("¿el everest está bajo el agua?", 0),
    ("¿el sahara es un océano?", 0),
    ("¿roma está en japón?", 0),
    ("¿la antártida es tropical?", 0),
    
    # ==========================================
    # SUPER ABSURDOS FINALES - NO (0)
    # ==========================================
    ("¿los teléfonos crecen en macetas?", 0),
    ("¿las nubes son hechas de algodón?", 0),
    ("¿los coches funcionan con unicornios?", 0),
    ("¿los ordenadores comen hamburguesas?", 0),
    ("¿el wifi viaja en burro?", 0),
    ("¿las estrellas son pegajosas?", 0),
    ("¿la lluvia sube hacia arriba?", 0),
    ("¿los volcanes hacen helado?", 0),
    ("¿las montañas son blandas como almohadas?", 0),
    ("¿los ríos fluyen en círculos?", 0),
    ("¿el arcoíris es comestible?", 0),
    ("¿las piedras pueden llorar?", 0),
    ("¿los cactus son esponjosos?", 0),
    ("¿el viento tiene forma cuadrada?", 0),
    ("¿la tierra es plana como una pizza?", 0),
    ("¿las plantas pueden hablar español?", 0),
    ("¿los robots tienen hambre?", 0),
    ("¿las nubes tienen wifi gratis?", 0),
    ("¿los semáforos bailan por la noche?", 0),
    ("¿las carreteras son de chocolate?", 0),
    ("ruben es gay", 0),
]

# ==========================
# PREPARACIÓN DE DATOS
# ==========================
print("📊 Preparando datos de entrenamiento...")

# Separar textos y etiquetas
texts = [item[0] for item in training_data]
labels = np.array([item[1] for item in training_data])

# Crear y entrenar tokenizer
tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(texts)

# Convertir textos a secuencias numéricas
sequences = tokenizer.texts_to_sequences(texts)

# Padding para que todas tengan la misma longitud
X_train = pad_sequences(sequences, maxlen=MAX_LEN)
y_train = labels

print(f"✅ Datos preparados: {len(X_train)} ejemplos")
print(f"📝 Vocabulario: {len(tokenizer.word_index)} palabras únicas")

# ==========================
# CREAR MODELO
# ==========================
print("\n🧠 Creando red neuronal...")

model = keras.Sequential([
    keras.layers.Input(shape=(MAX_LEN,)),
    keras.layers.Embedding(MAX_WORDS, 32),  # Añadido: convierte números en vectores
    keras.layers.GlobalAveragePooling1D(),  # Añadido: reduce dimensionalidad
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),  # Añadido: previene overfitting
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("✅ Modelo creado")
model.summary()

# ==========================
# ENTRENAMIENTO
# ==========================
print(f"\n🚀 Iniciando entrenamiento ({EPOCHS} épocas)...\n")

history = model.fit(
    X_train, 
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,  # 20% para validación
    verbose=1
)

# ==========================
# GUARDAR MODELO
# ==========================
print("\n💾 Guardando modelo entrenado...")

# Crear carpeta si no existe
os.makedirs("trained_models", exist_ok=True)

# Guardar modelo
model.save("trained_models/gripen_model.h5")
print("✅ Modelo guardado en: trained_models/gripen_model.h5")

# Guardar tokenizer
import pickle
with open("trained_models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("✅ Tokenizer guardado en: trained_models/tokenizer.pkl")

# ==========================
# EVALUACIÓN
# ==========================
print("\n📈 Resultados del entrenamiento:")
print(f"Precisión final: {history.history['accuracy'][-1]:.2%}")
print(f"Pérdida final: {history.history['loss'][-1]:.4f}")

if 'val_accuracy' in history.history:
    print(f"Precisión validación: {history.history['val_accuracy'][-1]:.2%}")

# ==========================
# PRUEBAS
# ==========================
print("\n🧪 Probando el modelo entrenado:")

test_phrases = [
    "¿el agua es húmeda?",
    "¿puedes volar?",
    "¿estás funcionando bien?",
    "¿eres humano?"
]

for phrase in test_phrases:
    seq = tokenizer.texts_to_sequences([phrase])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(padded, verbose=0)[0][0]
    result = "Sí" if pred > 0.5 else "No"
    print(f"  '{phrase}' → {result} ({pred:.2f})")

print("\n✨ ¡Entrenamiento completado!")
print("💡 Para usar el modelo entrenado, modifica manager.py para cargar los pesos guardados.")