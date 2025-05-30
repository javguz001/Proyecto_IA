from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Cargar modelo
model = load_model("modelo_mascarillas.keras")

# Etiquetas
etiquetas = ['Con mascarilla', 'Sin mascarilla']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediccion = None
    filename = None

    if request.method == 'POST':
        img_file = request.files['image']
        if img_file:
            filename = img_file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img_file.save(filepath)

            # Procesar imagen
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Predecir
            resultado = model.predict(img_array)
            prediccion = etiquetas[int(resultado[0][0] > 0.5)]

    return render_template('index.html', prediccion=prediccion, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
