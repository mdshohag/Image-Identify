import os
import random
from flask import Flask, render_template, request, url_for, send_from_directory
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16

app = Flask(__name__)

UPLOAD_FOLDER = './image/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = VGG16()

# Route to serve uploaded images
@app.route('/image/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']

     # Generate a random number for the filename
    random_number = random.randint(100000, 999999)
    file_ext = os.path.splitext(imagefile.filename)[1]  # Get the file extension
    new_filename = f"{random_number}{file_ext}" 


    image_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
    imagefile.save(image_path)

    # Process the image for prediction
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]

    classification = '%s (%.2f%%)' % (label[1], label[2] * 100)

    # Generate the URL for the uploaded image
    image_url = url_for('uploaded_file', filename=new_filename)

    return render_template('index.html', prediction=classification, image_url=image_url)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
