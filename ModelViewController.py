from sklearn.metrics import accuracy_score;
from sklearn.linear_model import LogisticRegression;
from sklearn.model_selection import train_test_split;
from flask import Flask, request, jsonify;
import numpy as np;
import pandas as pd;
from PIL import Image

# setting up the entire prediction model
X = np.load('image.npz')['arr_0'];
Y = pd.read_csv('labels.csv');

# train_test_split(X, Y, train_size=0.75, test_size=0.25);

lr = LogisticRegression(solver='saga', multi_class='multinomial');
model = lr.fit(X, Y);

y_pred = model.predict(X);
accuracy = accuracy_score(Y, y_pred);

print('accuracy of the model => ' + str(accuracy));

def image_prediction(image):
    im_pil = Image.open(image)
    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized, pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized-min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_pred = model.predict(test_sample)

# setting up the Flask part;
FlaskApp = Flask('ModelViewController');
@FlaskApp.route('/image-prediction', methods = ['POST'])
def image_predictionFlask():
    image = request.files.get('alphabet');
    prediction = image_prediction(image);
    return jsonify({
        'predictions' : prediction,
    }, 200)
    
FlaskApp.run(debug = True);
