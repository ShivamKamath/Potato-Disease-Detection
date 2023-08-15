from flask import Flask, render_template, request
from keras.models import load_model
from keras.utils import load_img,img_to_array
import numpy as np

app = Flask(__name__)

dic = {0 : 'early blight', 1 : 'late blight', 2 :"healthy leaf"}


model = load_model('model_inception.h5')


def predict_label(img_path):
	i = load_img(img_path, target_size=(256,256))
	i = img_to_array(i)
	i = i.reshape(1, 256,256,3)
	predictions = np.argmax(model.predict(i),axis=1)

	return dic[predictions[0]]


# routes
@app.route("/", methods=['GET', 'POST'])
def kuch_bhi():
	return render_template("index1.html")

@app.route("/about")
def about_page():
	return "About You..!!!"






@app.route("/submit", methods = ['GET', 'POST'])
def get_hours():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)



	return render_template("index1.html", prediction = p, img_path = img_path)





if __name__ =='__main__':
	#app.debug = True
	app.run(port=3000, debug = True)