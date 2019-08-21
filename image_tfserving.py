import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory
import cv2
import numpy as np
import utils
from PIL import Image
from plate_color_detect import detect_color
import os
import json
import requests
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



input_size      = 416

UPLOAD_FOLDER = '/home/cupcon/sqs/yolov3-tf/plate_rec_tfserving/pre_out/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS







@app.route('/', methods=['GET', 'POST'])
def upload_file():
    for i in os.listdir("./pre_out/"):
        if os.path.splitext(i)[1] == '.jpg':
            os.remove("./pre_out/" + i)
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    print(filename)
    g_plate, b_plate, y_plate = detect_color("pre_out/"+filename)
    is_img(g_plate, 'g')
    is_img(b_plate, 'b')
    is_img(y_plate, 'y')
    os.remove("pre_out/" + filename)
    for i in os.listdir("./pre_out/"):
        if os.path.splitext(i)[1] == '.jpg':
            return send_from_directory(app.config['UPLOAD_FOLDER'],
                                i)

def is_img(img_cv, color):
    j = 0
    if len(img_cv) != 0:
        print("---1312--------------")
        for i in range (len(img_cv)):
            im_cv_r = cv2.resize(img_cv[i], (1300, 414))
            gray = cv2.cvtColor(im_cv_r, cv2.COLOR_BGR2GRAY)
            equ = cv2.equalizeHist(gray)
            gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
            median = cv2.medianBlur(gaussian, 3)
            original_image = median
            original_image_size = original_image.shape[:2]
            image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
            image_data = image_data[np.newaxis, ...]
            data = json.dumps({"signature_name": "serving_default",
                   "instances": image_data.tolist()})
            headers = {"content-type": "application/json"}
            num_classes=65
            json_response = requests.post(
                'http://172.20.81.241:8500/v1/models/yolov3:predict', data=data, headers=headers)
            predictions = json.loads(json_response.text)['predictions']

            pred_sbbox, pred_mbbox, pred_lbbox =predictions[0]['pred_sbbox'],predictions[0]['pred_mbbox'],predictions[0]['pred_lbbox']
            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)
            bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
            bboxes = utils.nms(bboxes, 0.45, method='nms')
            if np.array(bboxes).shape[0] > 6:
                image = utils.draw_bbox(im_cv_r, bboxes)
                # print(image)
                name = color +'im' + str(i) + '.jpg'
                path = os.path.join("./pre_out/", name)
                cv2.imwrite(path,image)
                print("-------------")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
    # detect_color('/home/cupcon/sqs/yolov3-tf/plate_rec_tfserving/WechatIMG14.jpeg')
    # print('asd')
