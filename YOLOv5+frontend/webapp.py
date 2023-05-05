import os
import subprocess
import argparse
from flask import Flask, render_template, url_for, send_from_directory
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

app = Flask(__name__)
app.config['SECRET_KEY'] = 'kmkfskpgmfdlkg'
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)

class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos, 'Only images can be uploaded'),
            FileRequired('No file was selected')
        ]
    )
    submit = SubmitField('Upload')


@app.route('/runs/detect/exp/<filename>')
def get_file_detect(filename):
    return send_from_directory('runs/detect/exp', filename)

@app.route('/uploads/<filename>')
def get_file_common(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)

def detect_image(image_path):
    weights_path = 'runs/train/exp/weights/best.pt'
    exp_folder = 'runs/detect/exp'

    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    subprocess.run(['python', 'detect.py', '--source', image_path, '--weights', weights_path, '--img-size', '640', '--conf-thres', '0.25', '--project', 'runs/detect', '--name', 'exp', '--exist-ok'])


@app.route("/", methods=['GET', 'POST'])
def upload_and_detect_image():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_path = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename)
        detect_image(file_path)
        file_url_common = url_for('get_file_common', filename=filename)
        file_url_detect = url_for('get_file_detect', filename=filename)
    else:
        file_url_common = None
        file_url_detect = None
    return render_template('index.html', form=form, file_url_common=file_url_common, file_url_detect=file_url_detect)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=8000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)
