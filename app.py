import os, datetime
from werkzeug.utils import secure_filename
from flask import Flask, render_template, jsonify, request, url_for, redirect

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'run_data')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('select_files.html')


@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        tf = request.files['transcript_file']
        af = request.files['audio_file']
        tf_filename = secure_filename(tf.filename)
        af_filename = secure_filename(af.filename)
        TRANS_FILE_PATH = os.path.join(app.config['UPLOAD_FOLDER'], tf_filename)
        AUDIO_FILE_PATH = os.path.join(app.config['UPLOAD_FOLDER'], af_filename)
        app.config['TRANS_FILE_PATH'] = TRANS_FILE_PATH
        app.config['AUDIO_FILE_PATH'] = AUDIO_FILE_PATH
        tf.save(app.config['TRANS_FILE_PATH'])
        af.save(app.config['AUDIO_FILE_PATH'])

        print(app.config['AUDIO_FILE_PATH'])
        print(app.config['TRANS_FILE_PATH'])

    return render_template("result.html", TRANS_FILE_PATH=str(TRANS_FILE_PATH),AUDIO_FILE_PATH=str(AUDIO_FILE_PATH))


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
