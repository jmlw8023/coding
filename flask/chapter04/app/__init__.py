
from flask import Flask, render_template, request
from .views.jolly import jolly
import os


def create_app():

    app = Flask(__name__)

    app.secret_key = 'abcd'

    basedir = os.path.abspath(os.path.dirname(__file__))

    app.config.update(
        UPLOADED_PATH = os.path.join(basedir, 'upload'),
    )

    @app.route('/')
    def root():
        return render_template('root.html')


    @app.route('/in')
    def upload():
        import os
        if request.method == 'POST':
            f = request.files.get('file')
            file_path = os.path.join(app.config['UPLOADED_PATH'], f.filename)
            f.save(file_path)
        return render_template('upload.html')





    app.register_blueprint(jolly, url_prefix='/admin')   # url_prefix  访问url地址前面增加字符串



    return app
