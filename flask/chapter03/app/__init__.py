from flask import Flask, redirect, render_template, url_for
from .views.jolly import  jolly




def create_app():
    app = Flask(__name__)
    app.secret_key = 'awbhogheo'
    
    @app.route('/')
    def root():
        return '<h1> Home page </h1>'

    app.register_blueprint(jolly)
    # app.register_blueprint(jolly, url_prefix='/admin')   # url_prefix  访问url地址前面增加字符串


    return app



