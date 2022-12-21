<<<<<<< HEAD

# from data.config import Config
from flask import Flask, render_template, url_for
from .views.jolly import jolly



def create_app():
    app = Flask(__name__)

    app.secret_key = 'jolly'
    # app.config.from_object(config)   # error
    # app.config.from_pyfile('config.py')
    # app.config.from_object(Config)

    @app.route('/')
    def root():
        s = '欢迎来到 自动标注首页'

        return render_template('root.html', info=s)

    
    @app.errorhandler(404)
    def page_error(e):
        info = '页面不存在'
        return render_template('404.html', info=info), 404

    app.register_blueprint(jolly)



    return app

=======

# from data.config import Config
from flask import Flask, render_template, url_for
from .views.jolly import jolly



def create_app():
    app = Flask(__name__)

    app.secret_key = 'jolly'
    # app.config.from_object(config)   # error
    # app.config.from_pyfile('config.py')
    # app.config.from_object(Config)

    @app.route('/')
    def root():
        s = '欢迎来到 自动标注首页'

        return render_template('root.html', info=s)

    
    @app.errorhandler(404)
    def page_error(e):
        info = '页面不存在'
        return render_template('404.html', info=info), 404

    app.register_blueprint(jolly)



    return app

>>>>>>> 3efa523ff0c2328ea326da8c6ab7d5be45143578
