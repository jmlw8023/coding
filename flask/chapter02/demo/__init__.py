from flask import Flask
from .views.jolly import jolly      #  蓝图构建个例
from .views.lussy import lussy


def create_app():
    app = Flask(__name__)
    app.secret_key = 'ashgbgv'


    @app.route('/index')
    def index():
        return ' flask index'

    # from .views.jolly import jolly      #  蓝图构建个例
    # from .views.lussy import lussy

    # app.register_blueprint(jolly)     # url ： http://127.0.0.1:5050/l1
    # app.register_blueprint(lussy)
    app.register_blueprint(jolly, url_prefix='/admin')   # url_prefix  访问url地址前面增加字符串
    app.register_blueprint(lussy, url_prefix='/web')    # url： http://127.0.0.1:5050/web/l1


    return app




