from flask import Blueprint


jolly = Blueprint('jolly', __name__)

@jolly.route('/j1')
def j1():
    return 'jolly 1'


@jolly.route('/j2')
def j2():
    return 'jolly 2'



