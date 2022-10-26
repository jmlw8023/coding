from flask import Blueprint


lussy = Blueprint('lussy', __name__)


@lussy.route('/l1')
def l1():
    return 'lussy 1'

@lussy.route('/l2')
def l2():
    return 'lussy 2'

