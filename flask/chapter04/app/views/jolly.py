from flask import request, redirect, render_template, url_for, Blueprint, session
import functools


def is_login(fun):
    @functools.wraps(fun)
    def demo(*args, **kwargs):
        username = session.get('user')
        if not username:
            return redirect('login')
        return fun(*args, **kwargs)
    return demo




jolly = Blueprint('jolly', __name__)



# @jolly.route('/lg/<name>')
# def test(name):
#     return render_template('lg.html', names=name)

def fun(arg):
    # return f'<h3>你好啊 {arg}</h3>'
    return f'你好啊 {arg}'

@jolly.route('/lg')
def test():
    name_lis = ['笑话', 'jolly', '温蔼']
    return render_template('lg.html', names=name_lis, func=fun)

@jolly.route('/index')
@is_login
def index():
    return render_template('index.html')

@jolly.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'GET':
        return render_template('login.html')
    
    if request.method == 'POST':
        user = request.form.get('user')
        pwd = request.form.get('pwd')
        session['user'] = 'abcd'
        if user == 'jolly' and pwd == '123':
            return redirect('/admin/index')
        error_info = '用户名或者密码错误！！'
        return render_template('error.html', error=error_info)
    








