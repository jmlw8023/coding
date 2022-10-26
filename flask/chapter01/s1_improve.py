import functools
from flask import Flask, render_template, jsonify, request, redirect, url_for, session



app = Flask(__name__)

app.secret_key = 'suibian'

DATA_DICT = {
    '1' : {'name' : 'jolly', 'age' : 26},
    '2' : {'name' : 'lusse', 'age' : 23},
    '3' : {'name' : 'sunny', 'age' : 25},
    '4' : {'name' : '张三三', 'age' : 20}
}



def auth(func):
    @functools.wraps(func)
    def demo(*args, **kwargs):
        username = session.get('xxx')
        if not username:
            return redirect(url_for('idx'))
        return func(*args, **kwargs)
    
    return demo
        


@app.route('/')
def root():
    return '<h2>hello flask</h2>'



@app.route('/login', methods=['GET', 'POST' ])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    # print('login')
    if request.method == 'POST':
        user = request.form.get('user')
        pwd = request.form.get('pwd')
        print(' = ' * 15)
        print(user, '\t', pwd)
        if user == 'jolly' and pwd == '123':
            session['xxx'] = 'jolly'
            return redirect('/index')
    
        error = '用户名或者密码错误！！！'
        return render_template('login.html', error=error)



@app.route('/index', endpoint='idx')    # endponint 是函数重名设置
@auth
def index():
    # username = session.get('xxx')
    # if not username:
    #     return redirect('/login')
    data_dict = DATA_DICT

    return render_template('index.html', data_dict=data_dict)


@app.route('/edit', methods=['GET', 'POST'])
@auth
def edit():
    # username = session.get('xxx')
    # if not username:
    #     return redirect(url_for('idx'))

    nid = request.args.get('nid')  # 获取页面数据的 id号   GET 形势的参数传递
    if request.method == "GET":
        # print(nid)
        info = DATA_DICT[nid]
        return render_template('edit.html', info=info)
    if request.method == "POST":
        user = request.form.get('user')   # 字段名称来自 edit.html 的name中   POST 方式的参数传递
        age = request.form.get('age') 

        DATA_DICT[nid]['name'] = user
        DATA_DICT[nid]['age'] = age
        # return render_template('/index')
        return redirect(url_for('idx'))


    # return '修改'

# @app.route('/del')
# @app.route('/del/<int:nid>')
@app.route('/del/<nid>')
@auth
def delete(nid):
    # username = session.get('xxx')
    # if not username:
    #     return redirect('/login')
    # nid = request.args.get('nid')

    # print(nid)
    # opertation
    del DATA_DICT[nid]
    # return redirect('/index')
    return redirect(url_for('idx'))
    # return '删除'


if __name__ == '__main__':
    app.run(port=5050, debug=True)









