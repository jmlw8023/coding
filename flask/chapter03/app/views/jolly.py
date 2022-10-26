from flask import Blueprint, request, redirect, render_template, url_for, session
import functools

jolly = Blueprint('jolly', __name__)

DATA = {
    1001 : {'name': 'Python编程', 'author':'明智', 'price':23.45 }, 
    1002 : {'name': '深度学习开始之路', 'author':'DL之神', 'price':56.85 }, 
    1003 : {'name': 'Flask处步使用', 'author':'培训', 'price':33.66 }, 
    1004 : {'name': 'OpenCV 从精通到放弃', 'author':'jolly', 'price':73.01 } 
}



def auth(fun):
    @functools.wraps(fun)
    def demo(*args, **kwargs):
        username = session.get('cipher')
        if not username:
            return redirect('/login')
        return fun(*args, **kwargs)
    return demo

@jolly.route('/login', methods=['GET', 'POST'], endpoint='in')
def login():
    if request.method == 'GET':
        return render_template('login.html')

    if request.method == 'POST':
    
        user = request.form.get('user')
        pwd = request.form.get('pwd')
        print('user = ', user, '\t', 'password = ', pwd)
        if user == 'jolly' and pwd == '123':
            print('认证成功！！！！！！')
            session['cipher'] = 'abc'
            # return redirect(url_for('jhm'))   # why error ???
            return redirect('/home')

        else:
            error = '输入的密码或者账户有误'
            return render_template('error.html', error=error)




@jolly.route('/index', endpoint='idx')

def index():
    
    return render_template('index.html')


@jolly.route('/home', methods=['GET', 'POST'], endpoint='jhm')
@auth
def home():
    data_dict = DATA

    return render_template('home.html', data_dict=data_dict)



@jolly.route('/edit', methods=['GET', 'POST'])
@auth
def edit():
    # 由 上面的@auth 路由映射解决了，不必要重复代码
    # username = session.get('cipher')
    # if not username:
    #     return redirect('/login')
    index = request.args.get('nid')
    print(index)

    if request.method == 'GET':
        info = DATA[int(index)]
        info['key'] = index
        print(info)
        return render_template('edit.html', info=info, index=index)
    
    if request.method == 'POST':
        bk_id = request.form.get('bk_id')
        name = request.form.get('name')
        author = request.form.get('author')
        price = request.form.get('price')
        print(' = ' * 15)
        print(bk_id)
        print(name)
        # DATA[int(index)] = int(bk_id)   # 字典key 重名 'int' object does not support item assignment
        DATA[int(index)]['name'] = name
        DATA[int(index)]['author'] = author
        DATA[int(index)]['price'] = price

        return redirect('/home')


@jolly.route('/del', methods=['GET', 'POST'])
@auth
def delete():
    inx = request.args.get('d_key')
    # inx = 0
    print(' - ' * 15)
    print(inx)
    # print(request.form.get('d_key'))
    if request.method == 'GET':
        del DATA[int(inx)]

        return redirect('/home')

# # @jolly.route('/del/<int:ind>')  # 指定的ind输入变量类型为 int
# @jolly.route('/del/<ind>')  # 默认输入参数ind 为str类型
# def delete(ind):
#     print('index = ', ind, '\t', 'index type  = ', type(ind))
#     del DATA[int(ind)]

#     return redirect('/home')


