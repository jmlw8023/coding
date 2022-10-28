from re import T
from unicodedata import name
from app import create_app






if __name__ == '__main__':
    demo = create_app()

    demo.run(debug=True, port=5050)


