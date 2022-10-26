from app import create_app






demo = create_app()

if __name__ == '__main__':
    
    
    demo.run(debug=True, port=5050)



