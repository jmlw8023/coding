
pip install nuitka

nuitka --standalone --mingw64  --windows-disable-console --enable-plugin=pyqt5 --output-dir=results mainwindow_demo.py
