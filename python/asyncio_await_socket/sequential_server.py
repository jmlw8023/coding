# -*- encoding: utf-8 -*-
'''
@File    :   sequential_server.py
@Time    :   2023/01/04 09:52:49
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
link: https://tenthousandmeters.com/blog/python-behind-the-scenes-12-how-asyncawait-works-in-python/
'''

# import packets
# import os
import socket
import threading

def run_server(host='127.0.0.1', port=5055):
    # 创建一个新的 TCP/IP 套接字
    sock = socket.socket()  
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # 将套接字绑定到一个地址和端口上
    sock.bind((host, port))
    # 将套接字标记为监听状态
    sock.listen()

    while True:
        # 建立新的连接
        client_sock, addr = sock.accept()
        print('connection from -> ', addr)
        handle_client(client_sock)

def handle_client(sock):
    while True:
        # 从客户端接收数据
        received_data = sock.recv(4096)
        if not received_data:
            break
        # 将数据发送回客户端
        sock.sendall(received_data)
    
    print('client disconnect -> ', sock.getpeername())
    sock.close()


# 实现并发服务器最简单的方式是使用系统线程
def thread_run_server(host='127.0.0.1', port=5055):
    # 创建一个新的 TCP/IP 套接字
    sock = socket.socket()  
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # 将套接字绑定到一个地址和端口上
    sock.bind((host, port))
    # 将套接字标记为监听状态
    sock.listen()

    while True:
        # 建立新的连接
        client_sock, addr = sock.accept()
        print('connection from -> ', addr)
        # handle_client(client_sock)
        thread = threading.Thread(target=handle_client, args=[client_sock])
        thread.start()


if __name__ == '__main__':
    #   不支持并发，多个客户端同时连接时，其中一个连接成功并占用服务器，其它客户端必须等待该客户端断开连接后才能连接
    run_server()
    #  使用系统线程进行并发
    # thread_run_server()


