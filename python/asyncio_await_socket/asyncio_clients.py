# -*- encoding: utf-8 -*-
'''
@File    :   asyncio_clients.py
@Time    :   2023/01/04 10:06:24
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
'''


# import packets
# import os
import asyncio
import datetime




class Client():

    def __init__(self, HOST='127.0.0.1', PORT=5055, BUFSIZE=4096) -> None:
        self.host = HOST
        self.port = PORT
        self.bufsize = BUFSIZE

    def print_indent(self, indent, string):
        t = datetime.datetime.fromtimestamp(asyncio.get_event_loop().time())
        print('\t' * indent + f'[{t:%S.%f}] ' + string)


    async def client(self, name, indent):
        self.print_indent(indent, f'Client {name} tries to connect.')
        reader, writer = await asyncio.open_connection(host=self.host, port=self.port)
        # first make dummy write and read to show that the server talks to us
        writer.write(b'*')
        await writer.drain()
        resp = await reader.read(self.bufsize)
        self.print_indent(indent, f'Client {name} connects.')

        for msg in ['Hello', 'world!',]:
            await asyncio.sleep(0.5)
            writer.write(msg.encode())
            await writer.drain()
            self.print_indent(indent, f'Client {name} sends "{msg}".')
            resp = (await reader.read(self.bufsize)).decode()
            self.print_indent(indent, f'Client {name} receives "{resp}".')
        
        writer.close()
        self.print_indent(indent, f'Client {name} disconnects.')

    async def main(cls):
        # demo_client = Client()
        clients = [asyncio.create_task(cls.client(i, i)) for i in range(3)]
        await asyncio.wait(clients)



async def main():
    demo_client = Client()
    clients = [asyncio.create_task(demo_client.client(i, i)) for i in range(3)]
    await asyncio.wait(clients)


if __name__ == '__main__':
    # 类的方式
    # demo_client = Client()
    # asyncio.run(demo_client.main())

    # 普通函数
    asyncio.run(main())


