
# -*- encoding: utf-8 -*-
'''
@File    :   single_ListNode.py
@Time    :   2023/01/04 16:54:30
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
@Reference_Linking  : # https://zhuanlan.zhihu.com/p/60057180
'''

# import packets
# import os





class Node(object):
    """single linked list"""
    def __init__(self, data) -> None:
        self.data = data
        self.next= None

    def has_value(self, value):
        if self.data == value:
            return True
        else:
            return False


class SingleLinkList(object):

    def __init__(self):
        self._head = None
    
    def is_empty(self):
        return self._head is None
    
    def length(self):
        cur = self._head
        count = 0
        while cur is not None:
            count += 1
            cur = cur.next
        return count

    def items(self):
        cur = self._head
        while cur is not None:
            # 返回生成器
            yield cur.data
            # 指针下移
            cur = cur.next
    
    def add(self, item):
        # 向链表头增加元素
        node = Node(item)
        node.next = self._head

    def append(self, item):
        # 尾部增加元素
        node = Node(item)
        if self.is_empty():
            self._head = node
        else:
            # 链表非空
            cur = self._head
            while cur.next is not None:
                cur = cur.next
            cur.next = node
    
    def insert(self, index, item):
        # 指定位置插入元素
        if index <= 0:
            self.add(item)
        elif index > (self.length() - 1):
            self.append(item)
        else:
            node = Node(item)
            cur = self._head

            for i in range(index - 1):
                cur = cur.next
            node.next = cur.next
            cur.next = node
        
    def remove(self, item):
        cur = self._head
        pre = None
        if cur is not None:
            if cur.data == item:

                if not pre:
                    self._head = cur.next
                else:
                    pre.next = cur.next
                return True
            else:
                pre = cur
                cur = cur.next
    def find(self, item):
        return item in self.items()


# 链表
link_list = SingleLinkList()
for i in range(1, 15, 2):
    # print(i)
    link_list.append(i)

link_list.insert(3, 111)

for i in link_list.items():
    print(i, end='\t')
print()

# print(link_list.is_empty())
# print(link_list.find(5))

# for i in range(link_list.length()):
#     print(link_list._head.data)



#  # 节点
# node1 = Node(1)
# node2 = Node(3)
# node3 = Node(5)

# # 第一个节点加到链表
# link_list._head = node1
# node1.next = node2
# node2.next = node3


# while link_list._head:
#     print(link_list._head.item)
#     link_list._head

# print(link_list._head)
# print(link_list._head.next)
# print(link_list._head.next.next)
# print(link_list._head.next.next.next)
# print(link_list._head.next.next + 1)

# print(next(link_list._head.next.next))

