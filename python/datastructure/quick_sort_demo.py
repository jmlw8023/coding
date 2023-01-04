
# -*- encoding: utf-8 -*-
'''
@File    :   sort_demo.py
@Time    :   2022/10/19 14:54:26
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
'''

# import packets

# link: https://www.runoob.com/python3/python-quicksort.html

# 快排：分治法（Divide and conquer）




def partition(arr, low, high):
    i = low -1
    pivot = arr[high]

    for j in range(low, high):

        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i+1], arr[high] = arr[high], arr[i+1]

    return (i+1)

# 快排算法 
def quickSort(arr, low, high):
    if low < high:
        pt = partition(arr, low, high)

        quickSort(arr, low, pt-1)
        quickSort(arr, pt+1, high)


def quickSort2(arr):
    if(len(arr)<2): #不用进行排序
        return arr
    else:
        pivot=arr[0]
        less=[i for i in arr[1:] if(i<=pivot)]
        great=[i for i in arr[1:] if(i>pivot)]

        return quickSort2(less) + [pivot] + quickSort2(great)

def quickSort1(arr):    
    if len(arr) <= 1:        
        return arr    
    pivot = arr[len(arr) // 2]    
    left = [x for x in arr if x < pivot]    
    middle = [x for x in arr if x == pivot]    
    right = [x for x in arr if x > pivot]    
    return quickSort1(left) + middle + quickSort1(right)
# print(quicksort([3, 6, 8, 19, 1, 5]))  # [1，3, 5, 6, 8, 19]

def quickSort11(arr):
    if len(arr) < 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [i for i in arr if i < pivot]
    right = [i for i in arr if i > pivot]
    middle = [i  for i in arr if i == pivot]

    return quickSort11(left) + middle + quickSort11(right)



# 打印显示
def obj_print(arr_new):
    for index in range(len(arr_new)):
        print(arr_new[index], end=' -> ')
    print()



# 查找输入参数 arr 中符合条件的 a + b + c = object （此处是：元素集中 某元素 = 元素1 + 元素2）
def a_b_c_sum(arr: list):
    res_list = []
    if isinstance(arr, list):
        n = len(arr)
        if  n < 3:
            return -1
        else:
            if arr[0] > 0:
                return -1
            for m in range(n):
                if m > 0 and arr[m] == arr[m-1]:
                    continue
                l = m + 1
                r = n - 1
                while l < r:
                    sum = arr[m] + arr[l] + arr[r]
                    if sum == 0:
                        res_list.append((arr[m], arr[l], arr[r]))
                    while l < r and arr[l] == arr[l+1]:
                        l += 1
                    while l < r and arr[r] == arr[r-1]:
                        r -= 1
                    if sum > 0:
                        r -= 1
                    else:
                        l += 1
    return res_list



# 产生指定范围的随机整数
# arr = np.random.randint(-100, 100, 50).tolist()
# print(arr)
arr = [1, 32, -65, -24, -26, 22, 66, 84, -27, 14, -32, -16, 23, -64, -84, 81, -63, -92, -74, -48, -28, 81, -1, -18, -95, -71, -74, 27, -57, 90, 36, -19, 97, -49, 25, 79, 87, 63, 25, -62, -65, 71, 66, 57, 18, -11, -95, 88, -85, -45]
# arr = [29, 74, 94, 64, 61, 30, 17, 41, 60, 1, 67, 11, 79, 78]
# arr = [94, 64]

# t = tuple(arr)    # 转为元组类型
# print(t)

# 自定义的快排算法进行排序
# quickSort(arr, 0, len(arr)-1)
# arr_new = quickSort1(arr)
# arr_new = quickSort11(arr)

# 通过系统的 sorted 方法进行排序
arr_new = sorted(arr)
arr.sort()
obj_print(arr)
print()

# 通过随机数组，找出单个元素 = 元素1 + 元素2
res = a_b_c_sum(arr_new)
obj_print(res)
print(len(res))
