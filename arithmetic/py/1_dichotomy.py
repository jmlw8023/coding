#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   1_dichotomy.py
@Time    :   2024/07/31 10:56:24
@Author  :   hgh 
@Version :   1.0
@Desc    :   二分法
'''

# import module
import os
import numpy as np


class ARITH:
    
    def __init__(self) -> None:
        pass
    
    
    def search(self, nums : list[int], target : int) -> int:
        left = 0
        right = len(nums)
        
        while (left < right):
            
            middle = left + (right - left) // 2
            
            if nums[middle] < target:
                left = middle + 1
            elif nums[middle] > target:
                right = middle
            else:
                return middle
            
        return -1
    
    
    def get_num_len(self, nums : list[int], value : int) -> int:
        
        low = 0
        fast = 0
        
        # for i, num in enumerate(nums):
        while fast < len(nums):
            if (nums[fast] != value):
                nums[low] = nums[fast]
                low += 1
            fast += 1
        
        return low
            
        
    def removeElement(self, nums: list[int], val: int) -> int:
        # 快慢指针
        fast = 0  # 快指针
        slow = 0  # 慢指针
        size = len(nums)
        while fast < size:  # 不加等于是因为，a = size 时，nums[a] 会越界
            # slow 用来收集不等于 val 的值，如果 fast 对应值不等于 val，则把它与 slow 替换
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1
            
        return slow






if __name__ == '__main__':
    
    
    demo = ARITH()
    
    np.random.seed(22)
    
    data = np.random.randint(0, 20, 10)
    data = (sorted(data))
    print(data)
    
    target = 6
    
    res = demo.get_num_len(data.copy(), target)
    print(res)
    print(data)
    
    res = demo.removeElement(data.copy(), target)
    print(res)
    
    
    
    pass



