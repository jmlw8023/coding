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
    


