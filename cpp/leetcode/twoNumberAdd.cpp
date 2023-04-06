/**
 * @file twoNumberAdd.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-04-04
 * 
 * @copyright Copyright (c) 2023
 * 给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
 * 
 * 
 */

#include <iostream>
#include <vector>
#include <tr1/unordered_map>
#include <tr1/unordered_set>

using namespace std::tr1;

using namespace std;


class twoNumberAdd
{
private:
    /* data */
public:
    twoNumberAdd(/* args */);
    ~twoNumberAdd();
    vector<int> twoSum(vector<int> &nums, int target); 
};

twoNumberAdd::twoNumberAdd(/* args */)
{
}

twoNumberAdd::~twoNumberAdd()
{
}


vector<int> twoNumberAdd::twoSum(vector<int> &nums, int target)
{
    unordered_map<int, int> map;

    for (int i = 0; i < nums.size(); i++)
    {
        auto iter = map.find(target - nums[i]);
        if (iter != map.end())
        {
            return {iter->second, i};
        }

        map.insert(pair<int, int>(nums[i], i));
                
    }

    return {};
}


