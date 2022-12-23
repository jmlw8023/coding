

import requests
import json





import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
print(s.getsockname()[0])
res = s.getsockname()[0]



# # 本地局域网IP
# import socket
# # 函数 gethostname() 返回当前正在执行 Python 的系统主机名
# res = socket.gethostbyname(socket.gethostname())
# print(res)




def ipQuery(ip):
    # 淘宝api接口
    url = "http://ip.taobao.com/outGetIpInfo?ip={}&accessKey=alibaba-inc".format(ip)
    req = requests.get(url).text
    json1 = json.loads(req)
    print(json1)
    country = json1["data"]["country"]  # 国
    province = json1["data"]["region"]  # 省
    city = json1["data"]["city"]  # 市
    return "{}-{}-{}".format(country, province, city)
    
    # ip-api接口
    # url = "http://ip-api.com/json/111.121.64.21?lang=zh-CN"
    # country = json1["country"]  # 国
    # province = json1["regionName"]  # 省
    # city = json1["city"]  # 市
    # print("{}-{}-{}".format(country, province, city))
    
    # 太平洋api接口
    # url = "http://whois.pconline.com.cn/ipJson.jsp?ip=111.121.64.21&json=true"
    # province = json1["pro"]  # 省
    # city = json1["city"]  # 市
    # print("{}-{}".format(province, city))
 
# ipQuery("8.8.8.8")
ipQuery(res)


# import pycountry
# countries = {}
# for country in all_countries:
#     subdivisions = list(pycountry.subdivisions.get(country_code = country.alpha_2))
#     provinces = [sub for sub in subdivisions if sub.type == "Province"]
#     all_provinces = {province.name:{"code":province.code,"cities":{}} for province in provinces}

#     countries.update({country.name:{"alpha_2":country.alpha_2,
#                       "alpha_3":country.alpha_3,
#                       "numeric":country.numeric,
#                       #"official_name":country.official_name
#                       "provinces":all_provinces
#                     }})



