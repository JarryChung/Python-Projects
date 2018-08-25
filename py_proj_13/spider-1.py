import requests
import os
import json
import re
import datetime

# 主函数


def main():
    foreach_art_list()


def foreach_art_list():
    # 判断目录下是否存在jilv.txt文件 如果存在则读取里面的数值
    if os.path.exists('./jilv.txt'):
        f = open('./jilv.txt')
        n = f.read()
        n = int(n)
        f.close()
    else:
        n = 1
    while True:
        url = 'http://www.toutiao.com/search_content/?offset=' + \
            str(n) + '&format=json&keyword=%E6%B8%85%E7%BA%AF%E7%BE%8E%E5%A5%B3&autoload=true&count=1&cur_tab=3&from=gallery'
        re = requests.get(url)
        data = re.json()['data']
        if not data:
            break
        # 运行图片下载函数
        download_pic(data[0]['article_url'], n)
        n = n+1
        # 将n写入文件 防止程序运行出错 可以继续运行
        with open('./jilv.txt', 'w') as f:
            f.write(str(n))


def download_pic(url, n):
    download_pic_url = 'http://p3.pstatp.com/'
    # 这里必须带上协议头,否则会请求失败
    header = {
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.162 Safari/537.36'
    }
    res = requests.get(url, headers=header)
    content = res.text
    img_list_json = txt_wrap_by('gallery: JSON.parse("', '"),', content)
    # 正则获取所有的uri
    img_list = re.findall(r'uri\\":\\"(.*?)\\"', img_list_json)
    #判断是否有此目录
    if 'img' not in os.listdir('.'):
        os.mkdir('./img')
    if str(n) not in os.listdir('./img'):
        os.mkdir('./img/'+str(n))
    for v in img_list:
        img_path = download_pic_url + v
        img_path = img_path.replace("\\", "")
        # 读取图片
        atlas = requests.get(img_path).content
        # 保存图片
        with open('./img/' + str(n) + '/' + str(datetime.datetime.now()) + '.jpg', 'wb') as f:  # 把图片写入文件内
            f.write(atlas)

# 取出两个文本之间的内容


def txt_wrap_by(start_str, end, html):
    start = html.find(start_str)
    if start >= 0:
        start += len(start_str)
        end = html.find(end, start)
        if end >= 0:
            return html[start:end].strip()


# 运行程序
main()
