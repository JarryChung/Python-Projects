from __future__ import unicode_literals
from threading import Timer
from wxpy import *
import requests
from wechat_sender import Sender
import time

bot = Bot(console_qr=2, cache_path= "botoo.pkl")        # 以像素的形式显示二维码
# bot = Bot()       # 在win环境上运行时将上一行替换为此行


# 获取金山词霸每日一句
def get_news1():
    url = "http://open.iciba.com/dsapi/"
    r = requests.get(url)
    contents = r.json()['content']
    translation = r.json()['translation']
    return contents,translation


def send_news():
    try:
        my_friend = bot.friends().search(u'Jarry & Chung')[0]       # 对方的微信名称，而非备注或微信账号
        my_friend.send(get_news1()[0])
        my_friend.send(get_news1()[1][5:])
        my_friend.send(u'来自酷酷的展济的鸡汤')
        t = Timer(86400, send_news)         # 每天发送一次
        t.start()
    except:
        my_friend = bot.friends().search('匿名')[0]         # 己方的微信名称
        my_friend.send(u'今天消息发送失败了')


if __name__ == "__main__":
    send_news()