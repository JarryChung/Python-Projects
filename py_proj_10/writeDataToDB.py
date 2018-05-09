# -*- coding:utf-8 -*-

import pymysql as ms

con = ms.connect(host='120.79.212.69', user='root', passwd='123456', db='testdb', charset='utf8')
cur = con.cursor()
if con:
    print("Connected")

f = open("/var/lib/mysql/linuxv.txt","r")
print(f.read())

