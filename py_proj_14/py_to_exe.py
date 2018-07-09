# -*- coding: utf-8 -*-

"""
__title__ = '转换为exe文件'
"""
from PyInstaller.__main__ import run

if __name__ == '__main__':

    opts = ['i_love_you.py', '-w', '--onefile']
    # opts = ['douyin.py', '-F']
    # opts = ['douyin.py', '-F', '-w']
    # opts = ['douyin.py', '-F', '-w', '--icon=TargetOpinionMain.ico','--upx-dir','upx391w']
    run(opts)
