#coding=utf-8
from asyncio.tasks import sleep
import os
import random
import time

import cv2

import numpy as np


def getPic(id):
    os.system('adb shell screencap -p /sdcard/%s.png' % str(id))
    os.system('adb pull /sdcard/%s.png pic/test.png' % str(id))

#获取目标的位置（左上角坐标）
#img_rgb:传入的彩色图片
#target:目标图片
#value:阈值
def getPlayerLoc(img_rgb,target,value = 0.7):
    img_gray = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray,target,cv2.TM_CCOEFF_NORMED)
    t = value
    loc = np.where(res > t)
    point = []
    for pt in zip(*loc[::-1]):
        point = pt
    return point
#计算棋子的中心位置、
#target:目标的中心位置
#point:在图片中的位置
def getPlayerCenterPoint(target,point):
    w,h = target.shape[::-1]
    x_c = point[0] + w // 2
    y_c = point[1] + h
    print("棋子的中心位置：{}，{}".format(x_c,y_c))
    return x_c,y_c

def getNextPoint(img_rgb):
    #获取到图片的高度和宽度
    H,W,P= img_rgb.shape
    a_img_rgb = np.array(img_rgb)
    # print(H,W)
    #背景颜色
    background = img_rgb[0][0]
    a_background = np.array(background)
    print(background)
    h = H // 6
    
    # print(a_img_rgb)
    # 获取到第一个色差发生大变化的位置
    for i in range(h,H,20):
        for j in range(0,W,20):
            a = (int)(a_img_rgb[i][j][0])
            b = (int)(a_background[0])
            if( np.abs(a - b) > 20):
                print('下一跳的位置：{},{}'.format(i,j))
                return j,i

def jump(distance,value = 1.9):
    press_time = distance * value
    press_time = max(press_time,200)
    press_time = int(press_time)
    print(press_time)
    rand = random.randint(0,9) * 10
    cmd = 'adb shell input swipe {} {} {} {} {duration}'.format(
        374+rand,
        1060+rand,
        374+rand,
        1060+rand,
        duration=press_time
    )
    print(cmd)
    os.system(cmd)

while True:
    getPic(0)
    value = 1.85
    img_rgb = cv2.imread('pic/test.png')
    target = cv2.imread('pic/player.png',0)
    p = getPlayerLoc(img_rgb,target)
    p1 = getPlayerCenterPoint(target, p)
    p2 = getNextPoint(img_rgb)
    d = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 ) ** 0.5
    print('距离：{},系数：{}'.format(d,value))
    jump(d,value)
    time.sleep(random.uniform(0.7,1.2))
    