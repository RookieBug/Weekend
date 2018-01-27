#coding=utf-8
import cv2
import numpy as np

def getNextPoint(img_rgb):
    H,W,P= img_rgb.shape
    a_img_rgb = np.array(img_rgb)
    background = img_rgb[100][100]
    a_background = np.array(background)
    print(background)
    h = H // 5
    
    # print(a_img_rgb)
    
    for i in range(h,H,20):
        for j in range(0,W,20):
            a = (int)(a_img_rgb[i][j][0])
            b = (int)(a_background[0])
            if( np.abs(a - b) > 20):
                print('��һ����λ�ã�{},{}'.format(i,j))
                return j,i

img_rgb = cv2.imread('pic/test1.png')
p = getNextPoint(img_rgb)
print(p)