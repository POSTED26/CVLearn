import cv2
import numpy as np




def sobel_y_5(img, thresh):
    sobel_y = np.array([[-2,-2,-4,-2,-2],
                        [-1,-1,-2,-1,-1],
                        [0,0,0,0,0],
                        [1,1,2,1,1],
                        [2,2,4,2,2]])
    sobel = np.copy(img)
    sobel = cv2.filter2D(sobel, -1, sobel_y)

    if(thresh):
        ret, binary_imgy = cv2.threshold(sobel,100,255, cv2.THRESH_BINARY)


    return binary_imgy


def say():
    print('this sucks')