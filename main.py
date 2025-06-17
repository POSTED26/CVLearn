import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import importlib
import kernel

import os

def blue_screen(im1, im2):
    lower_limit = np.array([200,0,0])
    upper_limit = np.array([255,200,200])
    #mask = cv2.inRange(im1, lower_limit, upper_limit)
    mask = cv2.inRange(im1, lower_limit, upper_limit)
    clip_mask = np.copy(im1)
    clip_mask[mask != 0] = 0
    x = im1.shape[0]
    y = im1.shape[1]
    sky_crop = im2[0:x, 0: y]
    sky_crop[mask == 0] = 0

    return clip_mask + sky_crop



def main():
    #cv2.imshow("resources/megaman.png")
    #image = cv2.imread('resources/megaman.png')
    importlib.reload(kernel)
    #kernel.say()
    # weird laptop/computer issue solution
    isLaptop = False
    if(isLaptop):
        pizza = cv2.imread('pizza_bluescreen.jpg')
        sky = cv2.imread('sky.jpg')
        mm = cv2.imread('megaman.png',0)
    else:
        pizza = cv2.imread('resources/pizza_bluescreen.jpg')
        sky = cv2.imread('resources/sky.jpg')
        mm = cv2.imread('resources/megaman.png',0)
    
    img = blue_screen(pizza, sky)

    # fft 
    
    f = np.fft.fft2(mm)
    fshift = np.fft.fftshift(f)
    mag_spec = 20*np.log(np.abs(fshift))

    dft = cv2.dft(np.float32(mm), flags= cv2.DFT_COMPLEX_OUTPUT)
    dftshift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dftshift[:,:,0],dftshift[:,:,1]))

    mm_sobel = np.copy(mm)
    mm_sobely = np.copy(mm)
    sobel_x = np.array([[-1,0,1],
                        [-2,0,2],
                        [-1,0,1]])
    
    low_pass = np.array([[1/9,1/9,1/9],
                         [1/9,1/9,1/9],
                         [1/9,1/9,1/9]])
    
    #mm_sobely = kernel.sobel_y_5(mm_sobely, True)

    #mm_sobely = cv2.filter2D(mm_sobely, -1, low_pass)
    mm_lp = np.copy(mm_sobely)
    mm_sobel = cv2.filter2D(mm_sobel, -1, sobel_x)
    #mm_sobely = cv2.filter2D(mm_sobely, -1, sobel_y)
    mm_canny = np.copy(mm)
    mm_canny = cv2.Canny(mm_canny,100, 150)
    ret, binary_img = cv2.threshold(mm_sobel,50, 255,cv2.THRESH_BINARY)
    ret, binary_imgy = cv2.threshold(mm_sobely,100,255, cv2.THRESH_BINARY)
    '''
    plt.subplot(221), plt.imshow(mm, cmap='grey')
    plt.title('Input img')
    plt.subplot(222), plt.imshow(mag_spec, cmap='grey')
    plt.title('Magnitude Spectrum')
    plt.subplot(223), plt.imshow(mm, cmap='grey')
    plt.title('Input img')
    plt.subplot(224), plt.imshow(magnitude_spectrum, cmap='grey')
    plt.title('Magnitude Spectrum')
    '''

    plt.subplot(231) , plt.imshow(mm, cmap='gray')
    plt.title('Base Grey')
    plt.subplot(232), plt.imshow(mm_sobel, cmap='grey')
    plt.title('Sobel X')
    plt.subplot(233), plt.imshow(binary_img, cmap='grey')
    plt.title('Thresh')
    plt.subplot(234) , plt.imshow(mm_lp, cmap='gray')
    plt.title('Base Grey')
    plt.subplot(235), plt.imshow(mm_sobely, cmap='grey')
    plt.title('Sobel Y')
    plt.subplot(236), plt.imshow(binary_imgy, cmap='grey')
    plt.title('Thresh Y')

    window = 'Test'
    #cv2.imshow(window, mag_spec)
    cv2.imshow(window, mm_canny)
    plt.show()

    cv2.waitKey(0)

    cv2.destroyAllWindows


if __name__ == "__main__":
    main() 