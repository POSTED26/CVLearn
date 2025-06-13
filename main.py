import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def blue_screen(im1, im2):
    lower_limit = np.array([200,0,0])
    upper_limit = np.array([255,200,200])
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
    pizza = cv2.imread('resources/pizza_bluescreen.jpg')
    sky = cv2.imread('resources/sky.jpg')
    img = blue_screen(pizza, sky)

    # fft 
    mm = cv2.imread('resources/megaman.png',0)
    f = np.fft.fft2(mm)
    fshift = np.fft.fftshift(f)
    mag_spec = 20*np.log(np.abs(fshift))

    dft = cv2.dft(np.float32(mm), flags= cv2.DFT_COMPLEX_OUTPUT)
    dftshift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dftshift[:,:,0],dftshift[:,:,1]))



    plt.subplot(221), plt.imshow(mm, cmap='grey')
    plt.title('Input img')
    plt.subplot(222), plt.imshow(mag_spec, cmap='grey')
    plt.title('Magnitude Spectrum')
    plt.subplot(223), plt.imshow(mm, cmap='grey')
    plt.title('Input img')
    plt.subplot(224), plt.imshow(magnitude_spectrum, cmap='grey')
    plt.title('Magnitude Spectrum')

    window = 'Test'
    #cv2.imshow(window, mag_spec)
    plt.show()

    cv2.waitKey(0)

    cv2.destroyAllWindows


if __name__ == "__main__":
    main() 