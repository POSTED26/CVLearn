import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from collections import OrderedDict

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

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])
    testset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    #images, labels = next(dataiter)

    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    model = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                          ('relu2', nn.ReLU()),
                          ('output', nn.Linear(hidden_sizes[1], output_size)),
                          ('softmax', nn.Softmax(dim=1))]))
    #print(model)

    epochs = 3
    print_every = 40
    steps = 0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    for e in range(epochs):
        running_loss = 0
        for images, labels in iter(trainloader):
            steps += 1
            images.resize_(images.shape[0], 784)

            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if(steps % print_every == 0):
                print("Epoch: {}/{}...".format(e+1, epochs), "Loss: {:.4f}".format(running_loss/print_every))
                running_loss = 0
    

    images, labels = next(iter(trainloader))
    img = images[1].view(1, 784)
    with torch.no_grad():
        logits = model.forward(img)
    
    #logits = model.forward(img)
    ps = F.softmax(logits, dim=1)
    print(ps)
    plt.subplot(111), plt.imshow(images[1].numpy().squeeze()) #plt.barh(np.arange(10), ps.cpu().numpy())
    '''
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9),ncols=2)
    ax1.imshow(images[1].numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_xlim(0, 1.1)
'''
    #plt.subplot(231), plt.imshow(images[1].numpy().squeeze(), cmap='grey')

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
    
    mm_sobely = kernel.sobel_y_5(mm_sobely, True)

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
    '''
    #plt.subplot(231) , plt.imshow(mm, cmap='gray')
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
    '''
    window = 'Test'
    #cv2.imshow(window, mag_spec)
    #cv2.imshow(window, cv2.resize(img.numpy(), (400,400)))
    
    plt.show()

    cv2.waitKey(0)

    cv2.destroyAllWindows


if __name__ == "__main__":
    main() 