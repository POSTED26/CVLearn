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
import network

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

def validation(model, testloader, criterion):
    test_loss = 0
    accuacy = 0
    for images, labels in testloader:
        images.resize_(images.shape[0], 784)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuacy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuacy

def load_checkpoint(filepath):
    try:
        checkpoint = torch.load(filepath)
    except Exception as e:
        return None, False
    model = network.Network(checkpoint['input_size'],
                            checkpoint['output_size'],
                            checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    return model, True


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
    trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = datasets.MNIST('F_MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    #images, labels = next(dataiter)

    input_size = 784
    hidden_sizes = [516, 256]
    output_size = 10
    model, isFile = load_checkpoint('checkpoint.pth')
    if not isFile:
        model = network.Network(input_size, output_size, hidden_sizes, drop_p = 0.5)
        print('Did not find file')
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(model)

    epochs = 3
    print_every = 40
    steps = 0
    running_loss = 0
    
    
    for e in range(epochs):
        for images, labels in trainloader:
            steps += 1
            images.resize_(images.shape[0], 784)

            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if(steps % print_every == 0):
                model.eval()
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion)
                print("Epoch: {}/{}...".format(e+1, epochs), 
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Test Loss: {:.3f}".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
                running_loss = 0
                model.train()
    
    # test network

    model.eval()
    images, labels = next(iter(trainloader))
    img = images[0].view(1, 784)
    with torch.no_grad():
        logits = model.forward(img)
    
    #logits = model.forward(img)
    ps = torch.exp(logits)
    #ps = F.softmax(logits, dim=1)
    print(ps)
    plt.subplot(111), plt.imshow(images[0].numpy().squeeze()) #plt.barh(np.arange(10), ps.cpu().numpy())

    #plt.subplot(231), plt.imshow(images[1].numpy().squeeze(), cmap='grey')


    # save model
    checkpoint = {'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layers': [each.out_features for each in model.hidden_layers],
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, 'checkpoint.pth')


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