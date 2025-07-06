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
import net

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

def validation(model, testloader, criterion, device):
    test_loss = 0
    accuacy = 0
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
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


def check_accuracy_before(testloader, net, device):
    # Calculate accuracy before training
    correct = 0
    total = 0

    # Iterate through test dataset
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        # forward pass to get outputs
        # the outputs are a series of class scores
        outputs = net.forward(images)

        # get the predicted class from the maximum value in the output-list of class scores
        _, predicted = torch.max(outputs.data, 1)

        # count up total number of correct labels
        # for which the predicted and true labels are equal
        total += labels.size(0)
        correct += (predicted == labels).sum()

    # calculate the accuracy
    # to convert `correct` from a Tensor into a scalar, use .item()
    accuracy = 100.0 * correct.item() / total

    # print it out!
    print('Accuracy before training: ', accuracy)

def main():
    #cv2.imshow("resources/megaman.png")
    #image = cv2.imread('resources/megaman.png')
    
    # weird laptop/computer issue solution
    isLaptop = True
    if(isLaptop):
        pizza = cv2.imread('pizza_bluescreen.jpg')
        sky = cv2.imread('sky.jpg')
        mm = cv2.imread('megaman.png',0)
    else:
        pizza = cv2.imread('resources/pizza_bluescreen.jpg')
        sky = cv2.imread('resources/sky.jpg')
        mm = cv2.imread('resources/megaman.png',0)
    
    img = blue_screen(pizza, sky)



    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU.")

    # till I figure out cuda issues with test net
    device = "cuda"

    batch_size = 20
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])
    trainset = datasets.FashionMNIST('MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    #images, labels = next(dataiter)

    model = net.Net()
    model = model.to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(model)

    check_accuracy_before(testloader, model, device)
    
    # Training network
    epochs = 3
    loss_over_time = net.train_net(epochs, trainloader, optimizer, criterion, model, device)
    print(loss_over_time)
    # test network
    net.test_net(model,testloader,criterion,batch_size, classes, device)

    # save model

    model_dir = "saved models/"
    model_name = "FirstCNN.pt"
    torch.save(model.state_dict(), model_name)


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
 
    window = 'Test'
    #cv2.imshow(window, mag_spec)
    #cv2.imshow(window, cv2.resize(img.numpy(), (400,400)))
    
    plt.show()

    cv2.waitKey(0)

    cv2.destroyAllWindows


if __name__ == "__main__":
    main() 