from foveate_image import *
from matplotlib import pyplot as plt
from PIL import Image
# import numpy

def main():
    # img = open('DSC_1625.jpg')
    img = Image.open('that_dog_in_me.jpg')

    w, h = img.size
    print("Original Image Size", w, h)
    print("Pixel Count", w*h)

    img = np.array(list(img.getdata()))
    img = img.reshape((h,w,3))
    
    # print(img)

    # plt.imshow(img)
    # plt.show()

    fimg = FoveateImage()
    foveatedimg = fimg.foveate(img)
    #Number of non-zero pixels in the foveated image
    # import pdb; pdb.set_trace()
    print("Foveated Pixel Count", np.count_nonzero(foveatedimg[:,:,0]))

    plt.imshow(foveatedimg)
    plt.show()


if __name__ == "__main__":
    main()
