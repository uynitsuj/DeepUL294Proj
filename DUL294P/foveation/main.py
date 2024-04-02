from foveate_image import *
from matplotlib import pyplot as plt
from PIL import Image
import time
# import numpy

def main():
    
    # img = open('DSC_1625.jpg')
    img = Image.open('dog_in_me.jpeg')

    w, h = img.size
    print("Original Image Size", w, h)
    print("Pixel Count", w*h)

    img = np.array(list(img.getdata()))
    img = img.reshape((h,w,3))
    
    # print(img)

    # plt.imshow(img)
    # plt.show()

    fimg = FoveateImage(w, h)
    start = time.time()
    foveatedimg, idxs = fimg.foveate(img)
    elapsed = (time.time() - start)
    print(f"Elapsed time: {elapsed}(s)")
    print(f"Frequency: {1/elapsed}(fps)")
    print("Foveated Pixel Count", foveatedimg.shape)

    recon = torch.zeros((h,w,3), dtype=foveatedimg.dtype)
    # import pdb; pdb.set_trace()
    recon.view(-1, 3)[idxs] = foveatedimg[range(len(foveatedimg))]

    print(f"Compression: {len(idxs)/(w*h)*100}%")

    plt.imshow(recon)
    plt.show()


if __name__ == "__main__":
    main()
