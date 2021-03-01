import cv2
import numpy as np
from matplotlib import pyplot as plt
import PIL
from skimage import  io
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.filters import gaussian
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import data, color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage.io import imread
from skimage.color import rgba2rgb,rgb2gray,gray2rgb
from copy import deepcopy
from skimage.transform import resize


def gaussian_smoothing(input_img):
    gaussian_filter = np.array([[0.109, 0.111, 0.109],
                                [0.111, 0.135, 0.111],
                                [0.109, 0.111, 0.109]])

    return cv2.filter2D(input_img, -1, gaussian_filter)




def CountShapes_ID(input):
    input=resize(input,(640,480)).shape
    circles=[]
    Rows = input.shape[0]
    Cols = input.shape[1]

    # initializing the angles to be computed
    Sinangle = dict()
    Cosineangle = dict()

    # initializing the angles
    for angle in range(0, 360):
        Sinangle[angle] = np.sin(angle * np.pi / 180)
        Cosineangle[angle] = np.cos(angle * np.pi / 180)

    radius =  np.arange(20,150,5)


    threshold = 190

    for r in radius:
        # Initializing an empty 2D array with zeroes
        acc_cells = np.full((Rows, Cols), fill_value=0, dtype=np.uint64)

        # Iterating through the original image
        for x in range(Rows):
            for y in range(Cols):
                if input[x,y] == 255:  # edge
                    # increment in the accumulator cells
                    for angle in range(0, 360):
                        b = y - round(r * Sinangle[angle])
                        a = x - round(r * Cosineangle[angle])
                        if a >= 0 and a < Rows and b >= 0 and b < Cols:
                            acc_cells[int(a),int(b)] += 1

        print('For radius: ', r)
        acc_cell_max = np.amax(acc_cells)
        print('max acc value: ', acc_cell_max)

        if (acc_cell_max > 150):

            print("Detecting the circles for radius: ", r)

            # Initial threshold
            acc_cells[acc_cells < 150] = 0

            # find the circles for this radius
            for i in range(Rows):
                for j in range(Cols):
                    if (i > 0 and j > 0 and i < Rows - 1 and j < Cols - 1 and acc_cells[i][j] >= 150):
                        avg_sum = np.float32((acc_cells[i][j] + acc_cells[i - 1][j] + acc_cells[i + 1][j] +
                                              acc_cells[i][j - 1] + acc_cells[i][j + 1] + acc_cells[i - 1][j - 1] +
                                              acc_cells[i - 1][j + 1] + acc_cells[i + 1][j - 1] + acc_cells[i + 1][
                                                  j + 1]) / 9)
                        print("Intermediate avg_sum: ", avg_sum)
                        if (avg_sum >= 33):
                            print("For radius: ", r, "average: ", avg_sum, "\n")
                            circles.append((i, j, r))
                            acc_cells[i:i + 5, j:j + 7] = 0
    return circles

image=rgb2gray((io.imread("ball.JPG")))
smooth_img=gaussian_smoothing(image)
Edge=canny(smooth_img)
resize(Edge, (320, 240)).shape
hough_radii = np.arange(20,150,20)
circles=[]
circles=CountShapes_ID(Edge)
    # Detect Circle
circles=CountShapes_ID(gray2rgb(Edge))

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 10))

ax.imshow((circles))
result=circles
result.save("Circlesdetected.JPG")
