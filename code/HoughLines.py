import math
import sys
import os
import pickle
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pandas as pd
#sys.path.append('./')
import code.AdvancedLaneLines as m_AdvLine


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)

def get_vertice(image):
    vertices = np.array([[
        (image.shape[1]*1/19,   image.shape[0]*19/19),
        (image.shape[1]*9/19,  image.shape[0]*11/19),
        (image.shape[1]*10/19, image.shape[0]*11/19),
        (image.shape[1]*18/19,   image.shape[0]*19/19)
    ]])
    return vertices


def draw_lines2(img, lines, color=[255, 0, 0], thickness=5):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    pd_lines = pd.DataFrame(lines[:,0,:], columns=['x1', 'y1', 'x2', 'y2'])
    pd_lines['length']   = pd_lines.apply(lambda x: (x.x2-x.x1)**2+(x.y2-x.y1)**2, axis=1)**0.5
    pd_lines['slope']    = pd_lines.apply(lambda x: (x.y2-x.y1) / (x.x2-x.x1), axis=1)
    pd_lines['bias']     = pd_lines.apply(lambda x: (x.y1-x.slope*x.x1), axis=1)
    #pd_lines['bias']     = pd_lines.apply(lambda x: ((x.y2+x.y1) - (x.x2+x.x1))*x.slope/2, axis=1)
    pd_lines['interval'] = pd_lines.apply(lambda x: int(x.length/10), axis=1)
    l_slope = np.array([ pd_lines.iloc[row_i]['slope']   for row_i in range(pd_lines.shape[0])
                                                for col_i in range(int(pd_lines.iloc[row_i]['interval']))  ])

    l_bias  = np.array([ pd_lines.iloc[row_i]['bias']    for row_i in range(pd_lines.shape[0])
                                                for col_i in range(int(pd_lines.iloc[row_i]['interval']))  ])

    b1 = np.median(l_bias[l_slope >0])
    s1 = np.median(l_slope[l_slope>0])
    b2 = np.median(l_bias[l_slope <0])
    s2 = np.median(l_slope[l_slope<0])

    l_params = [[b1, s1], [b2, s2]]
    #print(pd_lines.head())

    y_top    = min( min(pd_lines['y1']), min(pd_lines['y2']) )
    y_bottom = max( max(pd_lines['y1']), max(pd_lines['y2']) )
    x_top1   = int((y_top    - l_params[0][0]) / l_params[0][1])
    x_top2   = int((y_top    - l_params[1][0]) / l_params[1][1])
    x_bottom1= int((y_bottom - l_params[0][0]) / l_params[0][1])
    x_bottom2= int((y_bottom - l_params[1][0]) / l_params[1][1])
    #print y_top, y_bottom
    #print x_top1, x_top2, x_bottom1, x_bottom2

    cv2.line(img, (x_top1, y_top), (x_bottom1, y_bottom),    color, thickness)
    cv2.line(img, (x_top2, y_top), (x_bottom2, y_bottom), color, thickness)
    return l_params

def hough_lines2(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    l_params = draw_lines2(line_img, lines, thickness=3)
    return line_img, l_params



class HoughLinePoints(m_AdvLine.AdvancedLaneLines):
    def __init__(self, kernelSize=5,min_line_len=150,max_line_gap=150,
                       rho=1,theta=np.pi/180,threshold=50,y_top=480):
        super(HoughLinePoints, self).__init__()
        self.kernelSize = kernelSize
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_len = min_line_len
        self.max_line_gap = max_line_gap
        self.y_top = y_top
        self.l_src = []
        if self.mtx is None:
            l_files = list(map(lambda x: "./camera_cal/%s" % x, os.listdir('./camera_cal/') ))
            self.cameraCal(l_files, n_row=6, n_col=9)

    def learnFromDirectRoad(self, image):
        img = m_AdvLine.loadRGBFile(image)
        img = self.undistortImage(img)
        mask_image = self.binarizeImage(img)
        canny_gradient_interest = region_of_interest(mask_image, get_vertice(img).astype(np.int))
        m1 = np.array(canny_gradient_interest / np.max(canny_gradient_interest) * 255, dtype=np.uint8)
        img_line,l_params = hough_lines2(m1, self.rho, self.theta, self.threshold, self.min_line_len, self.max_line_gap)

        img_result = weighted_img(img_line, img, beta=10)
        y_bottom = img.shape[0]
        x_top1   = int((self.y_top    - l_params[0][0]) / l_params[0][1])
        x_top2   = int((self.y_top    - l_params[1][0]) / l_params[1][1])
        x_bottom1= int((y_bottom - l_params[0][0]) / l_params[0][1])
        x_bottom2= int((y_bottom - l_params[1][0]) / l_params[1][1])
        src = np.array([[x_top1, self.y_top ], [x_top2, self.y_top ],
                        [x_bottom1, y_bottom], [x_bottom2, y_bottom]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, self.dst)
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img_result, M, img_size)
        self.l_src.append(src)
        return warped
