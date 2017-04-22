from __future__ import division
import os
import pickle
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pandas as pd

M_boundary = {
    "y" : [[5,120,60], [45,255,255]],
    "w" : [[0,0,130], [255,25,255]]
}

def color_filter(img, boundary):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    idx = (hsv[:,:,0] >= boundary[0][0]) & (hsv[:,:,0] <= boundary[1][0]) & \
          (hsv[:,:,1] >= boundary[0][1]) & (hsv[:,:,1] <= boundary[1][1]) & \
          (hsv[:,:,2] >= boundary[0][2]) & (hsv[:,:,2] <= boundary[1][2])
    return idx


def loadRGBFile(infile):
    img = cv2.imread(infile)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

class AdvancedLaneLines(object):
    def __init__(self, ):
        super(AdvancedLaneLines, self).__init__()
        self.radius = None
        self.center = None
        self.mtx = None
        self.dist = None
        self.M = None
        self.Minv = None

        self.left_fit_Smooth = []
        self.right_fit_Smooth = []
        self.leftx_base = 320
        self.rightx_base = 960
        self.ym_per_pix = 30 / 720
        self.xm_per_pix = 3.7 / (self.rightx_base-self.leftx_base)
        self.dst = np.array([   [960, 0],
                                [320, 0],
                                [960, 720],
                                [320, 720] ], dtype=np.float32)

    def cameraCal(self, l_calcImages, n_row=6, n_col=9):
        l_images = l_calcImages
        img = cv2.imread(l_images[0])
        img_size = (img.shape[1], img.shape[0])

        objp = np.zeros((n_row*n_col, 3), np.float32)
        objp[:,:2] = np.mgrid[0:n_col, 0:n_row].T.reshape(-1,2)
        objpoints = []
        imgpoints = []
        for idx, fname in enumerate(l_images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (n_col,n_row), None)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
        self.mtx  = mtx
        self.dist = dist


    def undistortImage(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)


    def set_srcPoint(self, src):
        self.M    = cv2.getPerspectiveTransform(src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, src)

    def perspectTrans(self, img):
        img_size = (img.shape[1], img.shape[0])
        warped  = cv2.warpPerspective(img, self.M, img_size)
        return warped

    def binarizeImage(self, img, mag_thresh=[10, 100]):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
        outMatrix_white = np.zeros_like(img[:,:,0])
        outMatrix_yellow = np.zeros_like(img[:,:,0])
        outMatrix_yellow[ color_filter(img, M_boundary['y']) ] = 1
        cutoff = max(np.percentile(hsv[:,:,2].ravel(), 98),180)
        outMatrix_white[hsv[:,:,2]>cutoff] = 1
        if sum( (np.sum(outMatrix_white, axis=1) > 0)[-200:] ) < 50:
            hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
            sobelx_hls = cv2.Sobel(hls[:,:,1], cv2.CV_64F, 1, 0)
            sobel_abs = np.int8(255*sobelx_hls/np.max(sobelx_hls))
            outMatrix_white = np.zeros_like(img[:,:,0])
            outMatrix_white[ (sobel_abs >mag_thresh[0]) & (sobel_abs <mag_thresh[1]) ] = 1


        outMatrix_white_gau = cv2.GaussianBlur((outMatrix_white)*255, (5, 5), 0)
        outMatrix_white_gau_out = np.zeros_like(outMatrix_white_gau)
        outMatrix_white_gau_out[outMatrix_white_gau > 0] = 1
        outMatrix_yellow = np.int8(outMatrix_yellow)
        outMatrix_white_gau_out = np.int8(outMatrix_white_gau_out)
        mask_merge = cv2.bitwise_or(outMatrix_yellow, outMatrix_white_gau_out).astype(np.float)
        mask_merge[mask_merge>0] = 1
        mask_merge[600:,0:160] = 0
        mask_merge[600:,1000:] = 0
        return mask_merge

    def __searchLeftSide(self, binary_warped, margin, xpos_current, nwindows, minpix, max_margin):
        isPass = 0
        margin = int(margin *1.5)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        window_height = np.int(binary_warped.shape[0]/nwindows)
        pos_lane_inds = []
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xpos_low = xpos_current - margin
            win_xpos_high = xpos_current + margin

            good_pos_inds  = ((nonzeroy >= win_y_low)    & (nonzeroy < win_y_high) &\
                              (nonzerox >= win_xpos_low) & (nonzerox < win_xpos_high)).nonzero()[0]
            pos_lane_inds.append(good_pos_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_pos_inds) > minpix:
                xpos_current = np.int(np.mean(nonzerox[good_pos_inds]))


        pos_lane_inds = np.concatenate(pos_lane_inds)
        if (len(pos_lane_inds) > minpix):
            isPass = 1
        if margin > max_margin:
            isPass = 1
        return isPass,margin



    def polyHist(self, binary_warped, max_margin=125, plot=False):
        margin = 40    # Set the width of the windows +/- margin
        minpix = 400    # Set minimum number of pixels found to recenter window
        nwindows = 9               # Choose the number of sliding windows

        binary_warped[binary_warped!=0] = 1
        window_height = np.int(binary_warped.shape[0]/nwindows)     # Set height of windows

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        out_img = np.array(out_img, dtype=np.uint8)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = self.leftx_base
        rightx_current = self.rightx_base

        # Step through the windows one by one

        margin_left = margin
        margin_right = margin

        leftPass = 0
        rightPass = 0
        while 1:
            if not leftPass:
                leftPass,margin_left = self.__searchLeftSide(binary_warped,
                                    margin_left, leftx_current, nwindows, minpix, max_margin)
            if not rightPass:
                rightPass,margin_right = self.__searchLeftSide(binary_warped,
                                    margin_right,rightx_current, nwindows, minpix, max_margin)

            if leftPass and rightPass:
                left_lane_inds = []
                right_lane_inds = []
                leftx_current = self.leftx_base
                rightx_current = self.rightx_base
                for window in range(nwindows):
                    # Identify window boundaries in x and y (and right and left)
                    win_y_low = binary_warped.shape[0] - (window+1)*window_height
                    win_y_high = binary_warped.shape[0] - window*window_height
                    win_xleft_low = leftx_current - min(max_margin,int(margin_left*((9-window) / 6+1)))
                    win_xleft_high = leftx_current + min(max_margin,int(margin_left*((9-window) / 6+1)))
                    win_xright_low = rightx_current - min(max_margin,int(margin_right*((9-window) / 6+1)))
                    win_xright_high = rightx_current + min(max_margin,int(margin_right*((9-window) / 6+1)))

                    # Draw the windows on the visualization image
                    if plot:
                        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
                        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)

                    good_left_inds  = ((nonzeroy >= win_y_low)      & (nonzeroy < win_y_high) &\
                                    (nonzerox >= win_xleft_low)  & (nonzerox < win_xleft_high)).nonzero()[0]
                    good_right_inds = ((nonzeroy >= win_y_low)      & (nonzeroy < win_y_high) &\
                                    (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                    left_lane_inds.append(good_left_inds)
                    right_lane_inds.append(good_right_inds)
                    # If you found > minpix pixels, recenter next window on their mean position
                    if len(good_left_inds) > minpix:
                        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                    if len(good_right_inds) > minpix:
                        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

                left_lane_inds = np.concatenate(left_lane_inds)
                right_lane_inds = np.concatenate(right_lane_inds)
                break

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit  = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        left_fit1  = np.polyfit(lefty *self.ym_per_pix, leftx *self.xm_per_pix, 2)
        right_fit1 = np.polyfit(righty*self.ym_per_pix, rightx*self.xm_per_pix, 2)

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx  = left_fit[0]*ploty**2  + left_fit[1]*ploty  + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        img_size = (binary_warped.shape[1], binary_warped.shape[0])
        if plot:
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        return out_img, left_fitx, right_fitx, left_fit1, right_fit1, left_fit, right_fit, ploty

    def parse_curve(self, fitx, y_total):
        r = (1+(2*fitx[0]*y_total*self.ym_per_pix+fitx[1])**2)**1.5 / abs(2*fitx[0])
        return r

    def calcRadiusAndPosToCenter(self, left_fit1, right_fit1, left_fitx, right_fitx, y_total):
        r1 = self.parse_curve(left_fit1,  y_total)
        r2 = self.parse_curve(right_fit1, y_total)
        pos_lineCent = (y_total/2-int((left_fitx[-1]+right_fitx[-1])/2)) * self.xm_per_pix
        return (r1+r2)/2, pos_lineCent

    def generateOut(self, img):
        udst = self.undistortImage(img)
        pxpt = self.perspectTrans(udst)
        img_bin = self.binarizeImage(pxpt)
        binary_warped = img_bin.copy()
        try:
            hist_warped,left_fitx, right_fitx, left_fit1, right_fit1, left_fit, right_fit, ploty = self.polyHist(binary_warped)
            color_warp = np.zeros_like(pxpt).astype(np.uint8)

            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
            newwarp = cv2.warpPerspective(color_warp, self.Minv, (binary_warped.shape[1], binary_warped.shape[0]))
            result = cv2.addWeighted(udst, 1, newwarp, 0.3, 0)
            r,pos_lineCent = self.calcRadiusAndPosToCenter(left_fit1, right_fit1,
                                                           left_fitx, right_fitx, img.shape[1])

            cv2.putText(result,'R = %1.2f m'               % (r),(60,80),  cv2.FONT_HERSHEY_PLAIN, 4,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(result,'PositionToCenter = %1.2fm' % (pos_lineCent),(60,160),  cv2.FONT_HERSHEY_PLAIN, 4,(255,255,255),2,cv2.LINE_AA)
        except:
            result = udst
            cv2.putText(result,"No line detected",(10,50),  cv2.FONT_HERSHEY_PLAIN, 4,(255,255,255),2,cv2.LINE_AA)
        return result


    def generateOutSmoothed(self, img):
        udst = self.undistortImage(img)
        pxpt = self.perspectTrans(udst)
        img_bin = self.binarizeImage(pxpt)
        binary_warped = img_bin.copy()
        try:
            hist_warped,left_fitx, right_fitx, left_fit1, right_fit1, left_fit, right_fit, ploty = self.polyHist(binary_warped)
            self.left_fit_Smooth.append(left_fit)
            self.right_fit_Smooth.append(right_fit)
            if len(self.left_fit_Smooth) > 100:
                self.left_fit_Smooth = self.left_fit_Smooth[-20:]
                self.right_fit_Smooth = self.right_fit_Smooth[-20:]

            left_fit_smooth = np.median(np.array(self.left_fit_Smooth)[-10:,:],0)
            right_fit_smooth = np.median(np.array(self.right_fit_Smooth)[-10:,:],0)
            left_fitx  = left_fit_smooth[0]*ploty**2  + left_fit_smooth[1]*ploty  + left_fit_smooth[2]
            right_fitx = right_fit_smooth[0]*ploty**2 + right_fit_smooth[1]*ploty + right_fit_smooth[2]

            color_warp = np.zeros_like(pxpt).astype(np.uint8)
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
            newwarp = cv2.warpPerspective(color_warp, self.Minv, (binary_warped.shape[1], binary_warped.shape[0]))
            result = cv2.addWeighted(udst, 1, newwarp, 0.3, 0)
            r,pos_lineCent = self.calcRadiusAndPosToCenter(left_fit1, right_fit1,
                                                           left_fitx, right_fitx, img.shape[1])

            cv2.putText(result,'R = %1.2f m'               % (r),(60,80),  cv2.FONT_HERSHEY_PLAIN, 4,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(result,'PositionToCenter = %1.2fm' % (pos_lineCent),(60,160),  cv2.FONT_HERSHEY_PLAIN, 4,(255,255,255),2,cv2.LINE_AA)
        except:
            result = udst
            cv2.putText(result,"No line detected",(10,50),  cv2.FONT_HERSHEY_PLAIN, 4,(255,255,255),2,cv2.LINE_AA)
        return result
