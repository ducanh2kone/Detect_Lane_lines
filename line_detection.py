# Using Hough Transform to detect line
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from sympy import false
#read video frame by frame
video = cv.VideoCapture("video_demo.mp4")
while True:
    ret, frame = video.read()
    # convert to gray img
    gray_image = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    # blur img
    gray_image = cv.GaussianBlur(gray_image, (5,5),0)
    #canny 
    canny_image = cv.Canny(gray_image, 50, 210)
    #Define the region of interest
    vertical =np.array([[(0,1000), (850, 580), (1215, 580), (1500, 1077)]], dtype= np.int32)
    # tạo 1 ảnh màu đen kích thước như gray_img
    mark = np.zeros_like(gray_image)
    # cv.imshow("1",mark)
    # Tạo một mark lọc vùng cần chọn,chuyển vùng cần chọn về màu trắng 255
    mark = cv.fillPoly(mark,vertical, 255)
    #
    mark_img = cv.bitwise_and(gray_image, mark)
    
    mark_img = cv.bitwise_and(canny_image, mark)
    lines = cv.HoughLinesP(mark_img,1, np.pi/360,20,5,maxLineGap=30)
    #   create a empty black img
    line_img = np.zeros((mark_img.shape[0], mark_img.shape[1],3),dtype= np.uint8)
    # từ bức ảnh trống, và điểm tìm từ hàm ta vẽ đường thẳng lên nó
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv.line(line_img,(x1,y1), (x2,y2), [0,255,0],10)
    cv.imshow("1",line_img)
    image_with_line = cv.addWeighted(frame,1,line_img,1,0)
    # plt.imshow(image_with_line)
    # plt.show()
    if ret == False:
        break
    # # cv.imshow("blur",gray_image)
    # # cv.imshow("cannyimg",canny_image)
    cv.imshow("frame",image_with_line)
    cv.waitKey(5)
    if cv.waitKey(1) == ord('q'):
            break
video.release()
cv.destroyAllWindows()




