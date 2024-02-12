import cv2
import numpy as np
 
# define a video capture object 
vid = cv2.VideoCapture(0)
  
while(True): 
      
    # Capture the video frame by frame 
    ret, frame = vid.read()

    #convert to grayscale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #blur image
    blur_gray = cv2.GaussianBlur(gray,(5, 5),0)

    #apply canny edge
    edges = cv2.Canny(blur_gray, 50, 150)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 70  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 100  # maximum gap in pixels between connectable line segments
    line_image = np.copy(frame) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments

    # code for mask taken from stack overflow
    # create a mask
    mask = np.zeros(frame.shape[:2], np.uint8)
    mask[100:600, 300:1000] = 255

    # compute the bitwise AND using the mask
    masked_img = cv2.bitwise_and(edges,edges,mask = mask)

    lines = cv2.HoughLinesP(masked_img, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    
    x1sum = y1sum = x2sum = y2sum = total = 0

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),15)


        x1sum += x1
        y1sum += y1
        x2sum += x2
        y2sum += y2

        total += 1
        
    # Drawing the center line with the slope
    cv2.line(line_image,(int(x1sum/total),int(y1sum/total)),(int(x2sum/total),int(y2sum/total)),(0,0,255),10)

    # Draw the lines on the  image
    lines_edges = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    # from geeks for geeks
    cv2.rectangle(lines_edges, (300,100), (1000,600), (0,0,0), 5) 



    
    cv2.imshow("Detected Circle", lines_edges) 

    if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows()
