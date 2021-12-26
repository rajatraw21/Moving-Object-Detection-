import cv2  # image
import time  # delay
import imutils  # resize

cam = cv2.VideoCapture(0)  # cam id
time.sleep(1)

firstFrame = None
area = 500

# pre-processing
while True:
    _, img = cam.read()  # read frame from camera
    text = "Normal"
    img = imutils.resize(img, width=500)  # resize
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # color 2 gray scale image
    guassianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)  # smoothing
    if firstFrame is None:
        firstFrame = guassianImg  # capturing 1st frame on 1st iteration
        continue
    # absolute diff b/w 1st and current frame
    imgDiff = cv2.absdiff(firstFrame, guassianImg)
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]  # converting to binary image(black&white)

    # dilating the image ( filling the gapes of empty pixels )
    threshImg = cv2.dilate(threshImg, None, iterations=2)
    cnts = cv2.findContours(
        threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "moving object detection"
    print(text)
    cv2.putText(img, text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("cameraFeed", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()