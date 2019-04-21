import stitcherclass
import sys
import imutils as imu
import cv2

img1 = sys.argv[1]
img2 = sys.argv[2]

imageA = cv2.imread(img1)
imageB = cv2.imread(img2)
imageA = imu.resize(imageA, width=500)
imageB = imu.resize(imageB, width=500)

stitcher = stitcherclass.ImageStitch();
(kpsA, featuresA, imgA) = stitcher.detectKps(imageA);
(kpsB, featuresB, imgB) = stitcher.detectKps(imageB); 

(result, vis) = stitcher.stitching([imageA, imageB], showMatches=True);

cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("KeyPointsA", imgA)
cv2.imshow("KeyPointsB", imgB)
cv2.imshow("Result", result)
cv2.imshow("Matches", vis)

cv2.waitKey(0)