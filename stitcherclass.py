import imutils as imu
import numpy
import cv2

class ImageStitch:
	

	def detectKps(self, img12):
		
		
		st = cv2.xfeatures2d.SIFT_create()
		(keyps, feat) = st.detectAndCompute(img12, None);
		
		fgh = cv2.drawKeypoints(img12,keyps,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		
		keyps = numpy.float32([kp.pt for kp in keyps])
		return (keyps, feat, fgh)

	def matchKps(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
		
		matcher = cv2.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		
		matches = []
		for m in rawMatches:
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))
		
		if len(matches) > 4:
			ptsA = numpy.float32([kpsA[i] for (_, i) in matches])
			ptsB = numpy.float32([kpsB[i] for (i, _) in matches])
 
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				reprojThresh)
 
			return (matches, H, status)
 
		return None

	def stitching(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
		
		(imageB, imageA) = images
		
		(kpsA, featuresA, img1) = self.detectKps(imageA)
		(kpsB, featuresB, img2) = self.detectKps(imageB)
		
		M = self.matchKps(kpsA, kpsB,
			featuresA, featuresB, ratio, reprojThresh)
 
		if M is None:
			return None
		
		(matches, H, status) = M
		final = cv2.warpPerspective(imageA, H,
			(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
		#finalimage1=result;
		final[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
		#finalimage2=result;
 				
		if showMatches:
			vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
				status)
 
			return (final, vis)
		return final

	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		
		vis = numpy.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		
		vis[0:hA, 0:wA] = imageA
		vis[0:hB, wA:] = imageB
 
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			if s == 1:
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
 
		return vis

	