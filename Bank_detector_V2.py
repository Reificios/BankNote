from __future__ import print_function

import cv2
import numpy as np

from video import create_capture

def main():
    import sys, getopt

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])

    # Read bank note template picture for SIFT
    bank = cv2.imread("bank100.jpg")
    bank = cv2.resize(bank, (1080, 480))

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    kp1,des1 = sift.detectAndCompute(bank,None)

    # Capturing video camera
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    cam = create_capture(video_src,fallback='synth:bg={}:noise=0.05'.format(cv2.samples.findFile('samples/data/lena.jpg')))

    while True:
        _ret, im = cam.read()
        (ori_h, ori_w, _) = im.shape
        imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        kp2,des2 = sift.detectAndCompute(imGray,None)

        # Find matches between SIFT features
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []

        # Extract the best matches
        for (m1, m2) in matches:
            if m1.distance < 0.6 * m2.distance:
                good_matches.append(m1)

        # If good matches surpass the threshold, draw a box over the bank note and identify the value of the bank note by color thresholding
        GOOD_MATCHES_THRESHOLD = 10
        if len(good_matches) > GOOD_MATCHES_THRESHOLD:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                # Drawing the box [obj_corner = bank template cornet -> dst_corners = corners in video image]
                (h, w, _) = bank.shape
                obj_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst_corners = cv2.perspectiveTransform(obj_corners, M)

                im = cv2.polylines(im, [np.int32(dst_corners)], isClosed=True, color=(0, 255, 0), thickness=2)

                # Find most middle point of the bank note
                textPos_x = 0
                textPos_y = 0

                mid_x = 0
                mid_y = 0
                for i, el in enumerate(np.int32(dst_corners)):
                    if i == 0:
                        textPos_x = el[0][0]
                        textPos_y = el[0][1]
                    mid_x += el[0][0]
                    mid_y += el[0][1]
                mid_x = mid_x // 4
                mid_y = mid_y // 4

                # Handle cases where mid point is outside the boundary of the image
                if(mid_x < 0):
                    mid_x = 0
                if(mid_x > ori_h):
                    mid_x = ori_h-1
                if(mid_y < 0):
                    mid_y = 0
                if(mid_y > ori_w):
                    mid_y = ori_w-1

                # Use HSV color domain since it's easier to seperate
                imHSV = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
                h,s,v = (imHSV[mid_x, mid_y])

                text =""

                # Color Threshold in H domain of HSV
                if(h < 20 or h > 150): # RED
                    text = "100 Baht"
                elif(h > 90): # LIGHT BLUE
                    text = "50 Baht"
                else: # GREEN
                    text = "20 Baht"

                im = cv2.putText(im, text, (textPos_x, textPos_y), cv2.FONT_HERSHEY_SIMPLEX ,  1, (255, 0, 0) , 2, cv2.LINE_AA)

        # Draw Bank note detection result
        cv2.imshow("Bank Note Detection", im)

        # Draw matches between pictures
        img3 = cv2.drawMatches(bank,kp1,imGray,kp2,good_matches,None,flags=2)
        cv2.imshow("Sift Matches", img3)

        if cv2.waitKey(5) == 27:
            break

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
