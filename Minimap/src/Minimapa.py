import cv2
import numpy as np

class Minimapa:
    def __init__(self, im_src, im_dst, pts_src, pts_dst):
        self.pts_src = pts_src
        self.pts_dst = pts_dst
        
        self.im_src = im_src
        self.im_dst = im_dst
        
    def representa_punto(self, pt, color):
        if len(self.pts_src) == 4:
            h, status = cv2.findHomography(self.pts_src, self.pts_dst)
            im_out = cv2.warpPerspective(self.im_src, h, (self.im_dst.shape[1], self.im_dst.shape[0]))

            pts = np.float32(pt).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, h)

            cv2.circle(self.im_dst, ((int(dst[0][0][0])), int(dst[0][0][1])), 6, color, -1)
            cv2.imshow("minimapa", self.im_dst)