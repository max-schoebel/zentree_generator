import matplotlib.pyplot as plt
import random
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
import cv2


def hextorgb(hexstring):
    r, g, b = int(hexstring[0:2], 16), int(hexstring[2:4], 16), int(hexstring[4:6], 16)
    return (r, g, b)


def dist(p1, p2):
    acc = 0
    for i in range(len(p1)):
        acc = acc + (p1[i] - p2[i]) ** 2
    return math.sqrt(acc)


def k_means(points, k):
    # init mean points
    means = []
    for i in range(k):
        means.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    
    nearest = []
    for p in points:
        dists = [dist(p, mean) for mean in means]
        nearest.append(dists.index(min(dists)))
    
    cluster_indxs = []
    for nearest_indx in range(k):
        cluster_indxs.append([i for i, x in enumerate(nearest) if x == nearest_indx])
    
    newmeans = []
    for indxs in cluster_indxs:
        if indxs != []:
            newmeans.append(np.mean(points[indxs], 0))
        else:
            newmeans.append([])
    
    return np.asarray(newmeans)


MAX_DEPTH = 15
SIGMA_ANGLE = 0.1

def length_scaling(depth):
    return math.exp(-depth/5)

def recursive_branch(image, init_length, depth, angle, start_point):
    # cv2.line(image, (0,0), (50,50), (233,233,233), 5)
    if False:
    # if (depth > MAX_DEPTH):
        return
    else:
        
        length = init_length * length_scaling(depth)
        length = np.random.normal(length, length/3/3)
        
        end_x = start_point[0] + length * math.cos(angle)
        end_y = start_point[1] - length * math.sin(angle)
        end_point = (int(end_x), int(end_y))
        
        # start_indx = int((depth -1)/MAX_DEPTH * len(zen_points))
        # end_indx = int(depth/MAX_DEPTH * len(zen_points))
        
        start_indx = int(((depth -1) / MAX_DEPTH) ** 1 * len(zen_points))
        end_indx = int((depth / MAX_DEPTH) ** 1 * len(zen_points))
        
        clr = zen_points[min(np.random.randint(start_indx, end_indx), len(zen_points)-1)]
        clr1, clr2, clr3 = clr.astype(np.int32)
        cv2.line(image, start_point, end_point, (int(clr1), int(clr2), int(clr3)), 2, cv2.LINE_AA)
        
        # angle1, angle2 = (math.pi/5, math.pi/5)
        angle1, angle2 = np.random.normal(math.pi / (6 + 0.6 * depth), SIGMA_ANGLE, 2)
        angle1 = angle - abs(angle1)
        angle2 = angle + abs(angle2)
        
        if np.random.rand() > 0.7 * min(math.exp(1 * (depth - MAX_DEPTH)), 1):
        # if np.random.rand() < 0.9 ** (0.5 * depth):
            recursive_branch(image, init_length, depth + 1, angle1, end_point)
            recursive_branch(image, init_length, depth + 1, angle2, end_point)


if __name__ == '__main__':
    global zen_points
    zen_points = np.array([hextorgb(string) for string in open('zenburn_colors.txt', 'r')])
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # xs = zen_points[:,0]
    # ys = zen_points[:,1]
    # zs = zen_points[:,2]
    # ax.scatter(xs, ys, zs, c=zen_points/255)
    # ax.set_xlabel('R')
    # ax.set_ylabel('G')
    # ax.set_zlabel('B')
    # plt.show()
    
    size_tuple = (1200, 1920, 3)
    (width, height, _) = size_tuple
    
    img = np.full(size_tuple, hextorgb("3f3f3f"), dtype=np.uint8)
    recursive_branch(img, 200, 1, math.pi/2, (700, 1200))
    
    # cv2.line(img, (960,1200), (960, 800), (255,0,0), 3)
    # plt.imshow(img)
    # plt.show()
    plt.imsave("/home/max/.wallpaper/wallpaper.png", img)

