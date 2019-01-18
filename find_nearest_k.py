#!/usr/bin/env python
import math

pts = [(1,0), (2,0), (-2,1), (3,1), (0.5, -0.5), (5,2), (10,0), (0.1, 0.2), (4,2)]
origin = (0,0)

def find_nearest_k_points(origin, pts, k):
    
    distance = []

    for idx, pt in enumerate(pts):
        d = math.sqrt( 1.0*(pt[0] - origin[0]) ** 2 + 1.0*(pt[1] - origin[1]) ** 2 )
        coordinate = pts[idx]
        distance.append( (d, coordinate) )

    res = []
    distance.sort(key=sort_first)

    for pair in distance:
        d = pair[0]
        coordinate = pair[1]
        res.append((d,coordinate))

    return res[0:k]

def sort_first(val):
    return val[0]    

def max_heap(arr):
    pass

print(pts)
print(find_nearest_k_points(origin, pts, 5))