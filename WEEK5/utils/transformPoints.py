import cv2

def rotatePoints(angle, points, center):
    # Compute original mask
    
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    # Get invert matrix
    M = cv2.invertAffineTransform(M)

    for point in points:
        point2 = point.copy()
        point[0] = int(M[0,0]*point2[0] + M[0,1]*point2[1] + M[0,2])
        point[1] = int(M[1,0]*point2[0] + M[1,1]*point2[1] + M[1,2])
    
    return points
        
        
