import cv2
import numpy as np
import os
from circle_fit import hyperLSQ
import statistics

PIXEL_PER_RATION = 0.0664
arr=[]

def detection(image_path):

    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Replace with your image path

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    t_lower = 60# Lower Threshold
    t_upper = 140 # Upper threshold


    # Apply Canny edge detection
    # Applying the Canny Edge filter
    edge = cv2.Canny(blurred, t_lower, t_upper)

    # Find contours
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    longest_contour = max(contours, key=lambda cnt: cv2.arcLength(cnt, False))

    points = [x[0].tolist() for x in longest_contour]
    xc, yc, r, sigma = hyperLSQ(points)

    diameter = r*2*PIXEL_PER_RATION

    arr.append(diameter)

    # Draw contours on the blank image
    cv2.drawContours(image, contours, -1, (0, 153, 0), 2)  # White for contours
    cv2.putText(image, 'dia: ' + str(diameter), (int(image.shape[1]/2), int(image.shape[0]/2)), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    # Color differentiate circles and scratches
    #result_image = cv2.addWeighted(image, 1, contours, 0.5, 0)

    # Display the result
    #cv2.imshow('Result Image', image)

    # Wait for a key event and close the window
    return image
    

for image_name in os.listdir('gauge'):
    image_path = os.path.join('gauge', image_name)
    result_image = detection(image_path)  # Assuming 'detection' returns the result image

    # Extract the file name from the path
    file_name = os.path.splitext(image_name)[0]

    # Show the canvas with all images
    cv2.imshow(file_name, result_image)
    cv2.imwrite("result.jpg", result_image)


cv2.waitKey(0)
cv2.destroyAllWindows()

print('Results : ',arr)
max = max(arr)
mean = sum(arr)/len(arr)
mode = statistics.mode(arr)
print("Max : ", max )
print("Mean : ", mean)
print("mode", mode)


'''for image in os.listdir('guage'):
    detection('guage/' + image)'''