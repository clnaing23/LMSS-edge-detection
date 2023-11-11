import numpy as np
import cv2
import os


#####
#CHANGE THIS TO THE IMAGE YOU WANT TO USE (MUST BE IN THE PHOTOS FOLDER). 
#MAKE SURE TO INCLUDE THE FILE EXTENSION (.png, .jpg, etc.) and the / at the beginning
#Example: imageFile = '/example.png'

imageFile = '/bulbasaur.jpg'
#####



photoPath = './photos'
imageName = imageFile[1:-4]
filetype = imageFile[-4:]

image = os.path.join(photoPath, imageName + filetype)

def read(image):
    img = cv2.imread(image, 0)
    return img

outputPath = './output'

def guassianBlur(img, s):
    # Apply Gaussian Blur
    return cv2.GaussianBlur(img, (s, s), 0)

def save(image, str):
    cv2.imwrite(str, image)

def sobel(image, str):

    # Apply Sobel filter in the x and y directions
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)


    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)


    # Save the result
    save(magnitude, str)
    return magnitude


def edgeLinking(mag, high, low):
    rows, cols = mag.shape
    result = np.zeros_like(mag, dtype=np.uint8)

    #high low thresholding
    strong_edges = mag > high
    weak_edges = (mag >= low) & (mag <= high)

    # Initialize the result image with strong edges
    result[strong_edges] = 255

    # Define 8-connectivity kernel
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    # Perform edge linking
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if weak_edges[i, j]:
                # Check 8-connectivity
                if np.any(result[i - 1:i + 2, j - 1:j + 2] == 255):
                    result[i, j] = 255

    return result

# Example usage:
input_image = read(image)

# Applying Gaussian Blur
blurred_image = guassianBlur(input_image, s=5)

# Applying Sobel
magnitude  = sobel(blurred_image, os.path.join(outputPath, imageName + '_sobel' + filetype))


# Applying edge linking

# The edge linking function takes the magnitude image and two thresholds
# Adjust the thresholds to get the best results, which differ between images
result_image_linked = edgeLinking(magnitude, high=150, low=50)

# Save the results
save(result_image_linked, os.path.join(outputPath, imageName + '_edge_linked' + filetype))
