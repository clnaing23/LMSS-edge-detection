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
    return cv2.GaussianBlur(img, (s, s), 0)

def save(image, str):
    cv2.imwrite(str, image)

def sobel(image, str):

    # Apply Sobel filter in the x and y directions
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    direction = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)


    # Save the result
    save(magnitude, str)
    return (magnitude,direction)

def nonMaxima(magnitude, degree):
    if magnitude.ndim == 3:
        magnitude = np.sqrt(np.sum(magnitude**2, axis=-1))

    rows, cols = magnitude.shape
    result = np.zeros_like(magnitude, dtype=np.uint8)

    angle_quantized = (np.round(degree / 45.0) % 4).astype(int) * 45  # Quantize angles to 0, 45, 90, 135 degrees

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            q = 255  # Default value, pixel is a local maximum

            # Check the neighbors based on the quantized angle
            if angle_quantized[i, j] == 0 and any(magnitude[i, j] >= x for x in [magnitude[i, j + 1], magnitude[i, j - 1]]) and all(magnitude[i, j] >= x for x in [magnitude[i - 1, j], magnitude[i + 1, j]]):
                q = max(magnitude[i, j], magnitude[i, j + 1], magnitude[i, j - 1])
            elif angle_quantized[i, j] == 45 and any(magnitude[i, j] >= x for x in [magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]]) and all(magnitude[i, j] >= x for x in [magnitude[i + 1, j - 1], magnitude[i - 1, j + 1]]):
                q = max(magnitude[i, j], magnitude[i - 1, j + 1], magnitude[i + 1, j - 1])
            elif angle_quantized[i, j] == 90 and any(magnitude[i, j] >= x for x in [magnitude[i + 1, j], magnitude[i - 1, j]]) and all(magnitude[i, j] >= x for x in [magnitude[i, j + 1], magnitude[i, j - 1]]):
                q = max(magnitude[i, j], magnitude[i + 1, j], magnitude[i - 1, j])
            elif angle_quantized[i, j] == 135 and any(magnitude[i, j] >= x for x in [magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]]) and all(magnitude[i, j] >= x for x in [magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]]):
                q = max(magnitude[i, j], magnitude[i - 1, j - 1], magnitude[i + 1, j + 1])

            # Perform non-maximum suppression
            if magnitude[i, j] >= q:
                result[i, j] = magnitude[i, j]
            else:
                result[i, j] = 0

    return result


def edgeLinking(mag, high, low):
    rows, cols = mag.shape
    result = np.zeros_like(mag, dtype=np.uint8)

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
magnitude, direction  = sobel(blurred_image, os.path.join(outputPath, imageName + '_sobel' + filetype))

# Applying non-maxima suppression

non_maximized = nonMaxima(magnitude, direction)

# Applying thresholding


# Applying edge linking
result_image_linked = edgeLinking(magnitude, high=150, low=50)

# Save the results
save(result_image_linked, os.path.join(outputPath, imageName + '_edge_linked' + filetype))
