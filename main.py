import os
import cv2
import numpy as np
import json
import argparse
from pathlib import Path


def blue_is_white(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_color = np.array([90, 100, 100])
    upper_color = np.array([120, 255, 255])
    mask = cv2.inRange(hsv_image, lower_color, upper_color)
    filtered_image = cv2.bitwise_and(image, image, mask=mask)
    white_pixels = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    white_pixels[np.where(white_pixels != 0)] = 255
    result_image = cv2.bitwise_or(image, cv2.cvtColor(white_pixels, cv2.COLOR_GRAY2BGR))
    return result_image

def find_contours(filename, img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray_image, 112, 255, cv2.THRESH_BINARY_INV)
    thresholded = cv2.medianBlur(thresholded, 11)
    thresholded = cv2.bitwise_not(thresholded)
    edged = cv2.Canny(thresholded, 60, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        height, width = img.shape[:2]
        center_x, center_y = width // 2, height // 2
        quarter_width, quarter_height = width // 4, height // 4

        point1 = (center_x - quarter_width, center_y - quarter_height)
        point2 = (center_x + quarter_width, center_y - quarter_height)
        point3 = (center_x + quarter_width, center_y + quarter_height)
        point4 = (center_x - quarter_width, center_y + quarter_height)

        points_vector = [point1, point2, point3, point4]
        return filename, thresholded, points_vector

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    if area < 500000.0:
        kernel = np.ones((9, 9), np.uint8)
        closed_image = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
        kernel2 = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(closed_image, kernel2, iterations=1)
        _, thresholded = cv2.threshold(erosion, 112, 255, cv2.THRESH_BINARY_INV)
        thresholded = cv2.medianBlur(thresholded, 3)
        thresholded = cv2.bitwise_not(thresholded)
        edged = cv2.Canny(thresholded, 0, 200)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            height, width = img.shape[:2]
            center_x, center_y = width // 2, height // 2
            quarter_width, quarter_height = width // 4, height // 4

            point1 = (center_x - quarter_width, center_y - quarter_height)
            point2 = (center_x + quarter_width, center_y - quarter_height)
            point3 = (center_x + quarter_width, center_y + quarter_height)
            point4 = (center_x - quarter_width, center_y + quarter_height)

            points_vector = [point1, point2, point3, point4]
            return filename, thresholded, points_vector

        largest_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) < 4:
        height, width = img.shape[:2]
        center_x, center_y = width // 2, height // 2
        quarter_width, quarter_height = width // 4, height // 4

        point1 = (center_x - quarter_width, center_y - quarter_height)
        point2 = (center_x + quarter_width, center_y - quarter_height)
        point3 = (center_x + quarter_width, center_y + quarter_height)
        point4 = (center_x - quarter_width, center_y + quarter_height)

        points_vector = [point1, point2, point3, point4]
        return filename, thresholded, points_vector

    sorted_by_sum = sorted(approx.reshape(-1, 2), key=lambda x: x[0] + x[1])
    point1 = sorted_by_sum[0]
    point3 = sorted_by_sum[-1]
    sorted_by_diff = sorted(approx.reshape(-1, 2), key=lambda x: x[0] - x[1])
    point2 = sorted_by_diff[-1]
    point4 = sorted_by_diff[0]

    points_vector = [point1, point2, point3, point4]
    points_tuples = [(point[0], point[1]) for point in points_vector]

    if len(points_vector) != 4:
        height, width = img.shape[:2]
        center_x, center_y = width // 2, height // 2
        quarter_width, quarter_height = width // 4, height // 4

        point1 = (center_x - quarter_width, center_y - quarter_height)
        point2 = (center_x + quarter_width, center_y - quarter_height)
        point3 = (center_x + quarter_width, center_y + quarter_height)
        point4 = (center_x - quarter_width, center_y + quarter_height)

        points_vector = [point1, point2, point3, point4]

    return filename, thresholded, points_vector

def straighten(data, img):
    filename, thresholded, points_vector = data
    src_points = np.float32(points_vector)
    dst_points = np.float32([[0, 10], [920, 10], [920, 228], [0, 228]])
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_image = cv2.warpPerspective(img, perspective_matrix, (924, 250))
    return filename, warped_image


def detect_chars(cut_images):
    filename, image = cut_images
    image = blue_is_white(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (25, 25), 0)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5, cv2.convertScaleAbs(sobel_y), 0.5, 0)
    _, binary = cv2.threshold(sobel, 134, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > w:
            area = w * h
            rectangles.append(((x, y, w, h), area))
    rectangles.sort(key=lambda x: x[1], reverse=True)
    largest_rectangles = []
    for rect in rectangles[:8]:
        largest_rectangles.append(rect)

    heights = [rect[0][3] for rect in largest_rectangles]

    if heights:
        avg_height = sum(heights) / len(heights)
        filtered_rectangles = [rect for rect in largest_rectangles if rect[0][3] >= 0.8 * avg_height]
        filtered_rectangles.sort(key=lambda rect: rect[0][0])

        cut_images = []
        for rect in filtered_rectangles:
            (x, y, w, h), _ = rect
            cut_image = image[y:y + h, x:x + w].copy()
            cut_images.append(cut_image)
    else:
        height, width = image.shape[:2]
        parts = 8
        step = width // parts
        cut_images = []
        for i in range(parts):
            start_x = i * step
            end_x = (i + 1) * step if i < parts - 1 else width
            cut_image = image[:, start_x:end_x].copy()
            cut_images.append(cut_image)

    return cut_images

def compare_with_templates(cut_images, template_folder):
    matched_letters = []
    for i, cut_image in enumerate(cut_images):
        gray_cut_image = cv2.cvtColor(cut_image, cv2.COLOR_BGR2GRAY)
        _, cut_image_thresh = cv2.threshold(gray_cut_image, 128, 255, cv2.THRESH_BINARY)
        kernel = np.ones((7, 7), np.uint8)
        cut_image_thresh = cv2.morphologyEx(cut_image_thresh, cv2.MORPH_CLOSE, kernel)
        cut_image_thresh = cv2.medianBlur(cut_image_thresh, 7)
        best_match_name = None
        best_match_value = float('inf')
        I_min_val = None

        for template_name in os.listdir(template_folder):
            if i < 2 and template_name.startswith(tuple(str(x) + ".png" for x in range(10))):
                continue

            template_path = os.path.join(template_folder, template_name)
            template_image = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            _, template_thresh = cv2.threshold(template_image, 128, 255, cv2.THRESH_BINARY)
            resized_cut_image = cv2.resize(cut_image_thresh, (template_thresh.shape[1], template_thresh.shape[0]))
            result = cv2.matchTemplate(resized_cut_image, template_thresh, cv2.TM_SQDIFF)
            min_val, _, _, _ = cv2.minMaxLoc(result)

            if template_name == "I.png":
                I_min_val = min_val
            else:
                if min_val < best_match_value:
                    best_match_value = min_val
                    best_match_name = template_name

        if I_min_val is not None and (I_min_val - best_match_value) < -200000000.0:
            best_match_name = "I.png"

        matched_letters.append(best_match_name[0])

    return ''.join(matched_letters)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str)
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    results_file = Path(args.results_file)
    template_folder = 'dane'

    images_paths = sorted([image_path for image_path in images_dir.iterdir() if image_path.name.endswith('.jpg')])
    results = {}
    for image_path in images_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f'Error loading image {image_path}')
            continue
        filename = image_path.name  #załaduj nazwy zdjec
        w1 = find_contours(filename, image) #filtruje obraz, wykrywa ramki tablicy rejestracyjnej i zwraca 4 punkty
        cut_images = straighten(w1, image) #na podstawie 4 punktow tworzy macierz, prostuje zdjecie i wycina obraz z rejestracją
        w2 = detect_chars(cut_images) #znajduje litery na obrazie a nstepnie je wycina
        matched_string = compare_with_templates(w2, template_folder) #dopasowuje wyciete litery do temlates i zwraca nazwe templates (literki)
        results[filename] = matched_string #zapisz do pliku json

        #to tak krótko opisane, wiecej jest w pliku pdf

    with results_file.open('w') as output_file:
        json.dump(results, output_file, indent=4)

if __name__ == '__main__':
    main()
