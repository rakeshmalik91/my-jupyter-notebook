import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.patches as patches

import pytesseract

import mynnlib
from mynnlib import *






def load_sudoku_image(image_path):
    # Load the Sudoku image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply preprocessing (Thresholding)
    _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours to detect grid
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area and select the largest (assuming it's the grid)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Set default width and height (in case grid detection fails)
    width, height = 450, 450  
    
    # Get bounding box of the grid
    sudoku_grid = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # Looking for a quadrilateral (Sudoku Grid)
            sudoku_grid = approx
            break
    
    # Warp perspective if necessary (To get a straight grid)
    if sudoku_grid is not None:
        pts = np.array([sudoku_grid[i][0] for i in range(4)], dtype="float32")
    
        # Sort the points in the correct order
        def order_points(pts):
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)  # Sum of x and y coordinates
            diff = np.diff(pts, axis=1)  # Difference (y - x)
    
            rect[0] = pts[np.argmin(s)]  # Top-left
            rect[2] = pts[np.argmax(s)]  # Bottom-right
            rect[1] = pts[np.argmin(diff)]  # Top-right
            rect[3] = pts[np.argmax(diff)]  # Bottom-left
    
            return rect
    
        ordered_pts = order_points(pts)  # Ensure correct order
        pts_dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
    
        # Compute transformation matrix
        M = cv2.getPerspectiveTransform(ordered_pts, pts_dst)
        image = cv2.warpPerspective(image, M, (width, height))
    else:
        image = cv2.resize(image, (width, height))  # Resize image instead if grid detection fails
    
    plt.figure(figsize=(5,5))
    plt.imshow(image, cmap='gray')
    plt.axis("off")
    plt.show()

    return image




def read_sudoku_from_image_using_tesseract(image, actual_grid):
    width, height = image.shape
    
    cell_size = width // 9
    sudoku_numbers = []
    undetected_cnt = 0
    incorrect_cnt = 0
    
    cells = []
    
    def isdigit(number):
        return number.isdigit() and len(number) == 1
    
    for row in range(9):
        row_numbers = []
        for col in range(9):
            x, y = col * cell_size, row * cell_size
            cell = image[y:y + cell_size, x:x + cell_size]
            
            cwidth, cheight = cell.shape
            cell = cell[int(cwidth*0.1):int(cwidth*0.95), int(cheight*0.2):int(cheight*0.9)]
            
            cell = cv2.GaussianBlur(cell, (3,3), 0)
            _, cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            kernel = np.ones((2, 2), np.uint8)
            cell = cv2.morphologyEx(cell, cv2.MORPH_OPEN, kernel)
    
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=123456789'
            number = pytesseract.image_to_string(cell, config=custom_config, lang="eng").strip()
    
            if not isdigit(number):
                cell = cv2.morphologyEx(cell, cv2.MORPH_CLOSE, kernel)
                number = pytesseract.image_to_string(cell, config=custom_config, lang="eng").strip()
    
            if not isdigit(number):
                cell = cv2.bitwise_not(cell)
                cell = cv2.dilate(cell, kernel, iterations=1)
                cell = cv2.bitwise_not(cell)
                number = pytesseract.image_to_string(cell, config=custom_config, lang="eng").strip()
    
            incorrect = isdigit(number) and str(actual_grid[row][col]) != number
            undetected = not isdigit(number) and (not actual_grid or actual_grid[row][col] != 0)
            cells += [{'img': cell, 'undetected': undetected, 'number': number, 'incorrect': incorrect}]
    
            if undetected:
                undetected_cnt += 1
            if incorrect:
                incorrect_cnt += 1
    
            # Store the recognized number
            row_numbers.append(number if isdigit(number) else "_")
        
        sudoku_numbers.append(row_numbers)
    
    print(f"undetected = {undetected_cnt}")
    print(f"incorrect = {incorrect_cnt}")

    # plot figure
    fig, axes = plt.subplots(9, 9, figsize=(5, 5)) 
    for ax, img in zip(axes.ravel(), cells):
        ax.imshow(img['img'], cmap='gray')
        ax.axis('off')
        height, width = img['img'].shape
        if img['undetected'] or img['incorrect']:
            rect = patches.Rectangle(
                (0, 0), width-2, height-2, linewidth=1, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
        if not img['undetected']:
            ax.text(
                height, width,
                img['number'],
                color='blue', fontsize=12, fontweight='normal',
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0, edgecolor='none')
            )
    plt.tight_layout()
    plt.show()

    return sudoku_numbers


def resnet_image_to_digit(model_data, digit_image, threshold):
    model_data['model'].eval()
    digit_image = model_data['transform']['val'](digit_image).unsqueeze(0).to(model_data['device'])
    with torch.no_grad():
        outputs = model_data['model'](digit_image)
        probabilities = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, 1)
        if top_probs[0].item() < threshold:
            return ""
        # print(top_probs[0].item())
    return f"{top_indices[0].item() + 1}"

def resnet_cv2_grey_image_to_digit(model_data, digit_image, threshold=0.5):
    digit_image = cv2.cvtColor(digit_image, cv2.COLOR_GRAY2BGR)
    digit_image = resnet_image_to_digit(model_data, Image.fromarray(digit_image), threshold)
    return digit_image

def read_sudoku_from_image_using_resnet(image, actual_grid, threshold, checkpoint_path="sudoku/checkpoint.pth"):
    model_data = torch.load(checkpoint_path, weights_only=False)
    
    width, height = image.shape
    
    cell_size = width // 9
    sudoku_numbers = []
    undetected_cnt = 0
    incorrect_cnt = 0
    
    cells = []
    
    def isdigit(number):
        return number.isdigit() and len(number) == 1
    
    for row in range(9):
        row_numbers = []
        for col in range(9):
            x, y = col * cell_size, row * cell_size
            cell = image[y:y + cell_size, x:x + cell_size]
            
            cwidth, cheight = cell.shape
            cell = cell[int(cwidth*0.1):int(cwidth*0.95), int(cheight*0.2):int(cheight*0.9)]
            
            cell = cv2.GaussianBlur(cell, (3,3), 0)
            _, cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            kernel = np.ones((2, 2), np.uint8)
            cell = cv2.morphologyEx(cell, cv2.MORPH_OPEN, kernel)
    
            number = resnet_cv2_grey_image_to_digit(model_data, cell, threshold)
    
            incorrect = isdigit(number) and actual_grid and str(actual_grid[row][col]) != number
            undetected = not isdigit(number) and (not actual_grid or actual_grid[row][col] != 0)
            cells += [{'img': cell, 'undetected': undetected, 'number': number, 'incorrect': incorrect}]
    
            if undetected:
                undetected_cnt += 1
            if incorrect:
                incorrect_cnt += 1
    
            # Store the recognized number
            row_numbers.append(number if isdigit(number) else "_")
        
        sudoku_numbers.append(row_numbers)
    
    print(f"undetected = {undetected_cnt}")
    if actual_grid:
        print(f"incorrect = {incorrect_cnt}")

    fig, axes = plt.subplots(9, 9, figsize=(5, 5)) 
    for ax, img in zip(axes.ravel(), cells):
        ax.imshow(img['img'], cmap='gray')
        ax.axis('off')
        height, width = img['img'].shape
        if img['undetected'] or img['incorrect']:
            rect = patches.Rectangle(
                (0, 0), width-2, height-2, linewidth=1, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
        if not img['undetected']:
            ax.text(
                height, width,
                img['number'],
                color='blue', fontsize=12, fontweight='normal',
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0, edgecolor='none')
            )
    
    plt.tight_layout()
    plt.show()

    return sudoku_numbers

def read_sudoku_from_image_hybrid(image, actual_grid, resnet_threshold, tess_mode=1, checkpoint_path="sudoku/checkpoint.pth"):
    model_data = torch.load(checkpoint_path, weights_only=False)
    
    width, height = image.shape
    
    cell_size = width // 9
    sudoku_numbers = []
    undetected_cnt = 0
    incorrect_cnt = 0
    
    cells = []
    
    def isdigit(number):
        return number.isdigit() and len(number) == 1
    
    for row in range(9):
        row_numbers = []
        for col in range(9):
            x, y = col * cell_size, row * cell_size
            cell = image[y:y + cell_size, x:x + cell_size]
            
            cwidth, cheight = cell.shape
            cell = cell[int(cwidth*0.1):int(cwidth*0.95), int(cheight*0.2):int(cheight*0.9)]
            
            cell = cv2.GaussianBlur(cell, (3,3), 0)
            _, cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            kernel = np.ones((2, 2), np.uint8)
            cell = cv2.morphologyEx(cell, cv2.MORPH_OPEN, kernel)
    
            number = resnet_cv2_grey_image_to_digit(model_data, cell, resnet_threshold)

            method = "resnet"

            if not isdigit(number) and tess_mode >= 1:
                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=123456789'
                number = pytesseract.image_to_string(cell, config=custom_config, lang="eng").strip()
                method = "tesseract-phase-1"

            if not isdigit(number) and tess_mode >= 2:
                cell = cv2.morphologyEx(cell, cv2.MORPH_CLOSE, kernel)
                number = pytesseract.image_to_string(cell, config=custom_config, lang="eng").strip()
                method = "tesseract-phase-2"
    
            if not isdigit(number) and tess_mode >= 3:
                cell = cv2.bitwise_not(cell)
                cell = cv2.dilate(cell, kernel, iterations=1)
                cell = cv2.bitwise_not(cell)
                number = pytesseract.image_to_string(cell, config=custom_config, lang="eng").strip()
                method = "tesseract-phase-3"
    
            incorrect = isdigit(number) and actual_grid and str(actual_grid[row][col]) != number
            undetected = not isdigit(number) and (not actual_grid or actual_grid[row][col] != 0)
            cells += [{'img': cell, 'undetected': undetected, 'number': number, 'incorrect': incorrect, 'method': method}]
    
            if undetected:
                undetected_cnt += 1
            if incorrect:
                incorrect_cnt += 1
    
            # Store the recognized number
            row_numbers.append(number if isdigit(number) else "_")
        
        sudoku_numbers.append(row_numbers)
    
    print(f"undetected = {undetected_cnt}")
    if actual_grid:
        print(f"incorrect = {incorrect_cnt}")

    fig, axes = plt.subplots(9, 9, figsize=(5, 5)) 
    for ax, img in zip(axes.ravel(), cells):
        ax.imshow(img['img'], cmap='gray')
        ax.axis('off')
        height, width = img['img'].shape
        if img['undetected'] or img['incorrect']:
            rect = patches.Rectangle(
                (0, 0), width-2, height-2, linewidth=1, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
        elif 'tesseract' in img['method']:
            rect = patches.Rectangle(
                (2, 2), width-4, height-4, linewidth=1, edgecolor='orange', facecolor='none'
            )
            ax.add_patch(rect)
        if not img['undetected']:
            ax.text(
                height, width,
                img['number'],
                color='blue', fontsize=12, fontweight='normal',
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0, edgecolor='none')
            )
    
    plt.tight_layout()
    plt.show()

    return sudoku_numbers






def is_valid_sudoku(grid):
    def is_unique(lst):
        """Check if a list contains unique numbers (ignoring '_')."""
        nums = [num for num in lst if num != "_"]
        return len(nums) == len(set(nums))  # Ensure no duplicates

    # Check rows
    for row in grid:
        if not is_unique(row):
            return False

    # Check columns
    for col in range(9):
        if not is_unique([grid[row][col] for row in range(9)]):
            return False

    # Check 3x3 subgrids
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            subgrid = [
                grid[r][c]
                for r in range(box_row, box_row + 3)
                for c in range(box_col, box_col + 3)
            ]
            if not is_unique(subgrid):
                return False

    return True
