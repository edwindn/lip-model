import matplotlib.pyplot as plt
import ast
import os
import numpy as np
import re
import sys
import cv2
import dlib
from collections import defaultdict
import shutil


# ERRORS FIXED:
# 1 - THE DOTS DETECTED ARE UPSIDE DOWN
# 2 - DO NOT RESCALE BUT ONLY CENTER

SCALE_FACTOR = 0.01 # we aim to have x values between -1 and 1
SHIFT_UP = 0.5


# Initialize error tracking
error_log = defaultdict(lambda: {"count": 0, "files": []})

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def log_error(error_type, file_path):
    error_log[error_type]["count"] += 1
    error_log[error_type]["files"].append(file_path)


def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

def create_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)

delete_folder('mask')
delete_folder('modified_images')
delete_folder('mouth_landmarks')

create_folder('mask')
create_folder('modified_images')
create_folder('mouth_landmarks')

# 1. Save the landmarks and visualize them

def plot_lms(lms, path):
    try:
        xs, ys = zip(*lms)
        plt.scatter(xs, ys)
        plt.savefig(path)
    except Exception as e:
        log_error(f"Error in plot_lms: {e}", path)

def rescale_lms(lms, file='N/A'):
    if lms.shape != (20, 2):
        log_error('Lms shape', f'{lms.shape}, {file}')
    try:
        lms_x = lms[:, 0]
        lms_y = lms[:, 1]
        x_mean = np.mean(lms_x)
        lms_x = lms_x - x_mean
        y_mean = np.mean(lms_y)
        lms_y = lms_y - y_mean

        scale_factor = SCALE_FACTOR
        lms_y = lms_y * scale_factor
        lms_x = lms_x * scale_factor

        lms_y = - lms_y

        y_max = np.max(lms_y)
        lms_y = lms_y - y_max + SHIFT_UP

        out = np.column_stack((lms_x, lms_y))
        return out
    except Exception as e:
        log_error(f"Error in rescale_lms: {e}", "N/A")
        return np.array([])

def remove_green(img):
    try:
        img_copy = img.copy()
        hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        img_copy[mask != 0] = [128, 128, 128]  # Change green pixels to gray
        return img_copy
    except Exception as e:
        log_error(f"Error in remove_green: {e}", "N/A")
        return img

def get_landmarks(img_path):
    try:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = detector(gray)

        if len(faces) == 0:
            log_error('No face initially detected', img_path)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            faces = detector(gray)

        if len(faces) == 0:
            log_error('No face detected', img_path)

        all_landmarks = []

        for face in faces:
            landmarks = predictor(gray, face)

            for i in range(48, 68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                all_landmarks.append([x, y])
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

        path = f'mouth_landmarks/{os.path.basename(os.path.dirname(img_path))}_{os.path.basename(img_path)}'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, img)

        all_landmarks = np.array(all_landmarks)
        try:
            norm_landmarks = rescale_lms(all_landmarks, path)
            return norm_landmarks.tolist()
        except Exception as e:
            log_error(f"Error in rescaling landmarks: {e}", img_path)
            print(img_path)
    except Exception as e:
        log_error(f"Error in get_landmarks: {e}", img_path)

def get_mask(img_path):
    try:
        image = cv2.imread(img_path)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])

        mask = cv2.inRange(hsv_image, lower_green, upper_green)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        coordinates = []

        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                coordinates.append((cx, cy)) # CORRECT FOR INVERSION

        for coord in coordinates:
            cv2.circle(image, coord, 10, (255, 0, 0), 2)

        path = f'mask/{os.path.basename(os.path.dirname(img_path))}_{os.path.basename(img_path)}'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, image)
        coordinates = np.array(coordinates)
        coordinates = rescale_lms(coordinates, img_path)
        return coordinates.tolist()
    except Exception as e:
        log_error(f"Error in get_mask: {e}", img_path)
        return []

def numeric_sort_key(s):
    try:
        match = re.search(r'/(\d+)_', s)
        return int(match.group(1)) if match else float('inf')
    except Exception as e:
        log_error(f"Error in numeric_sort_key: {e}", s)
        return float('inf')

def get_image_paths(folder):
    try:
        paths = []
        for item in os.listdir(folder):
            if item.endswith('png'):
                paths.append(os.path.join(folder, item))
        paths = sorted(paths, key=numeric_sort_key)
        return paths
    except Exception as e:
        log_error(f"Error in get_image_paths: {e}", folder)
        return []

def save_landmarks(folder):
    try:
        path_arr = get_image_paths(folder)

        landmark_output = []
        for path in path_arr:
            lms = get_mask(path)
            if np.array(lms).shape != (20, 2):
                print(f'Dots not found for image {path}')
                #print(np.array(lms).shape)
                img = cv2.imread(path)
                img_no_green = remove_green(img)  # Use a copy to avoid modifying the original
                modified_path = os.path.join('modified_images', os.path.basename(path))
                os.makedirs(os.path.dirname(modified_path), exist_ok=True)
                cv2.imwrite(modified_path, img_no_green)
                lms = get_landmarks(modified_path)  # Use the modified image path
            #print(np.array(lms).shape)
            if np.array(lms).shape != (20, 2): # MUST BE THE CORRECT SHAPE HERE
                raise ValueError
            landmark_output.append(lms)

        with open(f'{folder}_landmark_output.txt', 'w') as file:
            file.write(str(landmark_output))

        print(f"Saved array of {len(landmark_output)} mouth landmark(s)")

        # Save the shapes of the arrays in folder_LOG.txt
        #with open(f'{folder}_folder_LOG.txt', 'w') as log_file:
        #    log_file.write(f'Shape of landmark_output: {np.array(landmark_output).shape}\n')

    except Exception as e:
        log_error(f"Error in save_landmarks: {e}", folder)

# 2. Get training data as txt files

def get_data(label_folder, index):
    try:
        lm_path = f'{label_folder}_landmark_output.txt'
        with open(lm_path, 'r') as lm_file:
            lm_data = lm_file.read()
        lm_file = ast.literal_eval(lm_data)
        
        num_lms = len(lm_file)
        print(f'Number of samples: {num_lms}')
        
        lm_file = np.array(lm_file)
        model_inputs = lm_file.reshape(num_lms, -1)  # array of mouth landmarks
        model_inputs = model_inputs.tolist()
        
        INPUT_DIM = len(model_inputs[0])
        print(f'Input dim: {INPUT_DIM}')
        
        coord_paths = [os.path.join(label_folder, f) for f in os.listdir(label_folder) if f.endswith('txt')]
        coord_paths = sorted(coord_paths, key=numeric_sort_key)
        
        all_coords = []
        for path in coord_paths:
            with open(path, 'r') as coord_file:
                coords = coord_file.read()
                parsed_coords = ast.literal_eval(coords)
            shape = np.array(parsed_coords).shape
            if shape != (5,):
                print(shape)
            
            if parsed_coords:
                all_coords.append(parsed_coords)
            else:
                log_error(f"Empty or invalid coordinates in file: {path}", path)
        
        if not all_coords:
            raise ValueError("No valid coordinate data found.")
        
        model_labels = all_coords
        OUTPUT_DIM = len(model_labels[0])
        print(f'Output dim: {OUTPUT_DIM}')
        
        with open(f'{index}_model_inputs.txt', 'w') as f: ### label_folder
            f.write(str(model_inputs))
        
        with open(f'{index}_model_labels.txt', 'w') as f: ### label_folder
            f.write(str(model_labels))

    
    except Exception as e:
        log_error(f"Error in get_data: {e}", label_folder)


def main():
    try:
        if len(sys.argv) != 2:
            print("Usage: python file.py <input>")
            sys.exit(1)

        pfolder = sys.argv[1]
        folders = [f for f in os.listdir(pfolder) if os.path.isdir(os.path.join(pfolder, f))]
        folders = sorted(folders)

        INDEX = 112
        for folder in folders:
            INDEX += 1 ###
            folder = os.path.join(pfolder, folder)
            save_landmarks(folder)
            print('Saved images at ./mouth_landmarks')
            print(f'Saved landmarks as {folder}_landmark_output.txt')

            get_data(folder, INDEX)
            print(f'Saved data as {folder}_model_inputs.txt and {folder}_model_labels.txt')

    except Exception as e:
        log_error(f"Error in main: {e}", "N/A")

if __name__ == '__main__':
    main()

    