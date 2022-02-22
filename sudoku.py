import cv2
import numpy as np
import operator
import pyautogui
import imutils
import kivy
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import sudoku_solver as sol



def play():

    classifier = load_model("./digit_model.h5")

    marge = 4
    case = 28 + 2 * marge
    grid_size = 9 * case

    cap = cv2.VideoCapture(0)
    flag = 0
    solution_found = False

    while True:

        ret, frame = cap.read()

        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_scale = cv2.GaussianBlur(gray_scale, (7, 7), 0)
        thresh = cv2.adaptiveThreshold(
            gray_scale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)

        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_size = None
        maxArea = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 25000:
                arc_length = cv2.arcLength(contour, True)
                polygone = cv2.approxPolyDP(contour, 0.01 * arc_length, True)
                if area > maxArea and len(polygone) == 4:
                    contour_size = polygone
                    maxArea = area

        if contour_size is not None:
            cv2.drawContours(frame, [contour_size], 0, (0, 255, 0), 2)
            coordinates = np.vstack(contour_size).squeeze()
            coordinates = sorted(coordinates, key=operator.itemgetter(1))

            if coordinates[0][0] < coordinates[1][0]:
                if coordinates[3][0] < coordinates[2][0]:
                    coord1 = np.float32([coordinates[0], coordinates[1], coordinates[3], coordinates[2]])
                else:
                    coord1 = np.float32([coordinates[0], coordinates[1], coordinates[2], coordinates[3]])
            else:
                if coordinates[3][0] < coordinates[2][0]:
                    coord1 = np.float32([coordinates[1], coordinates[0], coordinates[3], coordinates[2]])
                else:
                    coord1 = np.float32([coordinates[1], coordinates[0], coordinates[2], coordinates[3]])
            coord2 = np.float32([[0, 0], [grid_size, 0], [0, grid_size], [
                              grid_size, grid_size]])
            M = cv2.getPerspectiveTransform(coord1, coord2)
            # is the original sudoku without the soluution
            gray_scale_extracted_points = cv2.warpPerspective(frame, M, (grid_size, grid_size))
            gray_scale_extracted_points = cv2.cvtColor(gray_scale_extracted_points, cv2.COLOR_BGR2GRAY)
            gray_scale_extracted_points = cv2.adaptiveThreshold(
                gray_scale_extracted_points, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 3)

            cv2.imshow("Extracted #s", gray_scale_extracted_points)

          


    # out.release()
    cap.release()
    cv2.destroyAllWindows()
