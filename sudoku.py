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

            if flag == 0:

                extracted_image_text = []
                for y in range(9):
                    extracted_line = "" # extracts the 9 rows of lines from the image
                    for x in range(9):
                        y2min = y * case + marge
                        y2max = (y + 1) * case - marge
                        x2min = x * case + marge
                        x2max = (x + 1) * case - marge
                        img = gray_scale_extracted_points[y2min:y2max, x2min:x2max]
                        x = img.reshape(1, 28, 28, 1)
                        if x.sum() > 10000:
                            prediction = classifier.predict_classes(x)
                            extracted_line += "{:d}".format(prediction[0])
                        else:
                            extracted_line += "{:d}".format(0)
                    extracted_image_text.append(extracted_line)
                print(extracted_image_text)
                result = sol.sudoku(extracted_image_text)
                solution_found = True
            # print("Result:", result)
            if solution_found:
            	# screen_shot = pyautogui.screenshot()
            	# screen_shot = cv2.cvtColor(np.array(screen_shot), cv2.COLOR_RGB2BGR)
            	cv2.imwrite("Solution.png", frame)
            # cv2.imshow('results', result)

            if result is not None:
                flag = 1
                fond = np.zeros(
                    shape=(grid_size, grid_size, 3), dtype=np.float32)
                for y in range(len(result)):
                    for x in range(len(result[y])):
                        if extracted_image_text[y][x] == "0":
                            cv2.putText(fond, "{:d}".format(result[y][x]), ((
                                x) * case + marge + 3, (y + 1) * case - marge - 3), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.9, (0, 0, 255), 1)
                M = cv2.getPerspectiveTransform(coord2, coord1)
                h, w, c = frame.shape
                fondP = cv2.warpPerspective(fond, M, (w, h))
                img2gray = cv2.cvtColor(fondP, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                mask = mask.astype('uint8')
                mask_inv = cv2.bitwise_not(mask)
                img1_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
                img2_fg = cv2.bitwise_and(fondP, fondP, mask=mask).astype('uint8')
                dst = cv2.add(img1_bg, img2_fg)
                dst = cv2.resize(dst, (1080, 620))
                cv2.imshow("frame", dst)
                cv2.imwrite("dst2.png", dst)
                # cv2.imwrite("mask_inv.png", mask_inv)
                # cv2.imwrite("img1_bg.png", img1_bg)
                # cv2.imwrite("img2_fg.png", img2_fg)
                # print("3")
                # out.write(dst)

            else:
                frame = cv2.resize(frame, (1080, 620))
                cv2.imshow("frame", frame)
                # print("4")
                # out.write(frame)

        else:
            flag = 0
            frame = cv2.resize(frame, (1080, 620))
            cv2.imshow("frame", frame)
            # print("5")
            # out.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


    # out.release()
    cap.release()
    cv2.destroyAllWindows()
