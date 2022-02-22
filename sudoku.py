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

    


    # out.release()
    cap.release()
    cv2.destroyAllWindows()
