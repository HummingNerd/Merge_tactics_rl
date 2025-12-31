import cv2 as cv
import numpy as np
from mss import mss
import pyautogui
import time
import random

### When Doing Actions Make Sure to Add Window Displacement Factor

def detect(needle, haystack_gray, draw_img=None, threshold=0.97):
    res = cv.matchTemplate(haystack_gray, needle, cv.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv.minMaxLoc(res)

    if max_val >= threshold:
        if draw_img is not None:
            h, w = needle.shape[:2]
            cv.rectangle(
                draw_img,
                max_loc,
                (max_loc[0] + w, max_loc[1] + h),
                (0, 0, 255),
                2
            )
        return True, max_loc[0]+w/2, max_loc[1]+h/2
    return False, 0, 0

# -----------------------
# Config
# -----------------------
scale = 0.5
screen = {"left": 880, "top": 58, "width": 370, "height": 555}

shop = [
    {"top": 640, "left": 978,  "width": 30, "height": 30},
    {"top": 640, "left": 1040, "width": 30, "height": 30},
    {"top": 645, "left": 1102, "width": 30, "height": 30}
]

deploy_phase = cv.imread("assets/deploy_phase.png")
deploy_phase = cv.resize(deploy_phase, (0, 0), fx=scale, fy=scale)
deploy_phase = cv.cvtColor(deploy_phase, cv.COLOR_BGRA2BGR)

play_again = cv.imread("assets/play_again.png")
play_again = cv.resize(play_again, (0, 0), fx=scale, fy=scale)
play_again = cv.cvtColor(play_again, cv.COLOR_BGRA2BGR)



lower_blue = np.array([70, 80, 0])
upper_blue = np.array([102, 255, 255])

# -----------------------
# Load & preprocess NEEDLE (ONCE)
# -----------------------
needle = cv.imread("assets/needle.png")
needle_bgr = cv.resize(needle, (0, 0), fx=scale, fy=scale)

needle_hsv = cv.cvtColor(needle_bgr, cv.COLOR_BGR2HSV)

needle_mask = cv.inRange(needle_hsv, lower_blue, upper_blue)

needle_filtered = cv.bitwise_and(needle_bgr, needle_bgr, mask=needle_mask)

# -----------------------
# Screen capture
# -----------------------
itr = 1980
itr_shop = 660
itr_image = 1980
itr_deploy = 1980
sct = mss()

while pyautogui.position()[1] > 25:
    # -----------------------
    # Capture frame
    # -----------------------
    frame = np.array(sct.grab(screen))
    frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)
    frame = cv.resize(frame, (0, 0), fx=scale, fy=scale)

    in_deploy_phase, _, _ = detect(deploy_phase, frame, True)

    # -----------------------
    # Frame HSV + mask
    # -----------------------
    if True:
        if itr_deploy <= 1:
            time.sleep(5)

            frame = np.array(sct.grab(screen))
            frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)
            frame = cv.resize(frame, (0, 0), fx=scale, fy=scale)

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frame_mask = cv.inRange(hsv, lower_blue, upper_blue)
        frame_filtered = cv.bitwise_and(frame, frame, mask=frame_mask)

        itr += 1
        itr_image += 1
        itr_deploy += 1

        # -----------------------
        # Template matching
        # -----------------------
        res = cv.matchTemplate(frame_mask, needle_mask, cv.TM_CCOEFF_NORMED)
        
        threshold = 0.425
        h, w = needle_mask.shape[:2]
        
        locations= np.where(res >= threshold)
        locations = list(zip(*locations[::-1]))

        rectangles = []
        for loc in locations:
            rect = [int(loc[0]), int(loc[1]), w, h]
            rectangles.append(rect)
            rectangles.append(rect)

        rectangles, weights = cv.groupRectangles(rectangles, 1, 0.1)

        if len(rectangles):
            line_color = (0,0,255)
            line_type = cv.LINE_4
            line_thickness = 1

            itr_x = 0
            for (x,y,w,h) in rectangles:
                if y < 100 or x > 280:
                    continue
                itr_x += 1

                position = (x + int(w/2),y + int(h/2))
                cv.drawMarker(frame, position, line_color, cv.MARKER_CROSS)

                # Level 
                level_off_set = (-23, -6)
                level_size = (10, 10)
                level_position_emulator = {"left" : position[0] + level_off_set[0], "top" : position[1] + level_off_set[1], "width" : level_size[0], "height" : level_size[1]}
                level_position_screen = {"left" : screen["left"] + position[0] + level_off_set[0], "top" : screen["top"] + position[1] + level_off_set[1], "width" : level_size[0], "height" : level_size[1]}

                # cv.rectangle(frame, top_left, bottom_right, line_color, 1, line_type)
                cv.rectangle(frame, (level_position_emulator["left"], level_position_emulator["top"]), (level_position_emulator["left"] + level_position_emulator["width"], level_position_emulator["top"] + level_position_emulator["height"]), line_color, line_thickness, cv.LINE_4)

                troop_off_set = (int(-w/2), +10)
                troop_size = (w, 40)

                troop_position_emulator = {"left" : position[0] + troop_off_set[0], "top" : position[1] + troop_off_set[1], "width" : troop_size[0], "height" : troop_size[1]}
                troop_position_screen = {"left" : screen["left"] + position[0] + troop_off_set[0], "top" : screen["top"] + position[1] + troop_off_set[1], "width" : troop_size[0], "height" : troop_size[1]}


                cv.rectangle(frame, (troop_position_emulator["left"], troop_position_emulator["top"]), (troop_position_emulator["left"] + troop_position_emulator["width"], troop_position_emulator["top"] + troop_position_emulator["height"]), line_color, line_thickness, cv.LINE_4)

                level = np.array(sct.grab(level_position_screen))
                troop = np.array(sct.grab(troop_position_screen))

                # cv.imwrite(f"data/level/{itr_image}.png", level)
                # cv.imwrite(f"data/troop/{itr_image}.png", troop)

                itr_image += 1
                if itr_image%25 == 0:
                    print(f"saved_level : {itr_image}")
                    print(f"saved_troop : {itr_image}")
                    print(f"saved_shop : {itr_shop}")

            for card in shop:
                itr_shop += 1
                card_img = np.array(sct.grab(card))
                # cv.imwrite(f"data/shop/{itr_shop}.png", card_img)


        # pyautogui.press(random.choice(["1", "2", "3"]))
        time.sleep(random.randint(1, 3)) #prevents spam
        


    else:
        itr_deploy = 0
        play_again_available, _, _ = detect(play_again, frame, True)
        if play_again_available:
            
            pyautogui.click(1050, 650)
            pyautogui.click(1050, 150)
        # time.sleep(7.5)

        print("Running")
        "Not in Phase"
    cv.imshow("Frame", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cv.destroyAllWindows()