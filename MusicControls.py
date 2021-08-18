import cv2
import mediapipe as mp
import time
import numpy as np
from math import dist


from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volumeRange = volume.GetVolumeRange()

from pynput.keyboard import Key, Controller
keyboard = Controller()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
drawing_styles = mp.solutions.drawing_styles

debug = False

fingerExtended = [False,False,False,False,False]
paused = False
lastOpen = None

next = False
previous = False
skipState = []
positions = []

startSetVolume = None
distances = []

# Skip
# When index and middle finger are extended and ring and pinkie are not, this will be referred to as two fingers up
# When two fingers up, keep track of most left that the tip of ring finger is
# If later that tip has moved a certain distance within a certain amount of time, skip
# Can't skip again but can go back to previous track
# Reset when going to previous track or not two fingers up

def pause():
    print("Pause")
    if not debug:
        keyboard.press(Key.media_play_pause)
        keyboard.release(Key.media_play_pause)
    
def nextSong():
    print("Next")
    if not debug:
        keyboard.press(Key.media_next)
        keyboard.release(Key.media_next)
    
def previousSong():
    print("Previous")
    if not debug:
        keyboard.press(Key.media_previous)
        keyboard.release(Key.media_previous)
    
def setVolume(vol):
    print("Set Volume:",round(np.interp(vol, [0.03,0.21], [0,100])))
    if not debug:
        vol = np.interp(vol, [0.03,0.21], [volumeRange[0],volumeRange[1]])
        volume.SetMasterVolumeLevel(vol, None)
    # interface.SetMasterVolume(volume, None)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.6
) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = image[130:, :400]
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        
        if debug:
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if debug:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                fingerExtended[0] = hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y
                fingerExtended[1] = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
                fingerExtended[2] = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
                fingerExtended[3] = hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y
                fingerExtended[4] = hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y
                
                
                if all(fingerExtended[0:2]) and not any (fingerExtended[2:]):
                    if not startSetVolume:
                        startSetVolume = time.time()
                    distance = dist((hand_landmarks.landmark[8].x,hand_landmarks.landmark[8].y),(hand_landmarks.landmark[4].x,hand_landmarks.landmark[4].y))
                    distances.append(distance)
                    if len(distances) > 10:
                        distances.pop(0)
                        
                    if time.time() - startSetVolume > 1.5:
                        setVolume(sum(distances)/len(distances))
                    elif time.time() - startSetVolume > 0.8:
                        print("Volume will be set to",np.interp(sum(distances)/len(distances), [0.03,0.21], [0,100]))
                else:
                    distances = []
                    startSetVolume = None
                
                if all(fingerExtended):
                    paused = False
                    lastOpen = time.time()
                elif not any(fingerExtended[1:]) and not paused and lastOpen and time.time()-lastOpen < 0.08:
                    paused = True
                    pause()
            
                skipState.append(int(all(fingerExtended[1:3]) and (not any (fingerExtended[3:]))))
                if len(skipState) > 15:
                    skipState.pop(0)
                    
                    
                if sum(skipState)/len(skipState) > 0.6:
                    positions.append(hand_landmarks.landmark[8].x)
                    if len(positions) > 20:
                        positions.pop(0)
                        
                    if len(positions) > 0:
                        if sum(positions)/len(positions) - hand_landmarks.landmark[8].x > 0.25 and not previous:
                            previousSong()
                            previous = True
                            next = False
                            positions = []
                        elif sum(positions)/len(positions) - hand_landmarks.landmark[8].x < -0.25 and not next:
                            nextSong()
                            next = True
                            previous = False
                            positions = []
                        
                else:
                    next = False
                    previous = False
                    positions = []
                
                    
            

        if debug:
            cv2.imshow("MediaPipe Hands", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
