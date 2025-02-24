# gesture based volume and brightness control

this project allows users to control their system volume and screen brightness using hand gestures. 
it utilizes opencv and mediapipe for hand tracking, and adjusts brightness and volume based on the distance between specific hand landmarks.

## features
(automatically takes 1st hand as left and 2nd as right, if u require to have left as left for sure use handeness)
- adjust system volume using right-hand gestures
- adjust screen brightness using left-hand gestures
- real-time processing using mediapipe's hand tracking
- simple and efficient implementation

## installation

1. clone the repository:
   ```sh
   git clone https://github.com/yourrepo/gesture-control.git
   ```
2. navigate to the project directory:
   ```sh
   cd gesture-control
   ```
3. install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## usage

1. run the script:
   ```sh
   python main.py
   ```
2. use the left hand to control brightness and the right hand to control volume.
3. close the application by pressing 'q'.

## requirements

- python 3.7 or higher
- opencv
- mediapipe
- numpy
- screen-brightness-control
- pycaw
- comtypes

## license

this project is open-source and available under the mit license.
