import math
import numpy as np
import cv2

import glob
import os
from ctypes import cdll
from pathlib import Path
#import zbar

#import pyzbar


cdll.LoadLibrary(str(Path('').joinpath('libiconv.dll')))

cdll.LoadLibrary(str(Path(__file__).parent.joinpath('libzbar-32.dll')))



from pyzbar.pyzbar import decode
from PIL import Image





inputfolder='..//Inputs'
outputfolder='..//Outputs'

names = [os.path.basename(x) for x in glob.glob(inputfolder+'/*.jpg')]
for fno in range(0,len(names)):
    
    print("Processing for "+names[fno])
    
    img = cv2.imread(inputfolder+'/'+names[fno])

    # rorate image if its row is greater than column
    if img.shape[0]>img.shape[1]:
        img=cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    decode(Image.open(inputfolder+'/'+names[fno]))