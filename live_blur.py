import numpy as np 
import cv2
import scipy
import scipy.ndimage
from scipy import signal

PATH_XML ='HaarCascade_FrontalFace_Default.xml'  
# Kernel
def generate_kernel(kernel_len=30, std_deviation=30):
    generate_kernel1d = signal.windows.gaussian(kernel_len, std_deviation)
    generate_kernel2d = np.outer(generate_kernel1d, generate_kernel1d)
    return generate_kernel2d
 
kernel = generate_kernel()
kernel_tile = np.tile(kernel, (3, 1, 1))
kernel_sum = kernel.sum()
kernel = kernel/kernel_sum

# Webcam Video Capture
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Failed to access WebCam!")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + PATH_XML)
if face_cascade.empty():
    print("Error: Failed to load XML!")
    exit()

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Failed to capture WebCam video frame!")
        break
    
    # Apply convolution process
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
         gray,
         scaleFactor=1.1, 
         minNeighbors=5,
         minSize=(30,30),
         flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    for x, y, w, h in faces:
         frame[y:y+h, x:x+w] = scipy.ndimage.convolve(frame[y:y+h, x:x+w], np.atleast_3d(kernel), mode='nearest')
         
         cv2.imshow('WebCam - Live Blur', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video.release()
cv2.destroyAllWindows() 