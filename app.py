import cv2
import imutils
import pytesseract
from PIL import Image
import streamlit as st
import numpy as np

st.title('Licence plate number detection')

def load_image(image_file):
    img = Image.open(image_file)
    return img

def main():
    images = st.file_uploader("Choose a file")
    #st.image(load_image(images))
    #Function to read the image  
    #image = cv2.imread(image) 
    # Function to resize the image  
    if images is not None:
        file_bytes = np.asarray(bytearray(images.read()), dtype=np.uint8)
        image=cv2.imdecode(file_bytes,1)
    
    image = imutils.resize(image,width=300)
    #Convert BGR image to GRAYSCALE
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #Canny edge detection to detect edges in an image 
    edged = cv2.Canny(gray_image, 30, 200)
    # Function to join similar edges 
    #Take two arguments : 
    #First:: It take all the contours but doesn't create parent-child relationship
    #Second:: It specify to take corner points off the counter
    cnts,new = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)   
    image1=image.copy()
    #This function is used to create counters based to given counter cordinates
    cv2.drawContours(image1,cnts,-1,(0,255,0),3)
    #Here we are sorting the conters and neglecting all the conters whose area is less than 20
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:20]
    screenCnt = None
    image2 = image.copy()
    for c in cnts:
        #To determine sqaure cure among the identified contours
        perimeter = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
        if len(approx) == 4: 
            screenCnt = approx
            x,y,w,h = cv2.boundingRect(c) 
            new_img=image[y:y+h,x:x+w]
            #PyTesseract to convert text in the image to string
            plate = pytesseract.image_to_string(new_img)
            if(plate==''):
                plate="Error"
            if(len(plate)>10): #Car number plate can not have more than 10 characters 
                st.write("Number plate is:", plate[:13])  
            else:
                st.write("Number plate is:", plate)  
            break
    cv2.drawContours(image,[screenCnt],-1,(0, 255, 0), 3)   
        
if __name__ == '__main__':
    main()