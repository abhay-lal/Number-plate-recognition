import cv2
import imutils
import pytesseract
from PIL import Image
import streamlit as st
import numpy as np
pytesseract.pytesseract.tesseract_cmd='\usr\local\lib\python3.7\site-packages\pytesseract\pytesseract.py'
#st.title('Licence plate number detection')
st.markdown("<h1 style='text-align: center; color: white;'>Licence plate number detection</h1>", unsafe_allow_html=True)
base="light"

def load_image(image_file):
    img = Image.open(image_file)
    return img

def words():
    st.markdown('In this Deep Learning Application, we have used OpenCv to detect the Licence plate in an image and give the converted string.\
    We have used the Pytesseract Library to perform OCR on the image.')
    
def main():
    st.markdown("![gif](https://cdn.discordapp.com/attachments/945603582462398464/948294399689912330/car-on-the-road-4851957-404227-unscreen.gif)")
    #st.subheader('Choose a photo')
    images = st.file_uploader('',type=['jpeg','png', 'jpg'])
    #st.image(load_image(images))
    #Function to read the image  
    #image = cv2.imread(image) 
    # Function to resize the image  
    if images is not None:
        file_bytes = np.asarray(bytearray(images.read()), dtype=np.uint8)
        image=cv2.imdecode(file_bytes,1)
        image = imutils.resize(image,height=620,width=480)
        #Convert BGR image to GRAYSCALE
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #Canny edge detection to detect edges in an image 
        edged=cv2.Canny(gray_image,30,200)
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
                invert = cv2.bitwise_not(new_img)
                gray_image1= cv2.cvtColor(invert,cv2.COLOR_BGR2GRAY)
                gray_image1=cv2.equalizeHist(gray_image1)
                threshold=cv2.threshold(gray_image1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                #rem_noise=cv2.medianBlur(threshold,5)
                #PyTesseract to convert text in the image to string
                plate = pytesseract.image_to_string(threshold, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8')
                if(plate==''):
                    plate="Error"
                if(len(plate)>10): #Car number plate can not have more than 10 characters 
                    st.subheader("Number plate is : "+plate[:13])  
                else:
                    st.subheader("Number plate is : "+plate)  
                break
        cv2.drawContours(image,[screenCnt],-1,(0,255,0),2) 
        stack=[gray_image,edged]
        stack1=[invert,gray_image1,threshold]
        col1, col2= st.columns(2)
        with col1:
            st.image(gray_image,caption='Gray-scale image',width=300)
        with col2:
            st.image(edged,caption='Canny Edge',width=300)
        
        col1, col2, col3= st.columns(3)
        with col1:
            st.image(invert,caption='Inverted B & W',width=150)
        with col2:
            st.image(gray_image1,caption='Grayscale Image',width=150)
        with col3:
            st.image(threshold,caption='Threshold Image',width=150)    
        
        words()
        
        
if __name__ == '__main__':
    main()


