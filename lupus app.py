import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('C:/Users/rpuni/OneDrive/Documents/STREAMLIT/lupus nephritis.h5')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Information","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("LUPUS NEPHRITIS STAGE DETECTION")
    image_path = "C:/Users/rpuni/OneDrive/Documents/STREAMLIT/lupus nephritis wallpaper.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Lupus Nephritis Disease Recognition System! 🔍
    ###Introduction       
    According to the ISN/RPS Classification report:
    
            Lupus nephritis Disease is six stages:-
    
                Class 1: Minimal mesangial lupus nephritis
    
                Class 2: Mesangial proliferative lupus nephritis.
    
                Class 3: Focal lupus nephritis.
    
                Class 4: Diffuse lupus nephritis.
    
                Class 5: Membranous lupus nephritis.
    
                Class 6: Advanced sclerosis lupus nephritis.

    **Our mission is to help in identifying which stage of lupus nephritis disease is affected to patient efficiently. Upload an image of a renal biopsy/glomeruli, and our system will analyze it to detect any signs of diseases.**

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a renal biopsy/glomeruli with suspected diseases.
    2. **Analysis:** Our system will process the image using convolutional neural network model  to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system uses Convolutional Neural Network model that is deep learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the efficiency of our Lupus Nephritis  Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset

                This dataset is recreated from the dataset uploaded in mendley.com also obtained few image from  various browser sites .
                This dataset consists of about 2.5k rgb images of healthy and diseased renal biospsy images which is categorized into 5 different classes.
                The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                
                **Size-2.1GB**

                **Directory consists of 5 classes**

                    1.Focal lupus nephritis(class3) -100 files
                    2.Diffuse lupus nephritis(class4)-100 files
                    3.Membranous lupus nephritis(class5)-100 files
                    4.Advanced sclerosis lupus nephritis(class6)-1143files
                    5.lupus nephritis normal -1143 files

                #### Content
                1. train (2586 images)
                2. test (259 images)
                3. validation (259 images)

                """)
#Disease Information
elif(app_mode=="Disease Information"):
    st.header("Frequently asked questions to doctors")
    st.markdown("""
                ####About Disease

                What is lupus nephritis?

                Lupus nephritis is a type of kidney disease caused by systemic lupus erythematosus  (SLE or lupus). 
                Lupus is an autoimmune disease a disorder in  which the body’s immune system attacks the body’s own cells and organs. Kidney disease caused by lupus may get worse over time and lead to kidney failure. 
                If your kidneys fail, you will need dialysis or a kidney transplant to maintain your health.

                Who gets lupus?

                Lupus is much more common in women than in men and most often strikes during the child-bearing years. 
                Nine out of 10 people who have lupus are women.  According to our study lupus is also more common in people of African or Asian background. 

                How common is lupus nephritis?

                Kidney damage is one of the more common health problems caused by lupus. 
                In adults who have lupus, as many as 5 out of 10 will have kidney disease. In children who have lupus, 8 of 10 will have kidney disease.

                What are the symptoms of lupus nephritis?

                The symptoms of lupus nephritis may include foamy urine and edema—swelling that occurs when your body has too much fluid, usually in the legs, feet, or ankles, and less often in the hands or face. You may also develop high blood pressure.
                Kidney problems often start at the same time or shortly after lupus symptoms and can include

                   •	joint pain or swelling
                   •	muscle pain
                   •	fever with no known cause
                   •	a red rash, often on the face, across the nose and cheeks, sometimes called a butterfly rash because of its shape.

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Focal lupus nephritis(class3)','Diffuse lupus nephritis(class4)','Membranous lupus nephritis(class5)','Advanced sclerosis lupus nephritis(class6)','lupus nephritis normal']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
