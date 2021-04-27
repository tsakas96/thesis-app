import pandas as pd
from PIL import Image
import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore, storage
from streamlit_drawable_canvas import st_canvas
import SessionState
import time
import numpy as np
import io
import datetime
import skimage.io
import os

def rerun():
    raise st.script_runner.RerunException(st.script_request_queue.RerunData(None))

def initializeFirebase():
    firebaseConfig = {
    'apiKey': st.secrets["apiKey"],
    'authDomain': st.secrets["authDomain"],
    'projectId': st.secrets["projectId"],
    'storageBucket': st.secrets["storageBucket"],
    'messagingSenderId': st.secrets["messagingSenderId"],
    'appId': st.secrets["appId"],
    'measurementId': st.secrets["measurementId"],
    'databaseURL': st.secrets["databaseURL"]}

    firebase = pyrebase.initialize_app(firebaseConfig)
    return firebase

def update_counter(count, userId):
    data = {
        u'count': count,
    }
    # Add a new doc in collection 'users' with ID 'userId'
    db = firestore.client()
    db.collection(u'users').document(userId).set(data)

def signup(email, password, auth):
    try:
        user = auth.create_user_with_email_and_password(email, password)
        update_counter(0, user['localId'])
        
        st.sidebar.success("Successfully created account!")
    except:
         st.sidebar.error("Something went wrong!")

def signin(email, password, auth, session_state):
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        st.sidebar.success("Successfully logged in!")
        session_state.user_logged_in = True
        session_state.user_id = user['localId']
        db = firestore.client()
        doc_ref = db.collection(u'users').document(user['localId']).get()
        count = doc_ref.to_dict()['count']
        session_state.history = count
    except:
        st.sidebar.error("Invalid email or password!")

def display_timer_and_icon(session_state, icon_path):
    left, right = st.beta_columns([.1,1])
    with left:
        st.markdown("Timer:")
    with right:
        timer_text = st.empty()
    start_time = time.time()
    seconds = 0
    
    icon = st.empty()
    timer_text.markdown(str(5 - seconds))
    while seconds < 5:
        if time.time() - start_time >= 1:
            start_time = time.time()
            seconds = seconds + 1
            timer_text.markdown(str(5 - seconds))
        icon.image(download_image(icon_path))
        

def upload_image(canvas_result,image_path):
    sketch_arr = canvas_result.image_data[:,:,0:3].astype(np.uint8)
    sketch = Image.fromarray(sketch_arr)
    sketch = sketch.resize(size=(256, 256))

    img_byte_arr = io.BytesIO()
    sketch.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    bucket = storage.bucket()
    blob = bucket.blob(image_path)
    blob.upload_from_string(
            img_byte_arr,
            content_type='image/png'
        )


def download_image(image_path):
    bucket = storage.bucket()
    blob = bucket.blob(image_path)
    image_url = blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')
    image_numpy = skimage.io.imread(image_url)
    return image_numpy

def main():
    if not firebase_admin._apps:
        firebaseConfig = {
            "type": st.secrets["type"],
            "project_id": st.secrets["project_id"],
            "private_key_id": st.secrets["private_key_id"],
            "private_key": st.secrets["private_key"],
            "client_email": st.secrets["client_email"],
            "client_id": st.secrets["client_id"],
            "auth_uri": st.secrets["auth_uri"],
            "token_uri": st.secrets["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["client_x509_cert_url"]}
        cred = credentials.Certificate(firebaseConfig)
        firebase_admin.initialize_app(cred, {'databaseURL': st.secrets["databaseURL"],'storageBucket': st.secrets["storageBucket"]})
    firebase = initializeFirebase()
    auth = firebase.auth()

    menu = ["Home", "Login", "Signup"]
    choice = st.sidebar.selectbox("Menu", menu)
    session_state = SessionState.get(user_logged_in = False, user_id = "", state = "Start", history=0)
    #icon_info = np.load("./Scripts/icon_info1.npy", allow_pickle=True)
    icon_info = np.load("icon_info1.npy", allow_pickle=True)

    st.title("Draw the sketch!")
    st.markdown(' ')

    if choice == "Home":
        if session_state.user_logged_in:
            if session_state.state == "Start":
                if session_state.history >= len(icon_info):
                    session_state.state = "End"
                    rerun()
                else:
                    start_button = st.button("Start")
                    if start_button:
                        session_state.state = "DisplayIcon"
                        rerun()
            elif session_state.state == "DisplayIcon":
                icon_path = "icon/" + icon_info[session_state.history][1] + "/" + icon_info[session_state.history][0]
                st.markdown("Be careful! You do not have a second chance.")
                display_timer_and_icon(session_state, icon_path)
                
                session_state.state = "Draw"
                rerun()
            elif session_state.state == "Draw":
                left, right = st.beta_columns([2,1])
                with left:
                    st.markdown("Draw what you have seen!")
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                        stroke_width=1,
                        stroke_color="rgba(0, 0, 0, 1)",
                        background_color="rgba(255, 255, 255, 1)",
                        height=448,
                        width=448,
                        drawing_mode="freedraw",
                        key="canvas",
                    )
                    if st.button("Save/Next"):
                        is_filled = bool(np.sum(canvas_result.image_data - 255))
                        if is_filled:
                            path_sketch = "sketch/" + session_state.user_id + "/" + icon_info[session_state.history][1] + "/" + icon_info[session_state.history][0].replace(".jpg",".png")
                            upload_image(canvas_result, path_sketch)
                            session_state.history = session_state.history + 1
                            update_counter(session_state.history, session_state.user_id)
                            if session_state.history >= len(icon_info):
                                session_state.state = "End"
                            else:
                                session_state.state = "DisplayIcon"
                            rerun()
                        else:
                            st.error("You have to draw something!")
                with right:
                    st.text(str(session_state.history+1)  + "/" + str(len(icon_info))) 
            elif session_state.state == "End":
                st.text("Thank you for your time!")
                st.balloons()
        else:
            st.warning("You have to sign in first!")

    elif choice == "Login":
        st.sidebar.subheader("Login Section")
        st.warning("Signup if you don't have an account!")

        email = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password", type='password')
        
        if st.sidebar.button("Login"):
            signin(email, password, auth, session_state)

    elif choice == "Signup":
        st.sidebar.subheader("Create New Account")
        new_user_email = st.sidebar.text_input("User Email")
        new_password = st.sidebar.text_input("Password", type='password')
        
        if st.sidebar.button("Signup"):
            signup(new_user_email, new_password, auth)

            
if __name__ == '__main__':
    main()