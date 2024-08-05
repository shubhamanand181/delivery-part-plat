import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from ortools.linear_solver import pywraplp
from dotenv import load_dotenv
import streamlit as st

# Load .env file
load_dotenv()

# Get the Google Maps API key
google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")

# Sidebar for navigation
st.sidebar.title("Delivery Partner Dashboard")
selection = st.sidebar.selectbox("Select Feature", ["Login", "Navigation", "GPS Tracking", "Route Optimization", "Data Analytics", "Interactive Maps"])

if selection == "Login":
    st.title("Delivery Partner Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        st.write("Logged in as", username)
    if st.button("Register"):
        st.write("Registered successfully")

elif selection == "Navigation":
    st.title("Navigation")
    # Add navigation features here

elif selection == "GPS Tracking":
    st.title("GPS Tracking")
    # Add GPS tracking features here

elif selection == "Route Optimization":
    st.title("Route Optimization")
    # Add route optimization features here

elif selection == "Data Analytics":
    st.title("Data Analytics")
    # Add data analytics features here

elif selection == "Interactive Maps":
    st.title("Interactive Maps")
    # Add interactive maps features here

