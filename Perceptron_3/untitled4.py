# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 11:41:19 2023

@author: vennapusa manoj
"""

import requests
from bs4 import BeautifulSoup
import os

# URL of the web page containing the .xlsx files
url = "https://www.rbi.org.in/Scripts/ATMView.aspx"

# Folder path to save the downloaded files
folder_path = r"C:\Users\venna\OneDrive\Desktop\Machine Learning\WebScrapping"

# Send a GET request to the URL
response = requests.get(url)

# Create a BeautifulSoup object to parse the HTML content
soup = BeautifulSoup(response.content, "html.parser")

# Find all <a> tags with .xlsx file links
xlsx_links = soup.find_all("a", href=lambda href: href and href.endswith(".XLSX"))
print(xlsx_links)


# Create the folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Download each .xlsx file to the folder
for link in xlsx_links:
    file_url = link["href"]
    file_name = os.path.basename(file_url)
    file_path = os.path.join(folder_path, file_name)

    # Send a GET request to download the file
    file_response = requests.get(file_url)

    # Save the file to the specified folder
    with open(file_path, "wb") as file:
        file.write(file_response.content)

    print(f"Downloaded: {file_name}")

print("All .xlsx files downloaded successfully!")