# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 10:04:18 2023

@author: vennapusa manoj
"""

import requests
from bs4 import BeautifulSoup
import os

# URL of the web page containing the links
url = "https://example.com/page"

# Folder path to save the downloaded files
folder_path = r"C:\path\to\folder"

# Send a GET request to the URL
response = requests.get(url)

# Create a BeautifulSoup object to parse the HTML content
soup = BeautifulSoup(response.content, "html.parser")

# Find all <a> tags with the specified href structure
xlsx_links = []
for link in soup.find_all("a", href=True):
    href = link["href"]
    if href.startswith("#") and "GetYearMonth" in link.get("onclick", ""):
        year = href.split('"')[1]
        month = href.split('"')[3]
        xlsx_links.append((year, month))


xlsx_links = []
for li in soup.find_all("li"):
    a_tag = li.find("a")
    if a_tag and "GetYearMonth" in a_tag.get("onclick", ""):
        year = a_tag["id"][:4]
        month = a_tag["onclick"].split('"')[3]
        xlsx_links.append((year, month))
        
        
# Create the folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Download each .xlsx file to the folder
for year, month in xlsx_links:
    download_url = f"https://example.com/files/{year}/{month}.xlsx"
    file_name = f"{year}{month}.xlsx"
    file_path = os.path.join(folder_path, file_name)

    # Send a GET request to download the file
    file_response = requests.get(download_url)

    # Save the file to the specified folder
    with open(file_path, "wb") as file:
        file.write(file_response.content)

    print(f"Downloaded: {file_name}")

print("All .xlsx files downloaded successfully!")
