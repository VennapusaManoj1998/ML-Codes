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
print(soup.prettify())

# Find all <a> tags with .xlsx file links and filter based on date
xlsx_links = []
for link in soup.find_all("a", href=True):
    print(link)
    href = link["href"]
    #print(href)
    
  
xlsx_links = []
for li in soup.find_all("li"):
    a_tag = li.find("a")
    if a_tag and "GetYearMonth" in a_tag.get("onclick", ""):
        year = a_tag["id"][:4]
        month = a_tag["onclick"].split('"')[3]
        if year == "2022" and int(month) >= 4 or year == "2023" and int(month) <= 3 and int(month) != 0:
            xlsx_links.append(a_tag)
        
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
