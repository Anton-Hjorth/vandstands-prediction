import pandas as pd
import zipfile
from playwright.sync_api import sync_playwright

def fetch_latest_data(playwright, indre_url):
    browser = playwright.chromium.launch(headless=True)  # Set headless=True if you don't need a browser UI
    context = browser.new_context()

    # Open new page
    page = context.new_page()

    # Navigate to the page
    page.goto(indre_url)

    with page.expect_download() as download_info:
    # Click the download button
        page.click("a#txtlink.btn.btn-primary.btn-sm.text-white.plotButton")  # Adjust the selector to match the download button

    download = download_info.value
    download_path = download.path()
    download.save_as("downloaded_file.zip")


    with zipfile.ZipFile("downloaded_file.zip", 'r') as zip_ref:
        zip_ref.extractall("CSV-Data")  # Extract all files to the 'extracted_files' directory

    browser.close()


indre_url = 'https://vandportalen.dk/plotsmaps?config=default&days=1&z=17&x=8.6798&y=55.3402&loc=212&id=620-0-0'
ydre_url = 'https://vandportalen.dk/plotsmaps?config=default&days=1&z=17&x=8.6798&y=55.3402&loc=2154&id=43948-0-0'


def import_csv(file_path):
    # Find the first row with data
    with open(file_path, 'r') as file:
        lines = file.readlines()
        valid_start = next(i for i, line in enumerate(lines) if line.strip() and line.strip()[0].isdigit())

    # Read the CSV with pandas starting from line that contains the first data entry
    data_frame = pd.read_csv(file_path, sep=";", skiprows=valid_start, usecols=[0, 1], names=["Timestamp", "Water Level"])
    return data_frame

def merge_latest_with_existing(path_to_read, path_to_write):
    with open(path_to_read,'r') as csv_file:
        lines = csv_file.readlines()
        with open(path_to_write,'a') as file_to_write:
            for row in lines:
                #print(row[0].isdigit())
                if row[0].isdigit():

                    file_to_write.write(row)
                #print(row)

    print('')




indre_vandstande = import_csv("CSV-Data/Indre_Vandstand - fra 2019.csv")
ydre_vandstande = import_csv("CSV-Data/Ydre_Vandstand - fra 2019.csv")

#with sync_playwright() as playwright:
#    fetch_latest_data(playwright, indre_url)
#    fetch_latest_data(playwright, ydre_url)

#merge_latest_with_existing('CSV-Data/38.05_Vandstand_Minut.csv','CSV-Data/Indre_Vandstand - fra 2019.csv') #læs ydre, og skriv til ydre
#merge_latest_with_existing('CSV-Data/9006701_Vandstand (Kystdirektoratet / Coastal Authority)_Minut.csv','CSV-Data/Ydre_Vandstand - fra 2019.csv') #læs indre, og skriv til indre