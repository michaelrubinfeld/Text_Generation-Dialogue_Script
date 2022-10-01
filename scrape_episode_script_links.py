from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
import time
from csv import writer
from typing import List, Union


DRIVER_PATH = "C:\Program Files (x86)\chromedriver.exe"
s = Service(DRIVER_PATH)
driver = webdriver.Chrome(service=s)
base_url = 'https://transcripts.foreverdreaming.org/viewforum.php?f=177'
season_suffixes = {1: '',
                   2: '&start=25',
                   3: '&start=50',
                   4: '&start=75',
                   5: '&start=100',
                   6: '&start=125',
                   7: '&start=150',
                   8: '&start=175',
                   9: '&start=200'}


def append_list_as_row(file_name: str, list_of_elem: List[Union[str, int, float]]):
    try:
        with open(file_name, 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(list_of_elem)
    except:
        pass


def main():
    for curr_season in range(1, 10):
        # go to transcript website
        driver.get(base_url + season_suffixes[curr_season])
        print(f'Started scraping links from season {curr_season}')
        time.sleep(10)

        elems = WebDriverWait(driver, 5).until(
            EC.presence_of_all_elements_located((By.XPATH, "//a[contains(@class, 'topictitle')]")))
        time.sleep(3)
        for count, elem in enumerate(elems, 1):
            if count > 1:   # first link is not an episode
                append_list_as_row(f'episode_links_by_season/season{curr_season}.txt',
                                   [elem.get_attribute("href")])
        print(f'Finished scraping links from season {curr_season}')


if __name__ == "__main__":
    main()
    driver.quit()


