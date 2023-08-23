import os
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
import time
import asyncio

SEASONS = list(range(2019, 2024))
DATA_DIR = "data"
STANDINGS_DIR = os.path.join(DATA_DIR, "standings")
SCORES_DIR = os.path.join(DATA_DIR, "scores")
CUR_SEASON = 2024

async def get_html(url, selector, sleep=5, retries=10):
    html = None
    for i in range(1, retries+1):
        time.sleep(sleep * i)
        try:
            async with async_playwright() as p: # Create an async instance of the playwright class
                browser = await p.chromium.launch() # Initialize browser (chrome)
                page = await browser.new_page() # Create tab in browser
                await page.goto(url) # Send tab to url
                html = await page.inner_html(selector) # Grab a piece of html determined by selector
                print(await page.title()) # Print page title
        except PlaywrightTimeout:
            print(f"Timeout error on {url}")
            continue
        else:
            break
    return html

async def scrape_season(season):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    html = await get_html(url, "#content .filter") # Get html from URL with id='content' with class filter (returns <a> for each month in season)
    if not html:
        print(f"Could not obtain html for {url}")
        return
    soup = BeautifulSoup(html) # Allows us to do stuff with our HTML
    links = soup.find_all("a") # Find all <a> in our HTML
    standings_pages = [f"https://www.basketball-reference.com{l['href']}" for l in links] # Get links for each <a> => link for each month in season
    
    for url in standings_pages:
        save_path = os.path.join(STANDINGS_DIR, url.split("/")[-1]) # Path to which we will save url for each month in season => standings dir path + 
                                                                    # end of url (https://www.basketball-reference.com/leagues/NBA_2018_games-june.html) end of this guy
        if os.path.exists(save_path):
            continue
        
        html = await get_html(url, "#all_schedule") # Get html for table containing box scores for every game in a month
        if not html:
            print(f"Could not obtain html for {url}")
            continue
        with open(save_path, "w+") as f:
            f.write(html)

async def scrape_games(monthly_schedule_file):
    with open(monthly_schedule_file, 'r') as f:
        html = f.read()

    soup = BeautifulSoup(html, features="lxml")
    links = soup.find_all("a")
    hrefs = [l.get('href') for l in links]
    box_scores = [f"https://www.basketball-reference.com{l}" for l in hrefs if l and "boxscore" in l and '.html' in l]  # All boxscores from monthly schedule 

    for url in box_scores:
        save_path = os.path.join(SCORES_DIR, url.split("/")[-1])
        if os.path.exists(save_path): 
            continue
        print(f"scraping {url} !!")
        html = await get_html(url, "#content")
        if not html:
            print(f"Could not obtain html for {url}")
            continue
        with open(save_path, "w+", encoding='utf-8') as f:
            f.write(html)

async def scrape_all():
    for season in SEASONS:
        scrape_season(season)

    standings_files = os.listdir(STANDINGS_DIR)
    monthly_schedule_file = [s for s in standings_files]
    
    for f in monthly_schedule_file:
        filepath = os.path.join(STANDINGS_DIR, f)
        await scrape_games(filepath)

async def scrape_cur_season(cur_season):
    scrape_season(cur_season)

    standings_files = os.listdir(STANDINGS_DIR)
    monthly_schedule_files = [s for s in standings_files if str(cur_season) in s]

    for f in monthly_schedule_files:
        filepath = os.path.join(STANDINGS_DIR, f)
        await scrape_games(filepath)

asyncio.run(scrape_cur_season(CUR_SEASON))