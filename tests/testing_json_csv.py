# Reddit Scraper w/ BeautifulSoup

# Import all libraries
# from re import sub
# import re
from bs4 import BeautifulSoup
import requests
import json
import csv
import time

# Function -> Main Scraper to get content from Reddit
def scrape_reddit() -> list[dict]:
    subreddits = [
        "https://old.reddit.com/r/datascience/"
    ]
    
    all_data = []
    
    for url in subreddits:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        subreddit_name = url.split("/")[:-1][-1]
        print(f"Scraping for: {subreddit_name}")
        
        try: 
            respone = requests.get(url.strip(), headers=headers, timeout=10)
            respone.raise_for_status()
            
            soup = BeautifulSoup(respone.content, "html.parser")
            
            subreddit_data = {
                "subreddit_name": subreddit_name,
                "url": url,
                "title": soup.title.string if soup.title else "No title",
                "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        
            topics = []
            keywords = ["python", "job", "tutorial", "project", "advice"]
            
            for post in soup.find_all("div", class_="thing"):
                title_elem = post.find("a", class_="title")
                if not title_elem:
                    continue
                text = title_elem.get_text(strip=True)
                
                if text and len(text) > 3:
                    if any(keyword in text.lower() for keyword in keywords):
                        topics.append({
                            "title": text,
                            "type": "datascience_topic"
                        })
                        
            discussions = []
            seen_urls = set()
            
            for post in soup.find_all("div", class_="thing"):
                title_elem = post.find("a", class_="title")
                if not title_elem:
                    continue
                text = title_elem.get_text(strip=True)
                href = title_elem.get("href", "")
                
                if href.startswith("/"):
                    full_url = "https://old.reddit.com" + href
                
                if text and len(text) > 1 and "/comments/" in href and href not in seen_urls:
                    seen_urls.add(href)
                    
                    discussions.append({
                        "title": text[:100] + "..." if len(text) > 100 else text,
                        "url": href,
                        "type": "discussion"
                    })
                    
            subreddit_data["datascience_topics"] = topics
            subreddit_data["discussions"] = discussions
            
            all_data.append(subreddit_data)
            time.sleep(2)
        except Exception as e:
            print(f"ERROR: {e}")
            
    return all_data
    
# Function -> Saved scraped data (both json and csv)
def save_scraped_data(data, filename_json="datascience_topics.json", filename_csv="datascience_topics.csv"):
    if not data:
        print("No data!")
        return
    
    # JSON
    try:
        with open(filename_json, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=True)
        print(f"Data Saved: {filename_json}")
    except Exception as e:
        print(f"JSON ERROR: {e}")
        
    # CSV
    try:
        with open(filename_csv, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Subreddit", "Type", "Title", "URL", "Scraped_at"])
            
            for subreddit_data in data:
                subreddit = subreddit_data["subreddit_name"]
                scraped_at = subreddit_data["scraped_at"]
                
                for topic in subreddit_data.get("datascience_topics", []):
                    writer.writerow([subreddit, topic["type"], topic["title"], "", scraped_at])
                    
                for discussion in subreddit_data.get("discussions", []):
                    writer.writerow([subreddit, discussion["type"], discussion["title"], discussion["url"], scraped_at])
            
        print(f"Data Saved: {filename_csv}")
    except Exception as e:
        print(f"CSV ERROR: {e}")         
        
# Main -> Execute the code above
def main() -> None:
    data = scrape_reddit()
    
    if data:
        print(f"Processing the data...")
        total_topics = 0
        total_discussions = 0
        
        for subreddit_data in data:
            topics_count = len(subreddit_data.get("datascience_topics", []))
            discussions_count = len(subreddit_data.get("discussions", []))
            
            total_topics += topics_count
            total_discussions += discussions_count
            
        print(f"\nTotal: {topics_count} Data Science Topics, {discussions_count} Discussions")
        save_scraped_data(data)
    else:
        print("There is not data returned!") 

if __name__ == "__main__":
    main()