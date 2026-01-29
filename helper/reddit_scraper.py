from bs4 import BeautifulSoup
import requests
import pandas as pd
import time

BASE_URL = "https://old.reddit.com/r/datascience/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
}

# Scrape post metadata
def scrape_listing(pages=3):
    url = "https://old.reddit.com/r/datascience/"
    posts = []

    for _ in range(pages):
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        for post in soup.find_all("div", class_="thing"):
            title_tag = post.find("a", class_="title")
            time_tag = post.find("time")
            comments_tag = post.find("a", class_="comments")

            if not title_tag:
                continue

            post_url = title_tag["href"]
            if post_url.startswith("/"):
                post_url = "https://old.reddit.com" + post_url
                
            if not post_url.startswith("https://old.reddit.com/r/datascience/"):
               continue

            comment_count = 0
            if comments_tag:
                parts = comments_tag.text.split()
                if parts and parts[0].isdigit():
                    comment_count = int(parts[0])
            
            if "alb.reddit.com" in post_url:    # Skip ads / promoted links
                continue

            posts.append({
                "url": post_url,
                "post_id": post.get("data-fullname"),
                "post_title": title_tag.text.strip(),
                "author": post.get("data-author"),
                "score": int(post.get("data-score") or 0),
                "comment_count": comment_count,
                "created_utc": (
                    time_tag["datetime"] if time_tag and time_tag.has_attr("datetime") else None
                )
            })

        next_btn = soup.find("span", class_="next-button")
        if not next_btn:
            break

        url = next_btn.a["href"]
        time.sleep(2)

    return posts

# Scrape post body
def scrape_body(post_url):
    try:
        response = requests.get(post_url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Main post container
        post_div = soup.find("div", class_="thing", attrs={"data-type": "link"})
        if not post_div:
            return ""

        # Text body (self posts)
        body_div = post_div.find("div", class_="md")
        if not body_div:
            return ""

        return body_div.get_text(strip=True)

    except Exception as e:
        print(f"Failed to scrape body: {post_url}")
        return ""

# Combine everything
def scrape_reddit_full(pages=3):
    listing_posts = scrape_listing(pages)
    full_data = []

    for i, post in enumerate(listing_posts):
        print(f"Scraping body {i+1}/{len(listing_posts)}")
        body = scrape_body(post["url"])

        full_data.append({
            "url": post["url"],
            "post_id": post["post_id"],
            "post_title": post["post_title"],
            "author": post["author"],
            "score": post["score"],
            "comment_count": post["comment_count"],
            "created_utc": post["created_utc"],
            "body": body
        })

        time.sleep(1)

    return pd.DataFrame(full_data)


def main():
    df = scrape_reddit_full(pages=7)

    if df.empty:
        print("No data scraped! Check selectors or request blocked.")
        return

    print("COLUMNS:", df.columns)
    print(df.head())
    
    df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce")
    df.dropna(subset=["post_title"], inplace=True)
    
    if "created_utc" in df.columns:
        df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce")

    df.to_csv("data/reddit_datascience.csv", index=False)
    print(f"Saved {len(df)} rows to data/reddit_datascience.csv")

if __name__ == "__main__":
    main()
