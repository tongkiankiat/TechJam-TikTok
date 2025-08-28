"""
This example run script shows how to run the yelp.com scraper defined in ./yelp.py
It scrapes product data and saves it to ./results/

To run this script set the env variable $SCRAPFLY_KEY with your scrapfly API key:
$ export $SCRAPFLY_KEY="your key from https://scrapfly.io/dashboard"
"""

import asyncio
import json
from pathlib import Path
import yelp

output = Path(__file__).parent / "results"
output.mkdir(exist_ok=True)


async def run():
    # enable scrapfly cache for basic use
    yelp.BASE_CONFIG["cache"] = False

    print("running Yelp scrape and saving results to ./results directory")

    reviews_data = await yelp.scrape_reviews(
        url="https://www.yelp.com/biz/vons-1000-spirits-seattle-4",
        # each 10 reviews represent a review page (one request)
        max_reviews=28,
    )
    with open(output.joinpath("reviews.json"), "w", encoding="utf-8") as file:
        json.dump(reviews_data, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(run())
