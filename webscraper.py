# flipkart_scraper.py
import asyncio
import csv
import json
import os
import re
import sys
import argparse
import urllib.parse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from playwright.async_api import async_playwright, Page, Browser, TimeoutError as PWTimeoutError
import aiohttp
import aiofiles

FLIPKART_BASE = "https://www.flipkart.com"


def make_search_url(query: str) -> str:
    q = urllib.parse.quote_plus(query.strip())
    return f"{FLIPKART_BASE}/search?q={q}"


def add_or_replace_page_param(url: str, page: int) -> str:
    parsed = urllib.parse.urlparse(url)
    qs = dict(urllib.parse.parse_qsl(parsed.query))
    qs["page"] = str(page)
    new_query = urllib.parse.urlencode(qs)
    return urllib.parse.urlunparse(parsed._replace(query=new_query))


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def absolute_url(href: str) -> str:
    if href.startswith("http"):
        return href
    return urllib.parse.urljoin(FLIPKART_BASE, href)


def extract_pid_from_url(url: str) -> str:
    # Try query param pid
    parsed = urllib.parse.urlparse(url)
    qs = dict(urllib.parse.parse_qsl(parsed.query))
    if "pid" in qs and qs["pid"]:
        return qs["pid"]
    # Else use last path token (itm... / p/itm...)
    slug = Path(parsed.path).name
    return slug or "unknown_pid"


async def fetch_html(page: Page, url: str, wait_selector: Optional[str] = None, timeout_ms: int = 60000) -> str:
    try:
        # Use a lighter event than "networkidle"
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    except PWTimeoutError:
        # If it still times out, just grab whatever loaded
        print(f"[WARN] Timeout while loading {url}, using partial content.")
    
    if wait_selector:
        try:
            await page.wait_for_selector(wait_selector, timeout=timeout_ms)
        except PWTimeoutError:
            print(f"[WARN] Selector {wait_selector} not found quickly on {url}. Continuing anyway.")
    
    return await page.content()



def find_product_links_from_listing(html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    links = set()

    # Heuristic: product links usually contain /p/ and item id
    for a in soup.select('a[href*="/p/"]'):
        href = a.get("href")
        if not href:
            continue
        # Reduce noise: avoid cart/help/etc
        if "/p/" in href and "offer" not in href.lower():
            links.add(absolute_url(href))

    # Some cards use different wrappers; also try anchors with pid query
    for a in soup.select('a[href*="pid="]'):
        href = a.get("href")
        if href:
            links.add(absolute_url(href))

    return list(links)


def parse_ld_json_images(soup: BeautifulSoup) -> Tuple[List[str], Dict[str, Any]]:
    """
    Returns (image_urls, meta) where meta may include name/brand/price/rating.
    """
    images: List[str] = []
    meta: Dict[str, Any] = {}

    # Find all ld+json and pick the first with @type Product
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(tag.string or "{}")
        except Exception:
            continue

        # Sometimes it's a list
        candidates = data if isinstance(data, list) else [data]
        for obj in candidates:
            if isinstance(obj, dict) and obj.get("@type") in ("Product", ["Product"]):
                # Images can be list or string
                imgs = obj.get("image")
                if isinstance(imgs, list):
                    images.extend([u for u in imgs if isinstance(u, str)])
                elif isinstance(imgs, str):
                    images.append(imgs)

                meta["name"] = obj.get("name")
                # brand can be dict or str
                brand = obj.get("brand")
                if isinstance(brand, dict):
                    meta["brand"] = brand.get("name")
                elif isinstance(brand, str):
                    meta["brand"] = brand

                # price via offers
                offers = obj.get("offers") or {}
                if isinstance(offers, dict):
                    meta["price"] = offers.get("price")
                    meta["priceCurrency"] = offers.get("priceCurrency")

                # rating
                agg = obj.get("aggregateRating") or {}
                if isinstance(agg, dict):
                    meta["ratingValue"] = agg.get("ratingValue")
                    meta["reviewCount"] = agg.get("reviewCount")
                return list(dict.fromkeys(images)), meta  # de-dup preserve order

    # Fallback: scrape visible gallery imgs
    candidates = []
    for img in soup.select("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-lazy")
        if src and src.startswith("http"):
            candidates.append(src)
    return list(dict.fromkeys(candidates)), meta


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8),
       retry=retry_if_exception_type(Exception))
async def download_image(session: aiohttp.ClientSession, url: str, dest_path: Path) -> None:
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=40)) as resp:
        resp.raise_for_status()
        ensure_dir(dest_path.parent)
        async with aiofiles.open(dest_path, "wb") as f:
            async for chunk in resp.content.iter_chunked(64 * 1024):
                await f.write(chunk)


async def scrape_listing_and_products(start_url: str, pages: int, out_dir: Path, concurrency: int = 4) -> pd.DataFrame:
    ensure_dir(out_dir)
    images_dir = out_dir / "images"
    ensure_dir(images_dir)

    records: List[Dict[str, Any]] = []

    async with async_playwright() as p:
        browser: Browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
                        " AppleWebKit/537.36 (KHTML, like Gecko)"
                        " Chrome/122.0.0.0 Safari/537.36"),
            viewport={"width": 1366, "height": 900},
            java_script_enabled=True,
        )
        page: Page = await context.new_page()

        all_product_urls: List[str] = []

        for i in range(1, pages + 1):
            url = add_or_replace_page_param(start_url, i)
            html = await fetch_html(page, url, wait_selector='a[href*="/p/"]')
            links = find_product_links_from_listing(html)
            all_product_urls.extend(links)

        # Deduplicate product URLs (normalize removing tracking params)
        normed = []
        seen = set()
        for u in all_product_urls:
            pu = urllib.parse.urlparse(u)
            # Keep only essential params (pid if present)
            qs = dict(urllib.parse.parse_qsl(pu.query))
            keep_params = {}
            if "pid" in qs:
                keep_params["pid"] = qs["pid"]
            clean = urllib.parse.urlunparse(pu._replace(query=urllib.parse.urlencode(keep_params)))
            if clean not in seen:
                seen.add(clean)
                normed.append(clean)

        # Visit each product page (bounded concurrency)
        sem = asyncio.Semaphore(concurrency)

        async def process_product(u: str):
            async with sem:
                prod_page = await context.new_page()
                try:
                    content = await fetch_html(prod_page, u, wait_selector='script[type="application/ld+json"]')
                    soup = BeautifulSoup(content, "lxml")
                    img_urls, meta = parse_ld_json_images(soup)
                    pid = extract_pid_from_url(u)
                    # Download images
                    paths = []
                    async with aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0"}) as sess:
                        for idx, img_url in enumerate(img_urls):
                            # sanitize extension
                            ext = os.path.splitext(urllib.parse.urlparse(img_url).path)[1] or ".jpg"
                            safe_ext = ext.split("?")[0][:6] if ext else ".jpg"
                            dest = images_dir / pid / f"{pid}_{idx+1}{safe_ext}"
                            try:
                                await download_image(sess, img_url, dest)
                                paths.append(str(dest))
                            except Exception:
                                # skip failed image without crashing
                                continue

                    record = {
                        "product_url": u,
                        "pid": pid,
                        "title": meta.get("name"),
                        "brand": meta.get("brand"),
                        "price": meta.get("price"),
                        "price_currency": meta.get("priceCurrency"),
                        "rating_value": meta.get("ratingValue"),
                        "review_count": meta.get("reviewCount"),
                        "image_urls": ";".join(img_urls),
                        "downloaded_image_paths": ";".join(paths),
                    }
                    records.append(record)
                finally:
                    await prod_page.close()

        await asyncio.gather(*(process_product(u) for u in normed))

        await context.close()
        await browser.close()

    df = pd.DataFrame.from_records(records)
    # Save CSV
    csv_path = out_dir / "products.csv"
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Saved {len(df)} products to {csv_path}")
    return df


def parse_args():
    ap = argparse.ArgumentParser(description="Flipkart product scraper with pagination & image downloader")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--query", type=str, help="Search query, e.g. 'running shoes'")
    g.add_argument("--start-url", type=str, help="Full Flipkart search/category URL")
    ap.add_argument("--pages", type=int, default=1, help="Number of listing pages to crawl")
    ap.add_argument("--out", type=str, default="./flipkart_out", help="Output directory")
    ap.add_argument("--concurrency", type=int, default=4, help="Concurrent product page workers")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.query:
        base = make_search_url(args.query)
    else:
        base = args.start_url.strip()

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    # Run async
    try:
        asyncio.run(scrape_listing_and_products(base, args.pages, out_dir, args.concurrency))
    except KeyboardInterrupt:
        print("Aborted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
