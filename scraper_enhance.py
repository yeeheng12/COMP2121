from playwright.sync_api import sync_playwright
import pandas as pd
import re
import time

def scrape_listing(page, url):
    try:
        page.goto(url, timeout=15000, wait_until="domcontentloaded")
    except:
        print(f"Skipping {url} — page did not load in time.")
        return None

    try:
        page.wait_for_selector("h2", timeout=3000)
        title = page.locator("article.property-portfolio h2").first.text_content(timeout=3000).strip()
        print("Title:", title)
    except:
        title = ""
        print("Could not extract title")

    try:
        price_node = page.locator("div.price span").first
        price = re.sub(r"[^\d.]", "", price_node.text_content(timeout=2000).strip())
    except:
        price = ""

    try:
        rent_node = page.locator("text=Rent Inclusive Of:").locator("xpath=..")
        rent_includes = rent_node.text_content(timeout=2000).replace("Rent Inclusive Of:", "").strip()
    except:
        rent_includes = ""

    try:
        deposit_section = page.locator("div.deposit-details").first.text_content(timeout=2000).strip()
        deposit_match = re.search(r"Deposit:\s*(.*)", deposit_section)
        deposit = deposit_match.group(1).strip() if deposit_match else ""
        print("Deposit:", deposit)
    except:
        deposit = ""
        print("Could not extract deposit")

    description_parts = []

    def extract_paragraphs(section_id):
        try:
            block = page.locator(f"#{section_id}")
            if block.count() > 0:
                return block.locator("p").all_text_contents()
        except:
            return []
        return []

    about_building = extract_paragraphs("collapseAboutBuilding")
    about_property = extract_paragraphs("collapseEight")

    seen = set()
    for para in about_building + about_property:
        cleaned = para.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            description_parts.append(cleaned)

    full_description = "\n".join(description_parts)

    return {
        "title": title,
        "weekly_rent": price,
        "deposit": deposit,
        "rent_includes": rent_includes,
        "description": full_description,
        "url": url
    }

def scrape_all_unipol():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        base_url = "https://www.unipol.org.uk/leeds/accommodation/"
        all_data = []

        property_types = ["Whole Property", "Rooms in a shared property/large development"]
        bedrooms_range = range(1, 10)

        for bedrooms in bedrooms_range:
            for property_type in property_types:
                print(f"\nSearching: {bedrooms} bedrooms | {property_type}")

                page.goto(base_url)
                time.sleep(6)

                # Select number of bedrooms by dropdown value
                try:
                    page.select_option("select[name='BedroomsFrom']", str(bedrooms))
                    time.sleep(1)
                except Exception as e:
                    print(f"Could not select bedrooms: {e}")
                    continue

                # Select Whole Property Required dropdown by id
                try:
                    value = "2" if property_type == "Whole Property" else "3"
                    page.wait_for_selector("#ddl-whole-property-required", state="attached", timeout=5000)
                    page.select_option("#ddl-whole-property-required", value)
                    time.sleep(1)
                except Exception as e:
                    print(f"Could not select Whole Property Required: {e}")
                    continue

                # Click the search button
                try:
                    page.locator("#btn-apply-filter").click()
                    time.sleep(6)
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    time.sleep(3)
                except Exception as e:
                    print(f"Search click failed: {e}")
                    continue

                listings = page.locator("li.popup").all()
                listing_ids = list(set([l.get_attribute("id") for l in listings if l.get_attribute("id")]))

                print(f"Found {len(listing_ids)} listings for {bedrooms} BR | {property_type}")

                details_base = "https://www.unipol.org.uk/accommodation/adverts/details/"
                full_urls = [details_base + lid for lid in listing_ids]

                for idx, url in enumerate(full_urls):
                    print(f"Scraping ({idx+1}/{len(full_urls)}): {url}")
                    try:
                        record = scrape_listing(page, url)
                        if record:
                            all_data.append(record)
                    except Exception as e:
                        print(f"Error scraping {url}: {e}")
                        continue

        browser.close()
        df = pd.DataFrame(all_data)
        df.to_csv("unipol_listings_raw.csv", index=False)
        print(f"\nSaved {len(df)} listings to unipol_listings_raw.csv")

if __name__ == "__main__":
    scrape_all_unipol()
