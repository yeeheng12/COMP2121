from playwright.sync_api import sync_playwright
import re

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    url = "https://www.unipol.org.uk/accommodation/adverts/details/338440/"
    print(f"Visiting: {url}")
    page.goto(url, wait_until="domcontentloaded")

    try:
        page.wait_for_selector("h2", timeout=3000)
        title = page.locator("article.property-portfolio h2").first.text_content(timeout=3000).strip()
        print("Title:", title)
    except:
        title = ""
        print("Could not extract title")

    # Weekly Rent
    try:
        price_text = page.locator("div.price span").first.text_content(timeout=2000).strip()
        price = re.sub(r"[^\d.]", "", price_text)
        print("Weekly Rent:", price)
    except:
        price = ""
        print("Price not found")

    # Rent Includes
    try:
        rent_block = page.locator("div.rentinclusive").first.text_content(timeout=2000)
        rent_includes = rent_block.replace("Rent Inclusive Of:", "").strip()
        print("Rent Includes:", rent_includes)
    except:
        rent_includes = ""
        print("Rent inclusions not found")

    # Deposit
    try:
        deposit_section = page.locator("div.deposit-details").first.text_content(timeout=2000).strip()
        deposit_match = re.search(r"Deposit:\s*(.*)", deposit_section)
        deposit = deposit_match.group(1).strip() if deposit_match else ""
        print("Deposit:", deposit)
    except:
        deposit = ""
        print("Could not extract deposit")

    # 🛏 Bedrooms listed (still useful for single units)
    try:
        bedroom_text = page.locator("div.row.bedrooms").first.text_content(timeout=2000).strip()
        bedroom_match = re.search(r"(\d+)\s+(bedroom|studio)", bedroom_text, re.IGNORECASE)
        bedrooms = bedroom_match.group(1) if bedroom_match else ""
        print("Bedrooms (listing says):", bedrooms)
    except:
        bedrooms = ""
        print("Could not extract bedroom info")

    # Total units available (e.g., "20" in "20 beds of 1 beds")
    try:
        beds_available = page.locator("div.row.bedrooms span.number").first.text_content().strip()
        print("Beds/Units Available:", beds_available)
    except:
        beds_available = ""
        print("Could not extract total beds/units available")

    # Distance to university placeholder
    distance_to_uni = ""
    print("Distance to university: (not extracted yet)")

    # Description from both collapsible sections
    description_lines = []

    def extract_paragraphs(section_id):
        try:
            block = page.locator(f"#{section_id}")
            if block.count() > 0:
                paragraphs = block.locator("p").all_text_contents()
                return [p.strip() for p in paragraphs if p.strip()]
        except:
            pass
        return []

    print("Extracting 'About the Building'...")
    description_lines += extract_paragraphs("collapseAboutBuilding")

    print("Extracting 'About the Property'...")
    description_lines += extract_paragraphs("collapseEight")

    unique_lines = list(dict.fromkeys(description_lines))
    full_description_cleaned = "\n".join(unique_lines)

    print("\nFULL CLEANED DESCRIPTION:\n")
    print(full_description_cleaned)

    browser.close()
