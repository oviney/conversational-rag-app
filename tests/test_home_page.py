import asyncio
import json
from playwright.async_api import async_playwright

def load_config(file_path):
    """Load configuration from a JSON file."""
    with open(file_path) as config_file:
        return json.load(config_file)

async def load_page(url: str, headless: bool = True):
    """Load the webpage and return the rendered HTML content."""
    async with async_playwright() as playwright:
        # Launch the browser
        browser = await playwright.chromium.launch(headless=headless)
        # Create a new browser context
        context = await browser.new_context()
        # Open a new page
        page = await context.new_page()
        
        # Navigate to the specified URL
        await page.goto(url)
        
        # Wait for the page to load completely
        await page.wait_for_load_state("networkidle")
        
        # Get the fully rendered HTML content
        rendered_html = await page.content()
        
        # Close the browser
        await browser.close()
        
        return rendered_html

async def main():
    """Main function to run the test."""
    # Load configuration from file
    config = load_config('./tests/config.json')
    
    # Get the URL from the configuration
    url = config["app_url"]
    
    # Load the page and get the rendered HTML content
    rendered_html = await load_page(url, headless=config["browser_headless"])
    print(rendered_html)

if __name__ == "__main__":
    asyncio.run(main())
