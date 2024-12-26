import asyncio
import json
from playwright.async_api import async_playwright
from streamlit_app_page import StreamlitAppPage

def load_config(file_path):
    """Load configuration from a JSON file."""
    with open(file_path) as config_file:
        return json.load(config_file)

async def run_test(config):
    """Run the end-to-end test for the Streamlit application."""
    async with async_playwright() as playwright:
        # Launch the browser in headless mode based on config
        browser = await playwright.chromium.launch(headless=config["browser_headless"])
        # Open a new browser page
        page = await browser.new_page()
        
        # Create an instance of the Page Object
        app_page = StreamlitAppPage(page)
        
        try:
            # Navigate to the application URL
            await app_page.goto(config["app_url"])
            
            # Wait for the app to load by waiting for a specific element
            await app_page.wait_for_main_title(config["main_title_text"])
            
            # Verify that the application loads successfully by checking for a specific element
            # Check if the title contains the main title text
            title = await app_page.get_title()
            assert config["main_title_text"] in title, f"Expected title '{config['main_title_text']}' not found in '{title}'"
                    
            # Check if the sidebar contains the sidebar title text
            sidebar_title = await app_page.get_sidebar_title()
            assert sidebar_title is not None, f"Expected sidebar title '{config['sidebar_title_text']}' not found"
            
            # Check if the main content contains the main title text
            main_title = await app_page.get_main_title(config["main_title_text"])
            assert main_title is not None, f"Expected main title '{config['main_title_text']}' not found"
            
            # Take a screenshot
            # Wait for the page to load completely
            await page.wait_for_load_state("networkidle")
            await page.screenshot(path="./tests/screenshots/screenshot.png")
            print("End-to-end test passed: Application is running and accessible.")
        except Exception as e:
            print(f"End-to-end test failed: {e}")
            raise
        finally:
            # Close the browser
            await browser.close()

if __name__ == "__main__":
    # Load configuration from file
    config = load_config('./tests/config.json')
    
    # Run the test
    asyncio.run(run_test(config))