from playwright.async_api import Page

class StreamlitAppPage:
    """Page Object Model for the Streamlit application."""

    def __init__(self, page: Page):
        self.page = page

    async def goto(self, url: str):
        """Navigate to the specified URL."""
        await self.page.goto(url)

    async def get_title(self) -> str:
        """Get the title of the current page."""
        return await self.page.title()

    async def wait_for_main_title(self, main_title_text: str):
        """Wait for the main title to appear on the page."""
        await self.page.wait_for_selector(f"h1:has-text('{main_title_text}')", timeout=10000)

    async def get_sidebar_title(self):
        """Get the sidebar title element."""
        # Update the selector based on the actual HTML structure
        return await self.page.query_selector("section.stSidebar > div[class*='emotion-cache'] h1")

    async def get_main_title(self, main_title_text: str):
        """Get the main title element."""
        return await self.page.query_selector(f"h1:has-text('{main_title_text}')")