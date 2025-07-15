#!/usr/bin/env python3
"""
Best Buy Review Scraper - Playwright Version

This script scrapes customer reviews from Best Buy product pages using Playwright
for better reliability and modern web scraping.

Usage:
    python bestbuy_scraper_playwright.py --url "https://www.bestbuy.com/site/..." --output reviews.html

Requirements:
    - Install Playwright: pip install playwright
    - Install browsers: playwright install chromium
"""

import argparse
import asyncio
import logging
import sys
from typing import List
from pathlib import Path

from playwright.async_api import async_playwright, Browser, Page
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BestBuyReviewScraperPlaywright:
    """
    A scraper for Best Buy product reviews using Playwright.
    """
    
    def __init__(self, headless: bool = True, timeout: int = 30000):
        """
        Initialize the scraper.
        
        Args:
            headless: Whether to run browser in headless mode
            timeout: Page timeout in milliseconds
        """
        self.headless = headless
        self.timeout = timeout
        self.browser = None
        self.page = None
    
    async def setup_browser(self):
        """Set up the Playwright browser."""
        try:
            playwright = await async_playwright().start()
            
            # Launch browser with options
            self.browser = await playwright.chromium.launch(
                headless=self.headless,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-web-security',
                    '--allow-running-insecure-content'
                ]
            )
            
            # Create new page
            self.page = await self.browser.new_page()
            
            # Set timeout
            self.page.set_default_timeout(self.timeout)
            
            # Set user agent
            await self.page.set_user_agent(
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36'
            )
            
            logger.info("Playwright browser initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Playwright browser: {e}")
            raise
    
    async def wait_for_reviews_to_load(self) -> bool:
        """
        Wait for the reviews section to fully load on the page.
        
        Returns:
            bool: True if reviews loaded successfully, False otherwise
        """
        try:
            # Multiple selectors to try for the reviews container
            review_container_selectors = [
                "div[data-testid='reviews-section']",
                ".reviews-wrapper",
                ".reviews-container",
                ".ugc-reviews",
                "[data-testid='ugc-reviews']",
                ".review-list",
                "#reviews-section"
            ]
            
            container_found = False
            for selector in review_container_selectors:
                try:
                    await self.page.wait_for_selector(selector, timeout=10000)
                    logger.info(f"Reviews container found with selector: {selector}")
                    container_found = True
                    break
                except Exception:
                    continue
            
            if not container_found:
                logger.warning("Reviews container not found with standard selectors")
            
            # Wait for individual review items
            review_selectors = [
                "li.review-item",
                ".review-item",
                "[data-testid='review']",
                ".review",
                ".review-content",
                ".customer-review"
            ]
            
            reviews_found = False
            for selector in review_selectors:
                try:
                    await self.page.wait_for_selector(selector, timeout=10000)
                    logger.info(f"Individual reviews found with selector: {selector}")
                    reviews_found = True
                    break
                except Exception:
                    continue
            
            if not reviews_found:
                logger.warning("No individual reviews found with standard selectors")
                return False
            
            # Wait a bit more for content to stabilize
            await asyncio.sleep(2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error waiting for reviews to load: {e}")
            return False
    
    async def scroll_to_load_more_reviews(self) -> None:
        """
        Scroll down to potentially load more reviews if they're lazy-loaded.
        """
        try:
            # Get initial page height
            last_height = await self.page.evaluate("document.body.scrollHeight")
            
            # Scroll down in steps
            for i in range(5):
                # Scroll to bottom
                await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                
                # Wait for new content to load
                await asyncio.sleep(3)
                
                # Check if page height changed (new content loaded)
                new_height = await self.page.evaluate("document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
                
                logger.info(f"Scrolled iteration {i+1}, page height: {new_height}")
            
            # Try to find and click "Load More" button if it exists
            load_more_selectors = [
                "button[data-testid='load-more-reviews']",
                ".load-more-reviews",
                "button:has-text('Load More')",
                "button:has-text('Show More')"
            ]
            
            for selector in load_more_selectors:
                try:
                    load_more_btn = await self.page.query_selector(selector)
                    if load_more_btn:
                        is_visible = await load_more_btn.is_visible()
                        is_enabled = await load_more_btn.is_enabled()
                        if is_visible and is_enabled:
                            await load_more_btn.click()
                            logger.info(f"Clicked load more button: {selector}")
                            await asyncio.sleep(3)
                            break
                except Exception:
                    continue
            
            logger.info("Completed scrolling to load additional reviews")
            
        except Exception as e:
            logger.warning(f"Error during scrolling: {e}")
    
    async def extract_review_elements(self) -> List[str]:
        """
        Extract review elements from the current page.
        
        Returns:
            List[str]: List of HTML strings for each review element
        """
        try:
            # Get the page content
            page_content = await self.page.content()
            soup = BeautifulSoup(page_content, 'html.parser')
            
            # Multiple selectors to try for review items
            review_selectors = [
                "li.review-item",
                ".review-item",
                "[data-testid='review']",
                ".review",
                ".review-content",
                ".customer-review",
                ".ugc-review",
                ".review-card"
            ]
            
            review_elements = []
            
            for selector in review_selectors:
                elements = soup.select(selector)
                if elements:
                    logger.info(f"Found {len(elements)} reviews with selector: {selector}")
                    review_elements = elements
                    break
            
            if not review_elements:
                logger.warning("No review elements found with any selector")
                # Try a more generic approach
                all_elements = soup.find_all(attrs={'data-testid': True})
                review_elements = [elem for elem in all_elements if 'review' in elem.get('data-testid', '').lower()]
                if review_elements:
                    logger.info(f"Found {len(review_elements)} reviews using generic data-testid approach")
            
            if not review_elements:
                return []
            
            # Extract outer HTML for each review
            review_html_list = []
            for element in review_elements:
                try:
                    # Get the complete HTML of the review element
                    review_html = str(element)
                    if review_html.strip():  # Only add non-empty HTML
                        review_html_list.append(review_html)
                except Exception as e:
                    logger.warning(f"Error extracting HTML from review element: {e}")
                    continue
            
            logger.info(f"Successfully extracted {len(review_html_list)} review elements")
            return review_html_list
            
        except Exception as e:
            logger.error(f"Error extracting review elements: {e}")
            return []
    
    async def scrape_reviews(self, url: str) -> List[str]:
        """
        Scrape reviews from a Best Buy product page.
        
        Args:
            url: The URL of the Best Buy product page
            
        Returns:
            List[str]: List of HTML strings for each review element
        """
        try:
            logger.info(f"Starting to scrape reviews from: {url}")
            
            # Navigate to the page
            await self.page.goto(url, wait_until='networkidle')
            logger.info("Page loaded successfully")
            
            # Wait for reviews to load
            if not await self.wait_for_reviews_to_load():
                logger.warning("Reviews may not have loaded properly")
            
            # Scroll to potentially load more reviews
            await self.scroll_to_load_more_reviews()
            
            # Extract review elements
            review_elements = await self.extract_review_elements()
            
            if not review_elements:
                logger.warning("No review elements found on the page")
                return []
            
            logger.info(f"Successfully scraped {len(review_elements)} reviews")
            return review_elements
            
        except Exception as e:
            logger.error(f"Error scraping reviews: {e}")
            return []
    
    def save_reviews_to_file(self, review_elements: List[str], output_path: str) -> bool:
        """
        Save review elements to an HTML file.
        
        Args:
            review_elements: List of HTML strings for each review
            output_path: Path to save the output file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not review_elements:
                logger.warning("No review elements to save")
                return False
            
            # Create output directory if it doesn't exist
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a complete HTML document with proper structure
            html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Best Buy Reviews</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .reviews-container {{ max-width: 1200px; margin: 0 auto; }}
        .review {{ margin-bottom: 20px; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .review-header {{ font-weight: bold; margin-bottom: 10px; color: #333; }}
        h1 {{ text-align: center; color: #0046be; margin-bottom: 30px; }}
        .review-count {{ text-align: center; color: #666; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>Best Buy Product Reviews</h1>
    <div class="review-count">Total Reviews: {count}</div>
    <div class="reviews-container">
{reviews}
    </div>
</body>
</html>"""
            
            # Wrap each review in a div for better structure
            wrapped_reviews = []
            for i, review in enumerate(review_elements):
                wrapped_review = f'        <div class="review" id="review-{i+1}">\n            {review}\n        </div>'
                wrapped_reviews.append(wrapped_review)
            
            # Combine all reviews
            combined_html = html_template.format(
                count=len(review_elements),
                reviews='\n'.join(wrapped_reviews)
            )
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(combined_html)
            
            logger.info(f"Successfully saved {len(review_elements)} reviews to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving reviews to file: {e}")
            return False
    
    async def close(self):
        """Clean up and close the browser."""
        if self.browser:
            try:
                await self.browser.close()
                logger.info("Browser closed successfully")
            except Exception as e:
                logger.warning(f"Error closing browser: {e}")


def validate_url(url: str) -> bool:
    """
    Validate that the URL is a Best Buy product page.
    
    Args:
        url: The URL to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not url:
        return False
    
    # Check if it's a Best Buy URL
    if 'bestbuy.com' not in url.lower():
        logger.error("URL must be a Best Buy product page")
        return False
    
    # Check if it's a product page (contains '/site/')
    if '/site/' not in url:
        logger.error("URL must be a Best Buy product page (should contain '/site/')")
        return False
    
    return True


async def main():
    """Main async function to run the scraper."""
    parser = argparse.ArgumentParser(
        description='Scrape customer reviews from Best Buy product pages using Playwright',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python bestbuy_scraper_playwright.py --url "https://www.bestbuy.com/site/..." --output reviews.html
    python bestbuy_scraper_playwright.py --url "https://www.bestbuy.com/site/..." --output reviews.html --headless false
        """
    )
    
    parser.add_argument(
        '--url',
        required=True,
        help='The full URL of the Best Buy product page'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='The file path to save the resulting HTML (e.g., reviews.html)'
    )
    
    parser.add_argument(
        '--headless',
        default='true',
        choices=['true', 'false'],
        help='Run browser in headless mode (default: true)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Page timeout in seconds (default: 30)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not validate_url(args.url):
        sys.exit(1)
    
    # Convert headless string to boolean
    headless_mode = args.headless.lower() == 'true'
    
    # Convert timeout to milliseconds
    timeout_ms = args.timeout * 1000
    
    # Initialize scraper
    scraper = None
    try:
        logger.info("Initializing Best Buy review scraper with Playwright...")
        scraper = BestBuyReviewScraperPlaywright(headless=headless_mode, timeout=timeout_ms)
        
        # Setup browser
        await scraper.setup_browser()
        
        # Scrape reviews
        review_elements = await scraper.scrape_reviews(args.url)
        
        if not review_elements:
            logger.error("No reviews found on the page")
            sys.exit(1)
        
        # Save reviews to file
        if scraper.save_reviews_to_file(review_elements, args.output):
            print(f"✅ Successfully scraped {len(review_elements)} reviews and saved to {args.output}")
        else:
            logger.error("Failed to save reviews to file")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        sys.exit(1)
    finally:
        if scraper:
            await scraper.close()


if __name__ == "__main__":
    asyncio.run(main())