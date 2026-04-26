import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor
from playwright.sync_api import sync_playwright
from markdownify import markdownify as md

# Domains known to use SPA (and therefore require headless browser)
SPA_DOMAINS = {
    "thehumaneleague.org"
}

class WebScraper:
    """
    WebScraper supports:
    1. scrape(url)          -> fetch and extract a single page (Markdown)
    2. crawl_and_scrape()   -> BFS crawl starting from seed(s)
    """

    INTERNAL_BLACKLIST = [
        '/login', '/cart', '/my-account', '/checkout', '/wp-admin',
        '/category/', '/tag/', '/author/', '?share=', 'feed/', '/comments/'
    ]

    def __init__(self, max_threads=3):
        self.max_threads = max_threads
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        })

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # -------------------
    # Helper Methods
    # -------------------
    def _is_valid(self, url, seed_url, visited):
        """Check domain match and filter blacklisted paths."""
        try:
            parsed_url = urlparse(url)
            parsed_seed = urlparse(seed_url)
            return (
                parsed_url.netloc == parsed_seed.netloc and
                not any(p in url for p in self.INTERNAL_BLACKLIST) and
                url not in visited
            )
        except Exception:
            return False

    def _process_url_dynamic(self, url):
        """Render JS-heavy pages using Playwright."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(2000)
                return page.content()
            except Exception as e:
                print(f"   ⚠️ Playwright error: {e}")
                return None
            finally:
                browser.close()

    def _extract_markdown_and_links(self, html_content, url, seed_url, visited, container_selector=None):
        """Convert HTML to cleaned Markdown and collect valid links."""
        soup = BeautifulSoup(html_content, 'lxml')

        # Remove boilerplate
        for noise in soup.select('header, footer, nav, script, style, aside, .admin-bar'):
            noise.decompose()

        # Containers
        containers = soup.select(container_selector) if container_selector else []
        if not containers:
            containers = [soup.find('main') or soup.find('article') or soup.body]

        all_texts, links = [], set()
        for container in containers:
            all_texts.append(str(container))
            for a in container.find_all('a', href=True):
                full_url = urljoin(url, a['href']).split('#')[0].rstrip('/')
                if self._is_valid(full_url, seed_url, visited):
                    links.add(full_url)

        # Convert combined HTML to Markdown
        combined_html = "\n".join(all_texts)
        markdown_text = md(combined_html)
        cleaned_markdown = "\n".join(line.rstrip() for line in markdown_text.split('\n') if line.strip())

        title = (soup.title.string or url).strip()
        return {"url": url, "title": title, "markdown": cleaned_markdown}, links

    # -------------------
    # Public Methods
    # -------------------
    def scrape(self, url, container_selector=None, use_js=False):
        """
        Fetch a single page and return {url, title, markdown}.
        use_js=True -> uses Playwright for JS-heavy pages
        """
        try:
            if use_js:
                page_content = self._process_url_dynamic(url)
            else:
                time.sleep(0.5)
                resp = self.session.get(url, timeout=15)
                resp.raise_for_status()
                page_content = resp.text

            if not page_content:
                return None, set()

            return self._extract_markdown_and_links(page_content, url, url, set(), container_selector)

        except Exception as e:
            print(f"   ❌ Scrape failed: {url} -> {e}")
            return None, set()

    def crawl_and_scrape(self, seed_url, max_depth=1, skip_ingesting_seed=False, container_selector=None):
        """
        BFS crawl starting from seed_url up to max_depth.
        Uses scrape() internally for each page.
        """
        all_results = []
        seed_url = seed_url.rstrip('/')
        visited = {seed_url}
        current_layer = [seed_url]

        for depth in range(max_depth + 1):
            if max_depth > 0:
                print(f"🌐 Depth {depth}: Exploring {len(current_layer)} pages...")
            next_layer = set()
            is_seed_layer = (depth == 0)

            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                use_js = get_scraping_mode(target_url) or is_seed_layer # JS for seed only, unless the domain is known to use SPA
                futures = [
                    executor.submit(
                        self.scrape,
                        u,
                        container_selector=container_selector,
                        use_js=use_js
                    )
                    for u in current_layer
                ]

                for f in futures:
                    result, found_links = f.result()
                    next_layer.update(found_links)
                    visited.update(found_links)

                    if result:
                        if is_seed_layer and skip_ingesting_seed:
                            print(f"   [Skipping Seed] {result['url']}")
                            continue
                        if len(result['markdown']) > 200:
                            print(f"   ✅ [MARKED TO INGEST] {result['url']} ({len(result['markdown'])} chars)")
                            all_results.append(result)

            if not next_layer or depth >= max_depth:
                break
            current_layer = list(next_layer)

        return all_results

    def close(self):
        """Final cleanup of all resources."""
        try:
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
            print("   ✅ Scraper resources released.")
        except Exception:
            # Silence errors during shutdown to prevent 'Double Crashes'
            pass