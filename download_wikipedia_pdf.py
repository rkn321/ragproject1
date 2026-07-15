import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright


def normalize_wikipedia_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    parsed = urlparse(url)
    if "wikipedia.org" not in parsed.netloc:
        raise ValueError("Please provide a valid Wikipedia article URL")
    return url


def sanitize_title(url: str) -> str:
    title = urlparse(url).path.strip("/").split("/")[-1]
    title = title.replace("_", " ")
    title = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("._")
    return title or "wikipedia_article"


def try_click(page, selectors: list[str], timeout: int = 8000) -> bool:
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if locator.count() > 0:
                locator.click(timeout=timeout)
                return True
        except Exception:
            continue
    return False


def open_tools_menu(page) -> None:
    """Reveal the collapsed 'Tools' sidebar section that holds 'Download as PDF' (Vector 2022 skin)."""
    checkbox = page.locator("#vector-page-tools-dropdown-checkbox")
    if checkbox.count() > 0:
        try:
            # The checkbox itself sits on top of its label and intercepts pointer
            # events, so a normal click times out; force=True bypasses that check.
            checkbox.check(force=True, timeout=5000)
            return
        except Exception:
            pass

    # Older/alternate skins expose a real button instead of the checkbox toggle.
    try_click(page, [
        'button[aria-label="Tools"]',
        'button[aria-label="More actions"]',
        'button[aria-label="More"]',
    ])


def download_wikipedia_pdf(url: str, output_path: str | None = None, headless: bool = True) -> Path:
    normalized_url = normalize_wikipedia_url(url)
    output = Path(output_path or f"downloads/{sanitize_title(normalized_url)}.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        context = browser.new_context(accept_downloads=True, viewport={"width": 1440, "height": 900})
        page = context.new_page()

        try:
            page.goto(normalized_url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_load_state("networkidle", timeout=60000)

            open_tools_menu(page)

            # Follow "Download as PDF" to Wikipedia's export/confirmation page.
            reached_export_page = try_click(page, [
                'a[href*="Special:DownloadAsPdf"]',
                "a:has-text('Download as PDF')",
            ])

            if reached_export_page:
                page.wait_for_load_state("networkidle", timeout=60000)
                try:
                    # The button click must happen *inside* this block, otherwise
                    # the download event fires before Playwright starts listening.
                    with page.expect_download(timeout=60000) as download_info:
                        clicked = try_click(page, [
                            "button:has-text('Download')",
                            "a:has-text('Download')",
                            "input[type='submit'][value='Download']",
                        ])
                        if not clicked:
                            raise RuntimeError("Could not find the Download button on the export page")
                    download_info.value.save_as(str(output))
                    return output
                except (PlaywrightTimeoutError, RuntimeError):
                    pass  # fall through to the page.pdf() fallback below

            # Fallback: print the article page to a PDF directly. Only works headless.
            if not headless:
                raise RuntimeError(
                    "Could not trigger Wikipedia's PDF export, and the page.pdf() "
                    "fallback only works in headless mode. Re-run without --headed."
                )
            if page.url != normalized_url:
                page.goto(normalized_url, wait_until="networkidle", timeout=60000)
            page.pdf(path=str(output), format="A4", print_background=True)
            return output
        finally:
            browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a Wikipedia article as a PDF")
    parser.add_argument("url", help="Wikipedia article URL")
    parser.add_argument("--output", help="Path where the PDF should be saved")
    parser.add_argument("--headed", action="store_true", help="Run the browser in headed mode")
    args = parser.parse_args()

    try:
        result = download_wikipedia_pdf(args.url, args.output, headless=not args.headed)
        print(f"Saved PDF to: {result}")
    except Exception as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        sys.exit(1)
