# Wikipedia PDF Downloader

This workspace contains a small Playwright-based automation that opens a Wikipedia article, tries the same menu flow you described, and saves a PDF locally.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m playwright install chromium
   ```
2. Run the downloader:
   ```bash
   python download_wikipedia_pdf.py "https://en.wikipedia.org/wiki/Python_(programming_language)" --output downloads/python.pdf
   ```

The script will save the PDF in the chosen output path or under the downloads folder by default.
