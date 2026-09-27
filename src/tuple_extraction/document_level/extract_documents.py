import re
from pathlib import Path

import pandas as pd
import requests
import trafilatura


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

SOURCE_FILE = (
    REPO_ROOT
    / "data"
    / "tuple_extraction"
    / "document_level"
    / "Source_12_Documents.xlsx"
)

DOCUMENTS_DIR = SCRIPT_DIR / "documents"


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

# English stopwords excluded from campaign keyword extraction
STOPWORDS_EN = {
    "the", "and", "for", "of", "to", "in", "on", "at", "a", "an", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "must", "shall",
    "can", "need", "with", "by", "about", "into", "through", "during", "before",
    "after", "above", "below", "between", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "but", "if", "or",
    "because", "as", "until", "while", "this", "that", "these", "those", "from",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "whose", "i", "me", "my", "myself", "we",
    "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
}

EXCLUDED_SECTIONS = [
    "references",
    "external links",
    "see also",
    "notes",
    "categories",
    "hidden categories",
    "navigation",
    "retrieved from",
]


def remove_service_sections(text):
    """Remove Wikipedia footer and non-content service sections."""

    # Remove everything after the "Retrieved from" footer
    text = re.split(
        r'\nRetrieved from\s*"https?://',
        text,
        flags=re.IGNORECASE,
    )[0]

    # Remove service sections from their header to the next section
    section_pattern = (
        r'(?:^|\n)(?:#{1,6}\s*|={2,6}\s*)(?:'
        + "|".join(re.escape(section) for section in EXCLUDED_SECTIONS)
        + r')[^\n]*\n.*?'
          r'(?=(?:\n(?:#{1,6}\s*|={2,6}\s*)[A-Za-z][^\n]*|$))'
    )

    text = re.sub(
        section_pattern,
        "\n",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Remove residual category lines
    text = re.sub(
        r"\nCategory:[^\n]*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\nHidden categories:[^\n]*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


def extract_keywords(campaign_name):
    """Extract meaningful keywords from the campaign name."""

    clean_name = campaign_name.lower().replace("_", " ")
    words = [
        word
        for word in re.findall(r"\b\w+\b", clean_name)
        if len(word) > 2
    ]

    filtered_words = [
        word for word in words
        if word not in STOPWORDS_EN
    ]

    # Fallback to the original words if too few keywords remain
    if len(filtered_words) < 2 and len(words) >= 2:
        filtered_words = [
            word
            for word in words
            if word not in {"for", "the", "and", "of", "to", "a"}
        ]

    return filtered_words or words


def filter_wikipedia(text, keywords):
    """
    Filter Wikipedia content using campaign keywords and a ±1 line
    context window.
    """

    clean_text = remove_service_sections(text)
    lines = [
        line
        for line in clean_text.split("\n")
        if line.strip()
    ]

    if not lines or not keywords:
        return clean_text

    relevance_mask = [
        any(keyword in line.lower() for keyword in keywords)
        for line in lines
    ]

    relevant_lines = []

    for index, is_relevant in enumerate(relevance_mask):
        if is_relevant:
            start = max(0, index - 1)
            end = min(len(lines), index + 2)

            for line_index in range(start, end):
                if lines[line_index] not in relevant_lines:
                    relevant_lines.append(lines[line_index])

    return "\n".join(relevant_lines)


# ---------------------------------------------------------------------------
# Document extraction
# ---------------------------------------------------------------------------

def extract_documents(
    source_file=SOURCE_FILE,
    documents_dir=DOCUMENTS_DIR,
):
    """
    Retrieve the selected source documents and store their extracted text
    in campaign-specific directories.
    """

    source_file = Path(source_file)
    documents_dir = Path(documents_dir)

    if not source_file.exists():
        print(f"Source file not found: {source_file}")
        return

    documents_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(source_file, sheet_name="Sheet1")

    print(f"Found {len(df)} source documents.")

    for index, row in df.iterrows():

        source = str(row["FONTE"]).strip()
        url = str(row["LINK"]).strip()
        campaign_name = str(row["CAMPAGNA"]).strip()

        print(
            f"\n[{index + 1}/{len(df)}] "
            f"Campaign: {campaign_name} | Source: {source}"
        )

        if not url or url.lower() == "nan":
            print(" -> Missing or invalid URL. Skipping.")
            continue

        safe_campaign_name = re.sub(
            r'[\\/*?:"<>|;]',
            "_",
            campaign_name,
        )

        campaign_dir = documents_dir / safe_campaign_name
        campaign_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------------------------
        # Download
        # -------------------------------------------------------------------

        downloaded = trafilatura.fetch_url(url)

        if not downloaded:
            print(" -> Trying fallback download with requests...")

            try:
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                }

                response = requests.get(
                    url,
                    headers=headers,
                    timeout=15,
                )

                if response.status_code == 200:
                    downloaded = response.text

            except Exception as exc:
                print(f" -> Fallback download failed: {exc}")

        if downloaded:
            extracted_text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
            )

            if not extracted_text:
                print(" -> Unable to extract text from the page.")
                continue

        else:
            print(" -> URL download failed.")
            continue

        # -------------------------------------------------------------------
        # Conditional Wikipedia filtering
        # -------------------------------------------------------------------

        if "wikipedia" in source.lower():

            keywords = extract_keywords(campaign_name)
            print(f" -> Extracted keywords: {keywords}")

            text_to_save = filter_wikipedia(
                extracted_text,
                keywords,
            )

            total_lines = len(
                [
                    line
                    for line in extracted_text.split("\n")
                    if line.strip()
                ]
            )

            filtered_lines = len(
                [
                    line
                    for line in text_to_save.split("\n")
                    if line.strip()
                ]
            )

            print(
                f" -> [Wikipedia] Filtered from "
                f"{total_lines} to {filtered_lines} relevant lines."
            )

            if filtered_lines == 0:
                print(
                    f" -> WARNING: No lines found for keywords {keywords}."
                )

        else:
            text_to_save = extracted_text

            print(
                f" -> [Standard] Full extracted text retained "
                f"({len(text_to_save)} characters)."
            )

        # -------------------------------------------------------------------
        # Save document
        # -------------------------------------------------------------------

        safe_source_name = re.sub(
            r'[\\/*?:"<>|;]',
            "_",
            source,
        )[:100]

        output_file = campaign_dir / f"{safe_source_name}.txt"

        with output_file.open(
            "w",
            encoding="utf-8",
        ) as out:
            out.write(f"CAMPAGNA: {campaign_name}\n")
            out.write(f"FONTE: {source}\n")
            out.write(f"URL: {url}\n")
            out.write("=" * 50 + "\n\n")
            out.write(text_to_save)

        print(f" -> Saved to: {output_file}")

    print("\nCompleted. All source documents have been processed.")


if __name__ == "__main__":
    extract_documents()