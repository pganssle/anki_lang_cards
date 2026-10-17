# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

import csv
import json
import argparse
import re
from collections import defaultdict
from itertools import tee


def process_book_data(file_path):
    book_pages = defaultdict(list)

    with open(file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")

        for row in reader:
            source_json = row.get("sentence_source")

            if source_json:
                try:
                    data = json.loads(source_json)
                    title = data.get("title")
                    pages = data.get("pages", [])

                    if title and isinstance(pages, list):
                        book_pages[title].extend(pages)
                except json.JSONDecodeError:
                    continue

    if not book_pages:
        print(f"No valid data found in {file_path}")
        return None

    # Display results
    print(f"\n{'Book Title':<35} | {'Page Range':<15}")
    print("-" * 55)

    for title, pages in sorted(book_pages.items()):
        clean_pages = [p for p in pages if isinstance(p, (int, float))]
        if clean_pages:
            print(f"{title:<35} | {min(clean_pages)} - {max(clean_pages)}")
    print()

    return book_pages


def filter_csv(input_path, output_path, filter_spec, all_books):
    """
    Filter CSV rows based on page range for a specific book.

    Args:
        input_path: Path to input CSV
        output_path: Path to output CSV
        filter_spec: Tuple of (regex_pattern, start_page, end_page)
        all_books: Dictionary of all books and their pages (for validation)
    """
    regex_pattern, start_page, end_page = filter_spec

    # Find matching book title
    pattern = re.compile(regex_pattern)
    matching_titles = [title for title in all_books.keys() if pattern.search(title)]

    if len(matching_titles) == 0:
        print(f"Error: No book titles match the pattern '{regex_pattern}'")
        return
    elif len(matching_titles) > 1:
        print(f"Error: Multiple book titles match the pattern '{regex_pattern}':")
        for title in matching_titles:
            print(f"  - {title}")
        return

    target_title = matching_titles[0]
    print(f"\nFiltering entries for book: {target_title}")
    print(f"Excluding pages {start_page}-{end_page}\n")

    total_lines = 0
    filtered_lines = 0

    with open(input_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Reading fieldnames consumes the header, so line_num counts from the
        # first data row. A quoted field may span several physical lines, so
        # each row covers the lines from the end of the previous row up to the
        # current line_num; record every physical line of an excluded row.
        _ = reader.fieldnames
        excluded_lines = set()
        row_start = reader.line_num
        for row in reader:
            total_lines += 1

            source_json = row.get("sentence_source", "")

            should_exclude = False
            if source_json:
                try:
                    data = json.loads(source_json)
                    title = data.get("title")
                    pages = data.get("pages", [])

                    # If this row is for our target book, check page range
                    if title == target_title:
                        # Exclude if ALL pages are within the specified range
                        if pages and all(start_page <= p <= end_page for p in pages):
                            should_exclude = True
                except json.JSONDecodeError:
                    pass

            if should_exclude:
                excluded_lines.update(range(row_start, reader.line_num))
                filtered_lines += 1
            row_start = reader.line_num

        f.seek(0)
        lines_to_keep = [line for ii, line in enumerate(f) if ii not in excluded_lines]

    # Write all kept lines to output file
    with open(output_path, mode="w", encoding="utf-8") as f:
        f.writelines(lines_to_keep)

    print(f"Filtering complete:")
    print(f"  Total data rows processed: {total_lines}")
    print(f"  Rows excluded: {filtered_lines}")
    print(f"  Rows kept: {total_lines - filtered_lines}")
    print(f"  Output written to: {output_path}\n")


def parse_filter_spec(filter_string):
    """Parse filter specification like '(regex, start, end)'"""
    # Remove outer parentheses and split by comma
    filter_string = filter_string.strip()
    if not (filter_string.startswith("(") and filter_string.endswith(")")):
        raise ValueError("Filter must be in format '(regex, start, end)'")

    inner = filter_string[1:-1]

    # Split by comma, but be careful about commas in the regex
    parts = []
    current = []
    paren_depth = 0

    for char in inner:
        if char == "," and paren_depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth -= 1
            current.append(char)

    if current:
        parts.append("".join(current).strip())

    if len(parts) != 3:
        raise ValueError("Filter must have exactly 3 parts: (regex, start, end)")

    regex = parts[0].strip('"').strip("'")
    start = int(parts[1])
    end = int(parts[2])

    return regex, start, end


def main():
    parser = argparse.ArgumentParser(
        description="Extract page ranges per book from a CSV, optionally filtering by page range."
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default="output/words_from_annotations.csv",
        help="Path to the CSV file (default: output/words_from_annotations.csv)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        help='Filter specification: "(regex, start, end)" - excludes rows where all pages fall within range for matching book',
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for filtered CSV (required when using --filter)",
    )

    args = parser.parse_args()

    # First, process the book data to show statistics
    all_books = process_book_data(args.filename)

    # If filter is specified, apply it
    if args.filter:
        if not args.output:
            parser.error("--output is required when using --filter")

        if all_books is None:
            print("Cannot apply filter: no valid data found")
            return

        try:
            filter_spec = parse_filter_spec(args.filter)
            filter_csv(args.filename, args.output, filter_spec, all_books)
        except ValueError as e:
            print(f"Error parsing filter specification: {e}")
            return


if __name__ == "__main__":
    main()
