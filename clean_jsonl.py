import json
import re

# ---------------------------------------------------------
# 1. File names
# ---------------------------------------------------------

INPUT_FILE = "corpus.jsonl"
OUTPUT_FILE = "data_clean.jsonl"
ERROR_FILE = "invalid_records.jsonl"


# ---------------------------------------------------------
# 2. Clean one piece of text
# ---------------------------------------------------------

def clean_text(text):
    """
    Clean the text field of one document.
    """

    # Convert anything to string
    text = str(text)

    # Convert Windows/Linux line endings to \n
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove Wikipedia-style infobox heading
    text = re.sub(
        r"\[Infobox[^\]]*\]\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Fix multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Fix repeated blank lines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    # Fix cases like:
    # 10:1811:30
    # becomes:
    # 10:18 and 11:30
    text = re.sub(
        r"(\b\d{1,2}:\d{2})(\d{1,2}:\d{2}\b)",
        r"\1 and \2",
        text
    )

    # Fix common extraction problems
    text = re.sub(
        r"\bHeatsSemifinals\b",
        "Heats and Semifinals",
        text,
        flags=re.IGNORECASE
    )

    # Remove spaces at the beginning/end of every line
    lines = []

    for line in text.split("\n"):

        line = line.strip()

        if not line:
            continue

        lines.append(line)

    # Join the cleaned lines again
    text = "\n\n".join(lines)

    # Final whitespace cleanup
    text = re.sub(r" +", " ", text)

    return text.strip()


# ---------------------------------------------------------
# 3. Clean one complete JSON record
# ---------------------------------------------------------

def clean_record(record, line_number):
    """
    Clean one JSON object.
    """

    # Get important fields
    doc_id = str(
        record.get("doc_id", f"doc_{line_number}")
    ).strip()

    title = str(
        record.get("title", "")
    ).strip()

    url = str(
        record.get("url", "")
    ).strip()

    wikidata_qid = str(
        record.get("wikidata_qid", "")
    ).strip()

    wikipedia_pageid = record.get(
        "wikipedia_pageid",
        None
    )

    text = record.get("text", "")

    # -----------------------------------------------------
    # Check whether text exists
    # -----------------------------------------------------

    if text is None:
        text = ""

    text = clean_text(text)

    if not text:
        return None

    # -----------------------------------------------------
    # Create clean record
    # -----------------------------------------------------

    clean_record = {
        "doc_id": doc_id,
        "title": title,
        "url": url,
        "wikidata_qid": wikidata_qid,
        "wikipedia_pageid": wikipedia_pageid,
        "text": text
    }

    return clean_record


# ---------------------------------------------------------
# 4. Main cleaning process
# ---------------------------------------------------------

def main():

    total_records = 0
    valid_records = 0
    invalid_records = 0
    duplicate_records = 0

    # Store IDs already seen
    seen_ids = set()

    # Open input file
    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as infile, \
    open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as outfile, \
    open(
        ERROR_FILE,
        "w",
        encoding="utf-8"
    ) as errorfile:

        # Read JSONL line by line
        for line_number, line in enumerate(
            infile,
            start=1
        ):

            total_records += 1

            # Remove spaces/newline
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # -------------------------------------------------
            # Convert JSON text into Python dictionary
            # -------------------------------------------------

            try:

                record = json.loads(line)

            except json.JSONDecodeError as error:

                invalid_records += 1

                error_data = {
                    "line_number": line_number,
                    "error": str(error),
                    "raw_line": line
                }

                errorfile.write(
                    json.dumps(
                        error_data,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                continue

            # -------------------------------------------------
            # Check that record is a dictionary
            # -------------------------------------------------

            if not isinstance(record, dict):

                invalid_records += 1

                error_data = {
                    "line_number": line_number,
                    "error": "Record is not a JSON object",
                    "raw_line": line
                }

                errorfile.write(
                    json.dumps(
                        error_data,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                continue

            # -------------------------------------------------
            # Get document ID
            # -------------------------------------------------

            doc_id = str(
                record.get(
                    "doc_id",
                    f"doc_{line_number}"
                )
            ).strip()

            # -------------------------------------------------
            # Remove duplicate documents
            # -------------------------------------------------

            if doc_id in seen_ids:

                duplicate_records += 1

                continue

            seen_ids.add(doc_id)

            # -------------------------------------------------
            # Clean record
            # -------------------------------------------------

            cleaned = clean_record(
                record,
                line_number
            )

            # -------------------------------------------------
            # Skip records with no useful text
            # -------------------------------------------------

            if cleaned is None:

                invalid_records += 1

                continue

            # -------------------------------------------------
            # Write cleaned record
            # -------------------------------------------------

            outfile.write(
                json.dumps(
                    cleaned,
                    ensure_ascii=False
                )
                + "\n"
            )

            valid_records += 1

    # ---------------------------------------------------------
    # 5. Print summary
    # ---------------------------------------------------------

    print("\nCleaning completed!")
    print("--------------------------------")
    print(f"Total records     : {total_records}")
    print(f"Valid records     : {valid_records}")
    print(f"Invalid records   : {invalid_records}")
    print(f"Duplicate records : {duplicate_records}")
    print("--------------------------------")

    print(f"Clean file        : {OUTPUT_FILE}")
    print(f"Error file        : {ERROR_FILE}")


# ---------------------------------------------------------
# 6. Start program
# ---------------------------------------------------------

if __name__ == "__main__":
    main()