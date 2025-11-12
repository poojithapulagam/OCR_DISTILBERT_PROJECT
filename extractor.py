import re
import time
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def extract_entities(texts):
    start_time = time.time()
    results = []

    # Load pretrained NER model
    model_name = "dslim/bert-base-NER"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    for text in texts:
        clean_start = time.time()
        cleaned_text = text.strip()

        entities = nlp(cleaned_text)
        name = None
        address = None
        tracking_number = None

        # Combine multi-token names correctly (e.g., John Smith)
        person_tokens = [ent['word'] for ent in entities if ent['entity_group'] == 'PER']
        if person_tokens:
            name = " ".join(person_tokens).title()

        # Extract tracking numbers (UPS, FedEx, Amazon, USPS)
        tracking_match = re.search(
            r'\b(?:1Z[0-9A-Z]{8,20}|TBA[0-9A-Z]{8,20}|9[0-9]{15,22})\b',
            cleaned_text,
            re.IGNORECASE
        )
        if tracking_match:
            tracking_number = tracking_match.group(0).upper()

        # Extract address
        if "ship to" in cleaned_text.lower():
            after_ship = re.split(r"ship to", cleaned_text, flags=re.IGNORECASE)[-1]
            addr_match = re.search(
                r'([0-9]+\s+[a-zA-Z0-9\s,.]+,\s*[A-Za-z\s]+,\s*[A-Za-z]{2,}\s*\d{4,6})',
                after_ship
            )
            if addr_match:
                address = addr_match.group(1).strip().title()
            else:
                address = re.search(r'([0-9]+\s+[a-zA-Z0-9\s,.]+[A-Za-z\s]+)', after_ship)
                if address:
                    address = address.group(1).strip().title()
        else:
            addr_match = re.search(
                r'(\d{1,5}\s+[A-Za-z0-9\s,.]+,\s*[A-Za-z\s]+,\s*[A-Z]{2,}\s*\d{4,6})',
                cleaned_text
            )
            if addr_match:
                address = addr_match.group(1).strip().title()

        clean_end = time.time()
        results.append({
            "Name": name,
            "Address": address,
            "TrackingNumber": tracking_number,
            "CleanedText": cleaned_text,
            "TimeTaken": round(clean_end - clean_start, 3)
        })

    total_time = time.time() - start_time

    final_output = {
        "Total_Texts_Processed": len(texts),
        "Total_Execution_Time_Seconds": round(total_time, 3),
        "Results": results
    }

    # Save to JSON file automatically
    with open("final_extracted_results.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    return final_output
