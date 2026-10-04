#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_DIR="$ROOT_DIR/samples"
mkdir -p "$SAMPLE_DIR"

python3 - "$SAMPLE_DIR" <<'PY'
import json
import random
import sys
from pathlib import Path

import pandas as pd
import pymupdf

sample_dir = Path(sys.argv[1]).resolve()
sample_dir.mkdir(parents=True, exist_ok=True)

random.seed(7)
months = ["2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08"]
base = {
    "Product A": {"North": 900, "South": 800, "East": 700},
    "Product B": {"North": 700, "South": 600, "East": 500},
    "Product C": {"North": 500, "South": 450, "East": 400},
    "Product D": {"North": 300, "South": 280, "East": 250},
}
growth = {"North": 0.03, "East": 0.025, "South": 0.002}
rows = []
for i, month in enumerate(months):
    for product, regions in base.items():
        for region, value in regions.items():
            revenue = value * (1 + growth[region]) ** i * random.uniform(0.98, 1.02)
            if product == "Product B" and month == "2024-08":
                revenue *= 0.70
            if product == "Product A" and region == "South" and month in {"2024-07", "2024-08"}:
                revenue *= 0.97
            rows.append((f"{month}-15", product, region, round(revenue, 2)))

df = pd.DataFrame(rows, columns=["Date", "Product", "Region", "Revenue"])
df.to_csv(sample_dir / "sales.csv", index=False)
df.to_excel(sample_dir / "sales.xlsx", index=False)

pages = [
    (
        "Acme Retail - Management Report, August 2024",
        "Executive summary\n\nAugust was a mixed month. Overall revenue was broadly stable, with growth in Products A, C and D\n"
        "offset by a sharp fall in Product B. Management is focused on stabilising supply and\nrestoring availability in September.\n\n"
        "Highlights:\n- Customer satisfaction remained at 4.3 out of 5.\n- The new loyalty programme reached 120,000 members.\n- Operating costs were in line with budget.",
    ),
    (
        "Marketing and customer activity",
        "The marketing team ran the summer brand campaign across social channels.\nWebsite traffic increased in August and email subscribers grew steadily.\n"
        "The loyalty programme improved repeat purchases among existing customers.\nNo pricing changes were made to any product during the quarter.",
    ),
    (
        "Product performance - Product B",
        "Product B revenue declined sharply in August.\n\n"
        "Management attributed the decline mainly to inventory shortages. A delayed shipment from the\n"
        "primary supplier left stores without stock for roughly two weeks, and several large orders\n"
        "were cancelled. A secondary factor was a short-lived promotion by a competitor.\n\n"
        "Corrective actions: a second supplier has been approved and safety stock levels for\nProduct B will be raised by 25 percent.",
    ),
    (
        "Regional review - South",
        "The South region delivered the weakest growth over the half-year.\n\n"
        "Management explained that warehouse delays at the Kochi distribution centre caused repeated\n"
        "supply disruptions in South. Product A, the best-selling product in the region, was most\n"
        "affected, with late deliveries and partial order fulfilment in July and August.\n\n"
        "A new regional logistics partner will start in October.",
    ),
    (
        "Outlook and risks",
        "For September, management expects revenue to recover as Product B stock returns.\n"
        "Key risks: supplier concentration, logistics capacity in the South region, and rising freight costs.\n"
        "Product D remains a small but steady contributor.",
    ),
]

doc = pymupdf.open()
for title, body in pages:
    page = doc.new_page()
    page.insert_text((72, 80), title, fontsize=16)
    page.insert_textbox(pymupdf.Rect(72, 110, 540, 760), body, fontsize=11, lineheight=1.4)
doc.save(sample_dir / "Management_Report.pdf")
doc.close()

with open(sample_dir / "eval_dataset.json", "w", encoding="utf-8") as fh:
    json.dump(
        [
            {
                "question": "What explanation did management give for the decline in Product B?",
                "expected_answer": "inventory shortages delayed shipment supplier",
                "expected_file": "Management_Report.pdf",
                "expected_page": 3,
            },
            {
                "question": "What caused supply problems in the South region?",
                "expected_answer": "warehouse delays Kochi distribution centre",
                "expected_file": "Management_Report.pdf",
                "expected_page": 4,
            },
            {
                "question": "What is the corrective action for Product B?",
                "expected_answer": "second supplier safety stock",
                "expected_file": "Management_Report.pdf",
                "expected_page": 3,
            },
        ],
        fh,
        indent=2,
    )

print(f"Created sample files in {sample_dir}")
for path in sorted(sample_dir.iterdir()):
    print(path.name)
PY
