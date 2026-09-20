from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from PIL import Image, ImageDraw, ImageFont
import os

OUTPUT = "data/sample/pdfs/mixed_test_4_pages.pdf"
TEMP_IMAGE = "data/sample/pdfs/_scanned_test_page.png"

W, H = A4

# ---------------------------------------------------------
# Create an image that will be embedded as a scanned page
# ---------------------------------------------------------
img = Image.new("RGB", (1653, 2339), "white")
draw = ImageDraw.Draw(img)

font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"

if not os.path.exists(font_path):
    font_path = "/System/Library/Fonts/Helvetica.ttc"

font = ImageFont.truetype(font_path, 54)
small_font = ImageFont.truetype(font_path, 38)

draw.text(
    (180, 180),
    "SCANNED DOCUMENT PAGE",
    fill="black",
    font=font
)

draw.line((180, 280, 1450, 280), fill="black", width=3)

lines = [
    "This page is intentionally embedded as an image.",
    "It should be classified as SCANNED.",
    "OCR TEST DOCUMENT",
    "Document ID: MIXED-TEST-2026",
    "Purpose: Validate OCR routing and preprocessing.",
    "Expected OCR text: SCANNED PAGE CONTENT",
]

y = 380

for line in lines:
    draw.text((180, y), line, fill="black", font=small_font)
    y += 100

img.save(TEMP_IMAGE, dpi=(200, 200))


# ---------------------------------------------------------
# Create the 4-page PDF
# ---------------------------------------------------------
pdf = canvas.Canvas(OUTPUT, pagesize=A4)

pdf.setTitle("Mixed PDF Pipeline Test - 4 Pages")


# PAGE 1 — Native PDF text
pdf.setFont("Helvetica-Bold", 20)
pdf.drawString(70, H - 80, "MIXED PDF PIPELINE TEST")

pdf.setFont("Helvetica", 12)

page1 = [
    "PAGE 1 — NATIVE TEXT",
    "This page contains real PDF text, not an embedded image.",
    "Expected classification: TEXT",
    "Expected routing: direct text extraction.",
    "Test ID: MIXED-TEST-2026-P1",
]

y = H - 130

for line in page1:
    pdf.drawString(70, y, line)
    y -= 28

pdf.showPage()


# PAGE 2 — Scanned image
pdf.drawImage(
    ImageReader(TEMP_IMAGE),
    0,
    0,
    width=W,
    height=H
)

pdf.showPage()


# PAGE 3 — Native PDF text
pdf.setFont("Helvetica-Bold", 20)
pdf.drawString(70, H - 80, "MIXED PDF PIPELINE TEST")

pdf.setFont("Helvetica", 12)

page3 = [
    "PAGE 3 — NATIVE TEXT",
    "This is another true PDF text page.",
    "Expected classification: TEXT",
    "Expected routing: direct text extraction.",
    "Test ID: MIXED-TEST-2026-P3",
]

y = H - 130

for line in page3:
    pdf.drawString(70, y, line)
    y -= 28

pdf.showPage()


# PAGE 4 — Scanned image
pdf.drawImage(
    ImageReader(TEMP_IMAGE),
    0,
    0,
    width=W,
    height=H
)

pdf.showPage()

pdf.save()

os.remove(TEMP_IMAGE)

print(f"Created: {OUTPUT}")
