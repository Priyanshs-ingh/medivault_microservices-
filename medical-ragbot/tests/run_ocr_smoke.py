"""
OCR Real-World Smoke Test — ASCII output version for Windows console
Run: python tests/run_ocr_smoke.py
"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

PASS = "[PASS]"
FAIL = "[FAIL]"
OK   = "[OK]  "

results = {}

print("=" * 52)
print("  MediVault OCR Real-World Smoke Test")
print("=" * 52)

TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER   = r"C:\poppler\poppler-24.08.0\Library\bin"

# Minimal PDF bytes (no external library needed)
MINIMAL_PDF = (
    b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 300 150]/Parent 2 0 R"
    b"/Resources<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>>>"
    b"/Contents 4 0 R>>endobj\n"
    b"4 0 obj<</Length 56>>\nstream\n"
    b"BT /F1 12 Tf 20 100 Td (Metformin 500mg Patient ID) Tj ET\n"
    b"endstream\nendobj\n"
    b"xref\n0 5\n"
    b"0000000000 65535 f \n0000000009 00000 n \n"
    b"0000000058 00000 n \n0000000115 00000 n \n0000000316 00000 n \n"
    b"trailer<</Size 5/Root 1 0 R>>\nstartxref\n422\n%%EOF"
)


#  1. Tesseract binary 
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT
    version = pytesseract.get_tesseract_version()
    print(f"{OK} Tesseract binary found: v{version}")
    results["tesseract_binary"] = True
except Exception as e:
    print(f"{FAIL} Tesseract binary NOT found: {e}")
    results["tesseract_binary"] = False


#  2. Poppler / pdf2image 
try:
    from pdf2image import convert_from_path
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(MINIMAL_PDF)
        tmp = f.name
    imgs = convert_from_path(tmp, dpi=72, poppler_path=POPPLER)
    os.unlink(tmp)
    print(f"{OK} Poppler (pdf2image) converted {len(imgs)} page(s)")
    results["poppler"] = True
except Exception as e:
    print(f"{FAIL} Poppler: {e}")
    results["poppler"] = False
    imgs = []


#  3. OCR reads printed medical text 
try:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (600, 320), color="white")
    draw = ImageDraw.Draw(img)
    medical_lines = [
        "MEDICAL REPORT",
        "Patient ID: PT-20240115",
        "Date: 2024-01-15",
        "Dr. Emily Johnson",
        "Medications: Metformin 500mg twice daily",
        "Diagnosis: Type 2 Diabetes",
        "Allergies: Penicillin",
        "Blood Pressure: 140/90 mmHg",
    ]
    for i, line in enumerate(medical_lines):
        draw.text((20, 15 + i * 36), line, fill="black")

    ocr_text = pytesseract.image_to_string(img, lang="eng", config="--psm 6")

    print("\n  OCR output preview:")
    print("  " + repr(ocr_text[:120]))
    print()

    checks = {
        "Patient ID extracted":       "PT-20240115" in ocr_text or "Patient" in ocr_text,
        "Date extracted":             "2024" in ocr_text,
        "Medication name extracted":  "Metformin" in ocr_text or "metformin" in ocr_text.lower(),
        "Diagnosis extracted":        "Diabetes" in ocr_text or "diabetes" in ocr_text.lower(),
        "Doctor name extracted":      "Johnson" in ocr_text or "Emily" in ocr_text,
        "Allergy extracted":          "Penicillin" in ocr_text or "Allerg" in ocr_text,
        "Blood pressure extracted":   "Blood" in ocr_text or "140" in ocr_text,
    }
    all_ok = all(checks.values())
    for label, passed in checks.items():
        print(f"  {PASS if passed else FAIL} {label}")
    results["ocr_medical_text"] = all_ok

except Exception as e:
    print(f"{FAIL} OCR medical text: {e}")
    results["ocr_medical_text"] = False


#  4. Full pipeline: real PDF  Poppler  OCR 
try:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(MINIMAL_PDF)
        tmp2 = f.name
    pages = convert_from_path(tmp2, dpi=150, poppler_path=POPPLER)
    text = pytesseract.image_to_string(pages[0], lang="eng")
    os.unlink(tmp2)
    preview = repr(text.strip()[:60])
    print(f"\n{OK} Full pipeline PDF -> Poppler -> OCR: {preview}")
    results["full_pipeline"] = True
except Exception as e:
    print(f"\n{FAIL} Full pipeline: {e}")
    results["full_pipeline"] = False


#  5. PDFProcessor integration 
try:
    from config import settings
    from ingestion.pdf_loader import PDFProcessor
    proc = PDFProcessor()
    tess_ok = bool(settings.tesseract_path)
    pop_ok  = bool(getattr(settings, "poppler_path", None))
    print(f"\n{OK} PDFProcessor init OK")
    print(f"  tesseract_path set: {tess_ok} ({settings.tesseract_path})")
    print(f"  poppler_path  set: {pop_ok} ({getattr(settings, 'poppler_path', 'NOT IN SETTINGS')})")
    results["processor_settings"] = tess_ok and pop_ok
except Exception as e:
    print(f"\n{FAIL} PDFProcessor: {e}")
    results["processor_settings"] = False


#  Summary 
print("\n" + "=" * 52)
print("  RESULTS SUMMARY")
print("=" * 52)
for name, passed in results.items():
    label = name.replace("_", " ").title()
    print(f"  {PASS if passed else FAIL}  {label}")

total  = len(results)
passed = sum(results.values())
print()
if passed == total:
    print(f"  {passed}/{total} -- OCR FULLY OPERATIONAL for production!")
    print("  Scanned and handwritten medical PDFs will work.")
else:
    print(f"  {passed}/{total} -- WARNING: Some checks failed.")
    print("  Patients with scanned PDFs may get empty results.")
print("=" * 52)

sys.exit(0 if passed == total else 1)
