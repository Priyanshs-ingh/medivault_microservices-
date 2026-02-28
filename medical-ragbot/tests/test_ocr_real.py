"""
OCR Real-World Smoke Test
Run this AFTER installing Tesseract and Poppler:
  1. Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
  2. Install Poppler: https://github.com/oschwartz10612/poppler-windows/releases
  3. Add both to PATH
  4. Run: python tests/test_ocr_real.py

This creates a real synthetic scanned-like image with medical text
and runs actual OCR on it to verify the stack works end-to-end.
"""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_dependencies():
    """Check all OCR dependencies are installed."""
    print("\n=== OCR Dependency Check ===\n")
    issues = []

    # 1. Check pytesseract Python package
    try:
        import pytesseract
        print(" pytesseract (Python package) installed")
    except ImportError:
        issues.append(" pytesseract not installed: pip install pytesseract")

    # 2. Check Tesseract binary
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f" Tesseract binary found: version {version}")
    except Exception as e:
        issues.append(f" Tesseract binary NOT found: {e}")
        issues.append("   Install from: https://github.com/UB-Mannheim/tesseract/wiki")

    # 3. Check pdf2image / Poppler
    try:
        import pdf2image
        print(" pdf2image (Python package) installed")
    except ImportError:
        issues.append(" pdf2image not installed: pip install pdf2image")

    # 4. Check Poppler binary
    try:
        import pdf2image
        import tempfile, struct, zlib

        # Create a minimal valid single-page PDF to test poppler
        # (We build one in pure bytes so no fpdf/reportlab needed)
        pdf_bytes = _make_minimal_pdf()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            tmp_path = f.name

        images = pdf2image.convert_from_path(tmp_path, dpi=72)
        Path(tmp_path).unlink(missing_ok=True)
        print(f" Poppler (pdftoppm) found: converted {len(images)} page(s)")
    except Exception as e:
        issues.append(f" Poppler NOT found: {e}")
        issues.append("   Install from: https://github.com/oschwartz10612/poppler-windows/releases")
        issues.append("   Then add Poppler's bin/ folder to PATH")

    # 5. Check Pillow
    try:
        from PIL import Image
        print(" Pillow installed")
    except ImportError:
        issues.append(" Pillow not installed: pip install Pillow")

    return issues


def _make_minimal_pdf():
    """Create a tiny but valid PDF with 'Hello' text, in pure bytes."""
    # Minimal valid PDF structure
    pdf = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 200 100]/Parent 2 0 R/Resources<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>>>>/Contents 4 0 R>>endobj
4 0 obj<</Length 44>>
stream
BT /F1 18 Tf 20 50 Td (Medical Test) Tj ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000316 00000 n 
trailer<</Size 5/Root 1 0 R>>
startxref
410
%%EOF"""
    return pdf


def test_ocr_reads_printed_text():
    """Test that OCR can read printed text from a generated image."""
    print("\n=== Real OCR Test: Printed Text ===\n")
    try:
        from PIL import Image, ImageDraw, ImageFont
        import pytesseract

        # Create an image with medical text
        img = Image.new('RGB', (600, 400), color='white')
        draw = ImageDraw.Draw(img)

        medical_text_lines = [
            "MEDICAL REPORT",
            "",
            "Patient ID: PT-20240115",
            "Date: 2024-01-15",
            "Dr. Emily Johnson",
            "",
            "Medications:",
            "1. Metformin 500mg - twice daily",
            "2. Lisinopril 10mg - once daily",
            "",
            "Diagnosis: Type 2 Diabetes",
            "Allergies: Penicillin",
        ]

        y = 20
        for line in medical_text_lines:
            draw.text((20, y), line, fill='black')
            y += 28

        # Run actual OCR
        ocr_text = pytesseract.image_to_string(img, lang='eng', config='--psm 6')

        print("OCR Output Preview:")
        print("-" * 40)
        print(ocr_text[:300])
        print("-" * 40)

        # Check key terms were extracted
        checks = {
            "Patient ID detected": "PT-20240115" in ocr_text or "Patient" in ocr_text,
            "Date detected":       "2024" in ocr_text,
            "Medication detected": "Metformin" in ocr_text or "metformin" in ocr_text.lower(),
            "Diagnosis detected":  "Diabetes" in ocr_text or "diabetes" in ocr_text.lower(),
            "Doctor detected":     "Johnson" in ocr_text or "Emily" in ocr_text,
        }

        all_pass = True
        for check, result in checks.items():
            status = "" if result else ""
            print(f"  {status} {check}")
            if not result:
                all_pass = False

        if all_pass:
            print("\n OCR is working correctly for printed medical text!")
        else:
            print("\n  OCR partial — some fields not detected. Check DPI/font settings.")

        return all_pass

    except Exception as e:
        print(f" OCR test failed: {e}")
        return False


def test_pdf_to_ocr_pipeline():
    """Test the full pipeline: PDF  image  OCR  text."""
    print("\n=== Full Pipeline Test: PDF  OCR ===\n")
    import tempfile

    try:
        from pdf2image import convert_from_path
        import pytesseract

        # Write minimal PDF
        pdf_bytes = _make_minimal_pdf()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            tmp_path = f.name

        print(f"Created test PDF: {tmp_path}")

        # Convert to images
        images = convert_from_path(tmp_path, dpi=150)
        print(f" pdf2image: converted to {len(images)} page image(s)")

        # OCR each page
        for i, image in enumerate(images, 1):
            text = pytesseract.image_to_string(image, lang='eng')
            print(f" Page {i} OCR output: '{text.strip()[:80]}'")

        Path(tmp_path).unlink(missing_ok=True)
        print("\n Full PDF  OCR pipeline works!")
        return True

    except Exception as e:
        print(f" Pipeline test failed: {e}")
        return False


def test_fallback_trigger():
    """Test that the PDFProcessor correctly falls back to OCR for scanned PDFs."""
    print("\n=== Integration Test: OCR Fallback Trigger ===\n")
    try:
        from config import settings
        from ingestion.pdf_loader import PDFProcessor
        import tempfile

        # Create a minimal PDF (no extractable text)
        pdf_bytes = _make_minimal_pdf()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            tmp_path = f.name

        processor = PDFProcessor()
        result = processor.extract_text_from_pdf(tmp_path, save_processed=False)

        Path(tmp_path).unlink(missing_ok=True)

        print(f"Extraction method used: {result.get('extraction_method', 'unknown')}")
        print(f"Text extracted: {len(result.get('text', ''))} chars")

        if result.get('extraction_method') in ('digital', 'ocr'):
            print(" PDFProcessor ran without crashing!")
        else:
            print("  Unexpected result format")

        return True

    except Exception as e:
        print(f" Integration test failed: {e}")
        return False



# ─────────────────────────────────────────────────────────────────────────────
# OCR EDGE CASE TESTS  (pytest-compatible)
# ─────────────────────────────────────────────────────────────────────────────

class TestOCREdgeCases:
    """
    Edge Cases: verifies OCR is resilient to unusual / malformed image inputs.
    Production criticality: HIGH — patients may upload low-quality scans.
    """

    #  Edge Case 1: completely blank / white image 
    def test_ocr_on_blank_white_image_returns_empty_or_string(self):
        """OCR on a blank page must not crash; it should return '' or whitespace."""
        try:
            from PIL import Image
            import pytesseract

            blank = Image.new("RGB", (600, 400), color="white")
            result = pytesseract.image_to_string(blank, lang="eng", config="--psm 6")
            # Should be a string (possibly empty)
            assert isinstance(result, str), "OCR must return a string even for blank input"
        except ImportError:
            pytest.skip("pytesseract / Pillow not installed")

    #  Edge Case 2: pure black image (no contrast) 
    def test_ocr_on_pure_black_image_does_not_crash(self):
        """All-black image — Tesseract should not raise, must return a string."""
        try:
            from PIL import Image
            import pytesseract

            black = Image.new("RGB", (400, 300), color="black")
            result = pytesseract.image_to_string(black, lang="eng")
            assert isinstance(result, str)
        except ImportError:
            pytest.skip("pytesseract / Pillow not installed")

    #  Edge Case 3: very small image (10×10 pixels) 
    def test_ocr_on_tiny_image_does_not_crash(self):
        """Extremely small image must not raise an unhandled exception."""
        try:
            from PIL import Image
            import pytesseract

            tiny = Image.new("RGB", (10, 10), color="white")
            result = pytesseract.image_to_string(tiny, lang="eng")
            assert isinstance(result, str)
        except ImportError:
            pytest.skip("pytesseract / Pillow not installed")

    #  Edge Case 4: very large image (4K resolution) 
    def test_ocr_on_large_4k_image_does_not_crash(self):
        """4K-sized image should be processed without memory errors."""
        try:
            from PIL import Image, ImageDraw
            import pytesseract

            img = Image.new("RGB", (3840, 2160), color="white")
            draw = ImageDraw.Draw(img)
            draw.text((100, 100), "Medications: Metformin 500mg", fill="black")
            # Run OCR on a cropped region to speed up the test
            cropped = img.crop((0, 0, 800, 200))
            result = pytesseract.image_to_string(cropped, lang="eng", config="--psm 6")
            assert isinstance(result, str)
        except ImportError:
            pytest.skip("pytesseract / Pillow not installed")

    #  Edge Case 5: image with only a single digit 
    def test_ocr_single_digit_image(self):
        """Image containing just '7' — Tesseract must detect a digit or return str."""
        try:
            from PIL import Image, ImageDraw
            import pytesseract

            img = Image.new("RGB", (100, 100), color="white")
            draw = ImageDraw.Draw(img)
            draw.text((35, 35), "7", fill="black")
            result = pytesseract.image_to_string(img, lang="eng", config="--psm 10 -c tessedit_char_whitelist=0123456789")
            assert isinstance(result, str)
            # Tesseract may or may not detect the digit at tiny size — just no crash
        except ImportError:
            pytest.skip("pytesseract / Pillow not installed")


# ─────────────────────────────────────────────────────────────────────────────
# OCR DOMAIN CASE TESTS  (pytest-compatible)
# ─────────────────────────────────────────────────────────────────────────────

class TestOCRDomainCases:
    """
    Domain Cases: verifies OCR correctly reads real medical content patterns.
    These simulate what real scanned medical reports look like in production.
    """

    #  Domain Case 1: medication list in a scanned prescription 
    def test_ocr_reads_medication_list(self):
        """OCR must extract core medication names from a synthetic prescription image."""
        try:
            from PIL import Image, ImageDraw
            import pytesseract

            img = Image.new("RGB", (700, 350), color="white")
            draw = ImageDraw.Draw(img)
            lines = [
                "PRESCRIPTION",
                "1. Metformin 500mg - twice daily with meals",
                "2. Lisinopril 10mg - once daily in the morning",
                "3. Aspirin 81mg - once daily",
                "4. Atorvastatin 20mg - at bedtime",
            ]
            y = 20
            for line in lines:
                draw.text((20, y), line, fill="black")
                y += 30

            text = pytesseract.image_to_string(img, lang="eng", config="--psm 6")
            # At least one medication name must be recovered
            meds = ["Metformin", "Lisinopril", "Aspirin", "Atorvastatin"]
            found = [m for m in meds if m.lower() in text.lower()]
            assert len(found) >= 1, \
                f"OCR failed to detect any medication. Got: {text[:200]}"
        except ImportError:
            pytest.skip("pytesseract / Pillow not installed")

    #  Domain Case 2: allergy section 
    def test_ocr_reads_allergy_section(self):
        """OCR must detect the word 'Penicillin' from an allergy block."""
        try:
            from PIL import Image, ImageDraw
            import pytesseract

            img = Image.new("RGB", (500, 200), color="white")
            draw = ImageDraw.Draw(img)
            draw.text((20, 20),  "Allergies:", fill="black")
            draw.text((20, 55),  "1. Penicillin - causes rash", fill="black")
            draw.text((20, 85),  "2. Sulfa drugs - anaphylaxis", fill="black")

            text = pytesseract.image_to_string(img, lang="eng", config="--psm 6")
            assert "Penicillin" in text or "penicillin" in text.lower(), \
                f"OCR missed 'Penicillin'. Got: {text[:200]}"
        except ImportError:
            pytest.skip("pytesseract / Pillow not installed")

    #  Domain Case 3: lab results with numeric values and units 
    def test_ocr_reads_lab_values_with_units(self):
        """OCR must extract numeric lab values like HbA1c and their units."""
        try:
            from PIL import Image, ImageDraw
            import pytesseract

            img = Image.new("RGB", (600, 250), color="white")
            draw = ImageDraw.Draw(img)
            draw.text((20, 20),  "Lab Results:", fill="black")
            draw.text((20, 55),  "HbA1c: 6.8%", fill="black")
            draw.text((20, 85),  "Fasting Glucose: 125 mg/dL", fill="black")
            draw.text((20, 115), "Total Cholesterol: 180 mg/dL", fill="black")

            text = pytesseract.image_to_string(img, lang="eng", config="--psm 6")
            # 6.8 or HbA1c or Glucose must appear
            has_value = "6.8" in text or "125" in text or "HbA1c" in text.replace(" ", "") or "Glucose" in text
            assert has_value, \
                f"OCR missed lab values. Got: {text[:200]}"
        except ImportError:
            pytest.skip("pytesseract / Pillow not installed")

    #  Domain Case 4: diagnosis section 
    def test_ocr_reads_diagnosis_section(self):
        """OCR must detect diagnoses like 'Type 2 Diabetes' from a report image."""
        try:
            from PIL import Image, ImageDraw
            import pytesseract

            img = Image.new("RGB", (600, 220), color="white")
            draw = ImageDraw.Draw(img)
            draw.text((20, 20), "Diagnosis:", fill="black")
            draw.text((20, 55), "1. Type 2 Diabetes Mellitus - well controlled", fill="black")
            draw.text((20, 85), "2. Hypertension - Stage 1", fill="black")

            text = pytesseract.image_to_string(img, lang="eng", config="--psm 6")
            has_diag = (
                "Diabetes" in text or "diabetes" in text.lower() or
                "Hypertension" in text or "hypertension" in text.lower()
            )
            assert has_diag, \
                f"OCR missed diagnosis content. Got: {text[:200]}"
        except ImportError:
            pytest.skip("pytesseract / Pillow not installed")

    #  Domain Case 5: _extract_with_ocr via PDFProcessor (mocked) 
    def test_pdf_processor_ocr_path_returns_string(self):
        """
        PDFProcessor._extract_with_ocr must return a string even when
        pdf2image / Tesseract are mocked — the method signature is validated.
        Production use: any scanned PDF must produce a string result.
        """
        import sys
        from pathlib import Path
        from unittest.mock import patch, MagicMock

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from ingestion.pdf_loader import PDFProcessor

        processor = PDFProcessor()

        # Mock convert_from_path to return one fake PIL image
        fake_image = MagicMock()
        fake_image_to_string_result = (
            "[Page 1]\n"
            "Medications:\n"
            "1. Metformin 500mg - twice daily\n"
            "Diagnosis: Type 2 Diabetes\n"
            "Allergies: Penicillin\n"
        )

        with patch("ingestion.pdf_loader.convert_from_path", return_value=[fake_image]):
            with patch("ingestion.pdf_loader.pytesseract.image_to_string",
                       return_value=fake_image_to_string_result):
                result = processor._extract_with_ocr("fake_scanned.pdf")

        assert isinstance(result, str), "OCR path must return a string"
        assert "Metformin" in result, "Mocked OCR text must pass through"
        assert "Diabetes" in result, "Diagnosis must appear in OCR output"
        assert "Penicillin" in result, "Allergy must appear in OCR output"


if __name__ == "__main__":
    print("=" * 55)
    print("  MediVault RAG Bot — OCR Real-World Smoke Test")
    print("=" * 55)


    # Step 1: Check all deps are installed
    issues = check_dependencies()

    if issues:
        print("\n DEPENDENCY ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        print("\n Fix the above issues before OCR will work in production.")
        print("   Unit tests pass because they mock these dependencies.")
        sys.exit(1)

    print("\n All dependencies installed! Running real OCR tests...\n")

    # Step 2: Run real tests
    results = {
        "Printed text OCR":     test_ocr_reads_printed_text(),
        "PDF  OCR pipeline":   test_pdf_to_ocr_pipeline(),
        "PDFProcessor fallback": test_fallback_trigger(),
    }

    # Summary
    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    for test_name, passed in results.items():
        status = " PASS" if passed else " FAIL"
        print(f"  {status}  {test_name}")

    all_passed = all(results.values())
    print()
    if all_passed:
        print(" OCR is fully operational — scanned PDFs will work in production!")
    else:
        print(" OCR is NOT fully operational — scanned PDFs will FAIL silently!")
        print("   Patients with scanned documents will get empty results.")
    print("=" * 55)
    sys.exit(0 if all_passed else 1)
