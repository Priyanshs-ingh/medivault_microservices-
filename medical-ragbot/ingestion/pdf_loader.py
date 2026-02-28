"""
PDF and OCR Processing Module
Handles both digital and handwritten medical reports
Supports dynamic processing of any number of PDFs
"""
import os
from typing import List, Dict, Optional
from pathlib import Path
import logging

import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Processes PDF files - both digital and scanned/handwritten.
    Designed for dynamic, scalable document ingestion.
    """
    
    def __init__(self):
        if settings.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_path
    
    def extract_text_from_pdf(
        self, 
        pdf_path: str,
        save_processed: bool = True
    ) -> Dict[str, any]:
        """
        Extract text from PDF. Tries digital extraction first,
        falls back to OCR if needed.
        
        Args:
            pdf_path: Path to PDF file
            save_processed: Whether to save extracted text to processed_text/
            
        Returns:
            Dictionary containing extracted text and rich metadata
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Try digital text extraction first
        text = self._extract_digital_text(pdf_path)
        extraction_method = "digital"
        
        # If digital extraction yields little content, use OCR
        if len(text.strip()) < 100:
            logger.info("Digital extraction insufficient, using OCR...")
            text = self._extract_with_ocr(pdf_path)
            extraction_method = "ocr"
        
        # Extract metadata
        filename = Path(pdf_path).name
        
        # Extract rich metadata from text content
        metadata = self._extract_metadata_from_text(text)
        
        # Save processed text if requested
        if save_processed and text:
            self._save_processed_text(text, filename)
        
        return {
            "text": text,
            "source": pdf_path,
            "filename": filename,
            "extraction_method": extraction_method,
            "page_count": self._get_page_count(pdf_path),
            "file_size_kb": os.path.getsize(pdf_path) / 1024,
            # Rich metadata extracted from document
            "doctor_name": metadata.get("doctor_name"),
            "hospital_name": metadata.get("hospital_name"),
            "report_date": metadata.get("report_date"),
            "report_type": metadata.get("report_type"),
            "patient_id": metadata.get("patient_id"),
        }
    
    def _extract_digital_text(self, pdf_path: str) -> str:
        """Extract text from digital PDF using pdfplumber"""
        try:
            text_content = []
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(f"[Page {page_num}]\n{page_text}")
                    
                    # Extract tables separately to preserve structure
                    tables = page.extract_tables()
                    if tables:
                        for table_idx, table in enumerate(tables, 1):
                            table_text = self._format_table(table)
                            text_content.append(
                                f"\n[Table {table_idx} on Page {page_num}]\n{table_text}"
                            )
            
            return "\n\n".join(text_content)
        except Exception as e:
            logger.error(f"Digital extraction failed: {e}")
            return ""
    
    def _extract_with_ocr(self, pdf_path: str) -> str:
        """Extract text using OCR for scanned/handwritten documents"""
        try:
            # Convert PDF to images (pass poppler_path for Windows compatibility)
            poppler_path = settings.poppler_path_or_none if hasattr(settings, 'poppler_path_or_none') else (settings.poppler_path or None)
            images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
            
            text_content = []
            for page_num, image in enumerate(images, 1):
                # Perform OCR on each page
                page_text = pytesseract.image_to_string(
                    image,
                    lang=settings.ocr_language,
                    config='--psm 6'  # Assume uniform text block
                )
                if page_text.strip():
                    text_content.append(f"[Page {page_num}]\n{page_text}")
            
            return "\n\n".join(text_content)
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""
    
    def _format_table(self, table: List[List]) -> str:
        """Format table data as readable text"""
        if not table:
            return ""
        
        formatted_rows = []
        for row in table:
            # Filter out None values and join
            cells = [str(cell).strip() if cell else "" for cell in row]
            formatted_rows.append(" | ".join(cells))
        
        return "\n".join(formatted_rows)
    
    def _get_page_count(self, pdf_path: str) -> int:
        """Get number of pages in PDF"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                return len(pdf.pages)
        except Exception as e:
            logger.warning(f"Could not get page count for {pdf_path}: {e}")
            return 0
    
    def _extract_metadata_from_text(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract rich metadata from document text using pattern matching.
        
        Args:
            text: Full document text
            
        Returns:
            Dictionary with extracted metadata fields
        """
        import re
        
        metadata = {
            "doctor_name": None,
            "hospital_name": None,
            "report_date": None,
            "report_type": None,
            "patient_id": None,
        }
        
        # Extract doctor name
        doctor_patterns = [
            r'(?i)(?:Dr\.?|Doctor)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
            r'(?i)Physician[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
            r'(?i)Attending[:\s]+Dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
            r'(?i)Signed[:\s]+Dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
        ]
        for pattern in doctor_patterns:
            match = re.search(pattern, text)
            if match:
                metadata["doctor_name"] = match.group(1).strip()
                break
        
        # Extract hospital/clinic name
        hospital_patterns = [
            r'(?i)(?:Hospital|Clinic|Medical Center|Health Center)[:\s]*([A-Z][A-Za-z\s&]+(?:Hospital|Clinic|Medical Center|Center))',
            r'(?i)([A-Z][A-Za-z\s&]+(?:Hospital|Clinic|Medical Center))\s*\n',
            r'(?i)Facility[:\s]+([A-Z][A-Za-z\s&]+)',
        ]
        for pattern in hospital_patterns:
            match = re.search(pattern, text)
            if match:
                hospital = match.group(1).strip()
                # Clean up and limit length
                if len(hospital) < 100:  # Reasonable hospital name length
                    metadata["hospital_name"] = hospital
                    break
        
        # Extract report date
        date_patterns = [
            r'(?i)Date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(?i)Report Date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(?i)(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s*\n',
            r'(?i)Date[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',  # January 15, 2026
            r'(?i)(\d{4}-\d{2}-\d{2})',  # ISO format
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text[:1000])  # Search first 1000 chars
            if match:
                metadata["report_date"] = match.group(1).strip()
                break
        
        # Extract report type
        report_type_patterns = [
            (r'(?i)(?:lab|laboratory)\s+report', 'lab_report'),
            (r'(?i)blood\s+test', 'blood_test'),
            (r'(?i)radiology\s+report', 'radiology'),
            (r'(?i)(?:x-ray|xray|radiograph)', 'xray'),
            (r'(?i)(?:mri|magnetic resonance)', 'mri'),
            (r'(?i)(?:ct|computed tomography)', 'ct_scan'),
            (r'(?i)ultrasound', 'ultrasound'),
            (r'(?i)discharge\s+summary', 'discharge_summary'),
            (r'(?i)consultation\s+note', 'consultation'),
            (r'(?i)progress\s+note', 'progress_note'),
            (r'(?i)operative\s+report', 'operative_report'),
            (r'(?i)pathology\s+report', 'pathology'),
        ]
        for pattern, report_type in report_type_patterns:
            if re.search(pattern, text[:500]):
                metadata["report_type"] = report_type
                break
        
        # Extract patient ID (anonymized)
        patient_id_patterns = [
            r'(?i)Patient ID[:\s]+(\w+-?\d+)',
            r'(?i)MRN[:\s]+(\w+-?\d+)',
            r'(?i)Medical Record[:\s]+(\w+-?\d+)',
        ]
        for pattern in patient_id_patterns:
            match = re.search(pattern, text[:1000])
            if match:
                metadata["patient_id"] = match.group(1).strip()
                break
        
        return metadata
    
    def _save_processed_text(self, text: str, filename: str):
        """Save extracted text to processed_text directory"""
        try:
            processed_dir = Path(settings.data_dir) / "processed_text"
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            text_filename = processed_dir / f"{Path(filename).stem}.txt"
            with open(text_filename, 'w', encoding='utf-8') as f:
                f.write(text)
            
            logger.info(f"Saved processed text to: {text_filename}")
        except Exception as e:
            logger.warning(f"Could not save processed text: {e}")
    
    def batch_extract(self, pdf_paths: List[str]) -> List[Dict[str, any]]:
        """
        Extract text from multiple PDFs.
        Fully dynamic - handles any number of documents.
        
        Args:
            pdf_paths: List of PDF file paths
            
        Returns:
            List of extraction results
        """
        results = []
        total = len(pdf_paths)
        
        logger.info(f"Starting batch extraction for {total} PDFs...")
        
        for idx, pdf_path in enumerate(pdf_paths, 1):
            try:
                logger.info(f"Processing {idx}/{total}: {Path(pdf_path).name}")
                result = self.extract_text_from_pdf(pdf_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                results.append({
                    "text": "",
                    "source": pdf_path,
                    "filename": Path(pdf_path).name,
                    "error": str(e)
                })
        
        logger.info(f"Batch extraction complete: {len(results)} documents processed")
        return results
    
    def extract_from_directory(self, directory: str) -> List[Dict[str, any]]:
        """
        Extract text from all PDFs in a directory.
        Enables truly dynamic ingestion from data/raw_pdfs/
        
        Args:
            directory: Path to directory containing PDFs
            
        Returns:
            List of extraction results
        """
        pdf_dir = Path(directory)
        if not pdf_dir.exists():
            logger.error(f"Directory not found: {directory}")
            return []
        
        # Find all PDF files
        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        if not pdf_files:
            logger.warning("No PDF files found")
            return []
        
        return self.batch_extract([str(pdf) for pdf in pdf_files])


# Example usage
if __name__ == "__main__":
    processor = PDFProcessor()
    
    # Example: Process single PDF
    # result = processor.extract_text_from_pdf("data/raw_pdfs/report1.pdf")
    
    # Example: Process all PDFs in directory
    # results = processor.extract_from_directory("data/raw_pdfs")
    
    print("PDF Processor initialized and ready")
