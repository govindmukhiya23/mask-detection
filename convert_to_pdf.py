#!/usr/bin/env python3
"""
Convert PROJECT_DOCUMENT.md to PDF
"""
import sys

try:
    from markdown_pdf import MarkdownPdf, Section
    
    pdf = MarkdownPdf(toc_level=2)
    pdf.meta["title"] = "Face Mask Detection Project - Complete Documentation"
    pdf.meta["author"] = "Deep Learning Lab Project"
    
    # Add the markdown file
    pdf.add_section(Section("PROJECT_DOCUMENT.md"))
    
    # Save as PDF
    pdf.save("PROJECT_DOCUMENT.pdf")
    
    print("✓ SUCCESS: PROJECT_DOCUMENT.pdf has been created!")
    print("✓ Location: c:\\Users\\Govin\\Downloads\\dl Lab project\\Lab project\\PROJECT_DOCUMENT.pdf")
    print("\nYou can now:")
    print("  1. Open PROJECT_DOCUMENT.pdf to view the formatted document")
    print("  2. Print the PDF for hard copy submission")
    print("  3. Submit the PDF file electronically")
    
except ImportError as e:
    print(f"Error: Required library not found - {e}")
    print("Please install: pip install markdown-pdf")
    sys.exit(1)
except Exception as e:
    print(f"Error creating PDF: {e}")
    print("\nAlternative method:")
    print("  1. Open PROJECT_DOCUMENT.md in VS Code")
    print("  2. Press Ctrl+Shift+V to preview")
    print("  3. Right-click > Print > Save as PDF")
    sys.exit(1)
