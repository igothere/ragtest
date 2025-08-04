import os
import sys
import re
import traceback
import fitz  # PyMuPDF
import docx
import pandas as pd
import psycopg2
from typing import List, Dict
from sentence_transformers import SentenceTransformer

TARGET_DIR = "/home/eden/rag/docs"
MODEL_NAME = "nlpai-lab/KURE-v1"
DB_CONFIG = {
    "host": "localhost", "port": "5432", "dbname": "ragtest",
    "user": "eden", "password": "qwer123"
}

def process_file_to_hlm_chunks(file_path: str) -> List[Dict]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return parse_pdf_to_hlm(file_path)
    elif ext == ".docx":
        return parse_docx_to_hlm(file_path)
    elif ext in [".md", ".txt"]:
        return parse_text_to_hlm(file_path)
    elif ext in [".xls", ".xlsx"]:
        return parse_excel_to_hlm(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def parse_pdf_to_hlm(file_path: str) -> List[Dict]:
    doc = fitz.open(file_path)
    chunks = []
    current_section = ""
    for i, page in enumerate(doc):
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: b[1])
        for block in blocks:
            text = block[4].strip()
            if not text:
                continue
            if re.match(r"^\d+(\.\d+)*\s", text):
                current_section = text.strip()
                depth = text.count(".") + 1
                chunks.append({
                    "title": current_section,
                    "content": "",
                    "depth": depth,
                    "metadata": {"section": current_section, "source": os.path.basename(file_path), "page": i+1}
                })
            elif chunks:
                chunks[-1]["content"] += " " + text
    return [c for c in chunks if c["content"].strip()]

def parse_docx_to_hlm(file_path: str) -> List[Dict]:
    doc = docx.Document(file_path)
    chunks = []
    current_section = ""
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if re.match(r"^\d+(\.\d+)*\s", text):
            current_section = text
            depth = text.count(".") + 1
            chunks.append({
                "title": current_section,
                "content": "",
                "depth": depth,
                "metadata": {"section": current_section, "source": os.path.basename(file_path)}
            })
        elif chunks:
            chunks[-1]["content"] += " " + text
    return [c for c in chunks if c["content"].strip()]

def parse_text_to_hlm(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    chunks = []
    current_section = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r"^#+\s", line):
            depth = line.count("#")
            current_section = re.sub(r"^#+\s", "", line)
            chunks.append({
                "title": current_section,
                "content": "",
                "depth": depth,
                "metadata": {"section": current_section, "source": os.path.basename(file_path)}
            })
        elif re.match(r"^\d+(\.\d+)*\s", line):
            depth = line.count(".") + 1
            current_section = line
            chunks.append({
                "title": current_section,
                "content": "",
                "depth": depth,
                "metadata": {"section": current_section, "source": os.path.basename(file_path)}
            })
        elif chunks:
            chunks[-1]["content"] += " " + line
    return [c for c in chunks if c["content"].strip()]

def parse_excel_to_hlm(file_path: str) -> List[Dict]:
    xl = pd.ExcelFile(file_path)
    chunks = []
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        text = df.to_string(index=False)
        chunks.append({
            "title": sheet_name,
            "content": text,
            "depth": 1,
            "metadata": {"section": sheet_name, "source": os.path.basename(file_path)}
        })
    return chunks