import httpx
import pdfplumber
import os
from langchain.docstore.document import Document
from typing import List
from urllib.parse import urlparse 

def table_to_markdown(table: list[list[str]]) -> str:
    """Converts a list of lists representing a table into a Markdown string."""
    markdown_table = ""
    if table:
        header = [str(cell).replace('\n', ' ') if cell is not None else "" for cell in table[0]]
        markdown_table += "| " + " | ".join(header) + " |\n"
        markdown_table += "| " + " | ".join(["---"] * len(header)) + " |\n"
        for row in table[1:]:
            body_row = [str(cell).replace('\n', ' ') if cell is not None else "" for cell in row]
            markdown_table += "| " + " | ".join(body_row) + " |\n"
    return markdown_table

def extract_chunks_from_pdf(pdf_path: str, source_url: str) -> List[Document]:
    """Extracts text and tables from a PDF, adding metadata for the source URL."""
    docs = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_number = i + 1
                text = page.extract_text()
                if text and text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={"page": page_number, "type": "text", "source": source_url}
                    ))
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        markdown_table = table_to_markdown(table)
                        docs.append(Document(
                            page_content=markdown_table,
                            metadata={"page": page_number, "type": "table", "source": source_url}
                        ))
    except Exception as e:
        print(f"Error processing PDF from {source_url}: {e}")
    return docs

async def process_document_from_url(url: str) -> List[Document]:
    """Orchestrator: Downloads a PDF from a URL and returns its content as Document chunks."""

    # --- THIS IS THE FIX ---
    # 1. Safely parse the URL to separate the path from the query string.
    parsed_url = urlparse(url)
    # 2. Get just the filename from the path component.
    pdf_filename = os.path.basename(parsed_url.path)
    # 3. Create a clean, valid temporary filename.
    temp_pdf_path = f"temp_{pdf_filename}"
    # --- END OF FIX ---

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
        
        # This will now work because temp_pdf_path is a valid filename
        with open(temp_pdf_path, "wb") as f:
            f.write(response.content)

        chunks = extract_chunks_from_pdf(temp_pdf_path, source_url=str(url))
        return chunks
    finally:
        # Clean up the temporary file after we're done with it
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)