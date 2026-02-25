"""File upload and retrieval service."""

from __future__ import annotations

import json
import io
from typing import List, Tuple
import logging

import numpy as np
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from .. import llm, models

logger = logging.getLogger(__name__)


def normalize_mime_type(mime_type: str | None) -> str:
    """
    Normalize MIME type for database storage.
    
    Some MIME types are very long (e.g., Office formats).
    This function shortens them for storage while preserving the essential info.
    
    Args:
        mime_type: Original MIME type from file upload
        
    Returns:
        Normalized MIME type (max 255 characters)
    """
    if not mime_type:
        return "application/octet-stream"
    
    # Common MIME type mappings
    mime_type_map = {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "application/docx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "application/xlsx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": "application/pptx",
        "application/vnd.ms-excel": "application/xls",
        "application/msword": "application/doc",
    }
    
    # Check for exact match
    if mime_type in mime_type_map:
        return mime_type_map[mime_type]
    
    # If not a known verbose type, just truncate to 255 chars if needed
    if len(mime_type) > 255:
        logger.warning(f"MIME type too long ({len(mime_type)} chars), truncating: {mime_type}")
        return mime_type[:255]
    
    return mime_type


async def extract_text_from_binary_file(data: bytes, filename: str) -> str:
    """
    Extract text from binary file formats (DOCX, XLSX, PDF, etc).
    
    Args:
        data: File binary data
        filename: Original filename for type detection
        
    Returns:
        Extracted text content
    """
    filename_lower = filename.lower()
    
    # DOCX files
    if filename_lower.endswith('.docx'):
        try:
            from docx import Document
            doc = Document(io.BytesIO(data))
            text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            return text if text.strip() else "[DOCX file with no extractable text]"
        except Exception as e:
            logger.warning(f"Failed to extract text from DOCX: {e}")
            return "[Error extracting DOCX content]"
    
    # XLSX/XLS files
    elif filename_lower.endswith(('.xlsx', '.xls')):
        try:
            import pandas as pd
            df = pd.read_excel(io.BytesIO(data))
            # Convert dataframe to readable text
            text = df.to_string()
            return text if text.strip() else "[Excel file with no data]"
        except Exception as e:
            logger.warning(f"Failed to extract text from Excel: {e}")
            return "[Error extracting Excel content]"
    
    # PDF files
    elif filename_lower.endswith('.pdf'):
        try:
            from PyPDF2 import PdfReader
            pdf = PdfReader(io.BytesIO(data))
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            return text if text.strip() else "[PDF file with no extractable text]"
        except Exception as e:
            logger.warning(f"Failed to extract text from PDF: {e}")
            return "[Error extracting PDF content]"
    
    # CSV files
    elif filename_lower.endswith('.csv'):
        try:
            import pandas as pd
            df = pd.read_csv(io.BytesIO(data))
            text = df.to_string()
            return text if text.strip() else "[CSV file with no data]"
        except Exception as e:
            logger.warning(f"Failed to extract text from CSV: {e}")
            return "[Error extracting CSV content]"
    
    # Text-based files - try standard encoding
    try:
        return data.decode("utf-8").strip()
    except UnicodeDecodeError:
        try:
            return data.decode("iso-8859-1").strip()
        except Exception:
            return "[Binary file - unable to extract text]"


async def process_file_upload(file: UploadFile) -> Tuple[str, List[str], List[List[float]], int]:
    """Read and process an uploaded file.

    The function extracts plain text from the file and splits it into
    reasonably sized chunks (~1000 characters) then generates a simple
    embedding for each chunk. The extracted text and embeddings are
    returned so they can be stored in the database.

    Args:
        file: A FastAPI UploadFile object representing the uploaded file.

    Returns:
        A tuple containing the extracted text, a list of text chunks,
        a list of embedding vectors, and the original file size in bytes.
    """
    # Read entire contents into memory. For production use you should
    # stream and chunk large files instead of loading them fully.
    data = await file.read()
    original_file_size = len(data)
    
    # Extract text from binary formats or decode text
    text_content = await extract_text_from_binary_file(data, file.filename or "unknown")
    
    # If the file is JSON parse and pretty print
    if file.filename and file.filename.lower().endswith(".json"):
        try:
            json_obj = json.loads(text_content)
            text_content = json.dumps(json_obj, indent=2)
        except Exception:
            pass
    
    # Split text into chunks of ~1000 characters
    chunk_size = 1000
    chunks: List[str] = [
        text_content[i : i + chunk_size] for i in range(0, len(text_content), chunk_size)
    ]
    # Generate simple embeddings for each chunk. The embed_text function
    # returns a list of vectors corresponding to the input list.
    embeddings = await llm.embed_text(chunks) if chunks else []
    return text_content, chunks, embeddings, original_file_size


async def add_file(
    db: AsyncSession, session_id: str, file: UploadFile
) -> models.UploadedFile:
    """Persist an uploaded file and its chunks to the database.

    Args:
        db: An async SQLAlchemy session.
        session_id: Identifier of the chat session to which the file belongs.
        file: The uploaded file.

    Returns:
        The persisted UploadedFile instance.
    """
    content_text, chunks, embeddings, file_size = await process_file_upload(file)
    new_file = models.UploadedFile(
        session_id=session_id,
        filename=file.filename,
        filetype=normalize_mime_type(file.content_type),
        size=file_size,
        content_text=content_text,
    )
    db.add(new_file)
    await db.flush()
    # Add chunks and embeddings
    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        chunk_model = models.FileChunk(
            file_id=new_file.id,
            chunk_index=idx,
            text=chunk,
            embedding=emb,
        )
        db.add(chunk_model)
    await db.commit()
    return new_file


async def retrieve_relevant_chunks(
    db: AsyncSession, session_id: str, query: str, top_k: int = 5
) -> List[models.FileChunk]:
    """Retrieve the most relevant text chunks for a given query.

    A simple cosine similarity over the naive embeddings is used to rank
    chunks. In production you should use a proper vector store and
    ANN search library. This function fetches all chunks for the
    session and computes similarity on the fly; it is therefore not
    suitable for large datasets.

    Args:
        db: Database session.
        session_id: Current chat session.
        query: User's question for which to retrieve context.
        top_k: Number of chunks to return.

    Returns:
        A list of FileChunk objects sorted by descending similarity.
    """
    # Compute embedding for the query
    query_emb = (await llm.embed_text([query]))[0]
    # Fetch all chunks for this session
    chunks = (
        await db.execute(
            models.FileChunk.__table__.join(models.UploadedFile).select().where(
                models.UploadedFile.session_id == session_id
            )
        )
    ).scalars().all()
    # Compute cosine similarity
    sims: List[Tuple[models.FileChunk, float]] = []
    query_vec = np.array(query_emb)
    for chunk in chunks:
        emb = chunk.embedding or []
        if not emb:
            sims.append((chunk, 0.0))
            continue
        chunk_vec = np.array(emb)
        # Avoid division by zero
        denom = (np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec)) or 1.0
        sim = float(np.dot(query_vec, chunk_vec) / denom)
        sims.append((chunk, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in sims[:top_k]]
