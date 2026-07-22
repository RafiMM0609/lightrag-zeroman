import os
import shutil
import tempfile
import uuid
import io
import time
import json
import asyncio
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any
import requests


from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmConvertOptions,
    VlmPipelineOptions,
)
from docling.datamodel.vlm_engine_options import (
    ApiVlmEngineOptions,
    VlmEngineType,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("docling_service")

app = FastAPI(
    title="Docling VLM Service for LightRAG",
    description="Microservice to convert PDF documents to Markdown/JSON ZIP bundle using Vision LLMs via OpenRouter with Standard Pipeline Fallback"
)

# Konfigurasi OpenRouter dari Environment Variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("OPENROUTER_MODEL", "google/gemini-3.5-flash")

# Preset selection: "granite_docling" is specifically for Granite-Docling models
# producing XML DocTags. Generic Vision LLMs (Gemini, Llama, Qwen, etc.) require
# a markdown-based preset like "qwen", "phi4", or "pixtral".
VLM_PRESET = os.getenv("DOCLING_VLM_PRESET")
if not VLM_PRESET:
    if "granite-docling" in MODEL_NAME.lower():
        VLM_PRESET = "granite_docling"
    else:
        VLM_PRESET = "qwen"

# Thread pool for asynchronous background processing
executor = ThreadPoolExecutor(max_workers=4)

# In-memory storage for async tasks
TASKS: Dict[str, Dict[str, Any]] = {}

def get_vlm_doc_converter():
    if not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="OPENROUTER_API_KEY is not set in environment variables."
        )

    logger.info(f"Using VLM preset '{VLM_PRESET}' for model '{MODEL_NAME}'")
    vlm_options = VlmConvertOptions.from_preset(
        VLM_PRESET,
        engine_options=ApiVlmEngineOptions(
            runtime_type=VlmEngineType.API,
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://github.com/HKUDS/LightRAG", 
                "X-Title": "LightRAG Docling Service",
            },
            params={
                "model": MODEL_NAME,
                "max_tokens": 8192,
                "temperature": 0.0,
            },
            timeout=120,
        ),
    )

    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_options,
        enable_remote_services=True,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                pipeline_cls=VlmPipeline,
            )
        }
    )

def get_standard_doc_converter():
    return DocumentConverter()

def fix_docling_dict_for_lightrag(json_dict: dict):
    """
    Ensures that DoclingDocument dict can be properly parsed by LightRAG DoclingIRBuilder.
    1. Ensures body dict exists and has content_layer="body".
    2. Ensures all texts, tables, pictures have content_layer="body".
    3. Traverses existing group/children references to find reachable items.
    4. Appends any unreferenced texts, tables, pictures directly to body.children.
    """
    if "body" not in json_dict or not isinstance(json_dict["body"], dict):
        json_dict["body"] = {"self_ref": "#/body", "children": [], "content_layer": "body", "name": "_root_", "label": "unspecified"}

    body = json_dict["body"]
    body["content_layer"] = "body"
    children = body.setdefault("children", [])

    reachable_refs = set()
    ref_index = {}
    for key in ("texts", "tables", "pictures", "groups"):
        items = json_dict.get(key)
        if isinstance(items, list):
            for i, item in enumerate(items):
                ref_str = f"#/{key}/{i}"
                if isinstance(item, dict):
                    ref_index[ref_str] = item
                    item["content_layer"] = "body"

    def collect_refs(ref_node):
        ref_str = ""
        if isinstance(ref_node, dict):
            ref_str = ref_node.get("$ref") or ref_node.get("ref") or ""
        elif isinstance(ref_node, str):
            ref_str = ref_node
        
        if ref_str and ref_str not in reachable_refs:
            reachable_refs.add(ref_str)
            item = ref_index.get(ref_str)
            if item and isinstance(item, dict):
                for child_ref in item.get("children") or []:
                    collect_refs(child_ref)

    for child in children:
        collect_refs(child)

    added_count = 0
    for key in ("texts", "tables", "pictures"):
        items = json_dict.get(key)
        if isinstance(items, list):
            for i, item in enumerate(items):
                ref_str = f"#/{key}/{i}"
                if ref_str not in reachable_refs:
                    children.append({"$ref": ref_str})
                    reachable_refs.add(ref_str)
                    added_count += 1

    if added_count > 0:
        logger.info(f"ℹ️ Injected {added_count} unreferenced text/table/picture item(s) into body.children for LightRAG IR Builder.")

def translate_tables_to_narrative(markdown_text: str) -> str:
    """
    Calls OpenRouter LLM to translate any tables in the markdown text into narrative rules.
    Returns the narrative text (or empty string if failed/no tables).
    """
    if "|" not in markdown_text:
        return ""
        
    if not OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY is not set. Skipping table translation.")
        return ""

    logger.info("Tables found in markdown. Calling OpenRouter to translate them to narrative...")
    
    prompt = (
        "Tugas Anda adalah menganalisis teks Markdown berikut dan mendeteksi semua tabel di dalamnya.\n"
        "Untuk setiap tabel yang Anda temukan (terutama tabel persetujuan/matriks kewenangan/checkbox), "
        "terjemahkan seluruh isi tabel tersebut menjadi penjelasan tertulis (narasi aturan deklaratif) dalam Bahasa Indonesia.\n"
        "Ikuti aturan berikut:\n"
        "1. Jelaskan batasan nominal/value secara eksplisit (contoh: jika nominal antara 0 sampai 100 Juta, maka membutuhkan approval dari Head of AP Management).\n"
        "2. Sebutkan nama peran approver secara lengkap (seperti 'Head of AP Management' sebagai Approver 1, 'Deputy VP Finance' sebagai Approver 2, dst).\n"
        "3. Jika ada simbol titik '•' atau centang pada kolom peran, artinya peran tersebut diwajibkan memberikan approval.\n"
        "4. Tulis penjelasan Anda dalam bentuk paragraf atau poin-poin yang mudah dipahami oleh sistem RAG. Hindari penggunaan tabel markdown baru di dalam penjelasan Anda.\n"
        "5. Cukup kembalikan hasil penjelasan aturan naratif tersebut dalam format Markdown. Jangan sertakan teks intro seperti 'Berikut adalah penjelasan...' atau kata penutup lainnya.\n\n"
        "Berikut adalah dokumen Markdown:\n\n"
        f"{markdown_text}"
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/HKUDS/LightRAG",
        "X-Title": "LightRAG Docling Table-to-Text Postprocessor",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        if response.status_code == 200:
            translated = response.json()["choices"][0]["message"]["content"].strip()
            logger.info("Successfully translated markdown tables into narrative rules.")
            return translated
        else:
            logger.error(f"Failed to call OpenRouter for table translation: {response.status_code} - {response.text}")
            return ""
    except Exception as e:
        logger.error(f"Error calling OpenRouter for table translation: {e}", exc_info=True)
        return ""

def append_narrative_to_document(json_dict: dict, markdown_text: str, narrative: str) -> tuple[dict, str]:
    if not narrative.strip():
        return json_dict, markdown_text
        
    # 1. Update Markdown
    augmented_markdown = (
        f"{markdown_text}\n\n"
        f"## Interpretasi Aturan Tabel (Otomatis)\n\n"
        f"{narrative}"
    )
    
    # 2. Update JSON dict
    # Ensure texts array exists
    texts = json_dict.setdefault("texts", [])
    body = json_dict.setdefault("body", {})
    body_children = body.setdefault("children", [])
    
    # We will add a section header first
    header_idx = len(texts)
    header_ref = f"#/texts/{header_idx}"
    texts.append({
        "text": "Interpretasi Aturan Tabel (Otomatis)",
        "label": "section_header",
        "level": 1,
        "content_layer": "body",
        "self_ref": header_ref
    })
    body_children.append({"$ref": header_ref})
    
    # Split the narrative into paragraphs and add them
    paragraphs = [p.strip() for p in narrative.split("\n\n") if p.strip()]
    for p in paragraphs:
        p_idx = len(texts)
        p_ref = f"#/texts/{p_idx}"
        
        # Check if the paragraph itself is a header (starts with #)
        if p.startswith("#"):
            # Count the number of '#' to determine the level
            level = len(p) - len(p.lstrip("#"))
            clean_text = p.lstrip("#").strip()
            texts.append({
                "text": clean_text,
                "label": "section_header",
                "level": level,
                "content_layer": "body",
                "self_ref": p_ref
            })
        else:
            texts.append({
                "text": p,
                "label": "text",
                "content_layer": "body",
                "self_ref": p_ref
            })
        body_children.append({"$ref": p_ref})
        
    logger.info(f"Appended narrative section containing {len(paragraphs)} paragraphs to JSON structure.")
    return json_dict, augmented_markdown

def process_conversion_task(task_id: str, temp_file_path: Path, original_filename: str):
    start_time = time.time()
    logger.info(f"[Task {task_id}] 🚀 Started processing file: '{original_filename}' with model '{MODEL_NAME}'")
    try:
        TASKS[task_id]["status"] = "started"
        
        converter = get_vlm_doc_converter()
        logger.info(f"[Task {task_id}] ⏳ Running Docling VLM conversion pipeline via OpenRouter...")
        result = converter.convert(temp_file_path)

        json_dict = result.document.export_to_dict()
        md_text = result.document.export_to_markdown()

        texts_count = len(json_dict.get("texts", []))
        
        # Fallback to Standard Pipeline if VLM returned empty result
        if not md_text.strip() or texts_count == 0:
            logger.warning(
                f"[Task {task_id}] ⚠️ VLM Pipeline returned 0 texts (model response format mismatch). "
                f"Falling back to Docling Standard Pipeline..."
            )
            std_converter = get_standard_doc_converter()
            result = std_converter.convert(temp_file_path)
            json_dict = result.document.export_to_dict()
            md_text = result.document.export_to_markdown()
            texts_count = len(json_dict.get("texts", []))
            logger.info(f"[Task {task_id}] ✅ Standard Pipeline extracted {texts_count} text elements.")

        # Ensure body.children is populated and items carry content_layer="body"
        fix_docling_dict_for_lightrag(json_dict)

        # Translate any tables into narrative rules and append to both JSON and Markdown
        narrative = translate_tables_to_narrative(md_text)
        json_dict, md_text = append_narrative_to_document(json_dict, md_text, narrative)

        stem = Path(original_filename).stem
        
        # Package into a ZIP in memory to match LightRAG DoclingRawClient expectation
        logger.info(f"[Task {task_id}] 📦 Packaging outputs ({stem}.json, {stem}.md) into ZIP bundle...")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            json_bytes = json.dumps(json_dict, ensure_ascii=False, indent=2).encode("utf-8")
            zf.writestr(f"{stem}.json", json_bytes)
            zf.writestr(f"{stem}.md", md_text.encode("utf-8"))

        zip_bytes = zip_buffer.getvalue()
        elapsed = time.time() - start_time

        TASKS[task_id]["status"] = "success"
        TASKS[task_id]["zip_bytes"] = zip_bytes
        TASKS[task_id]["filename"] = f"{stem}.zip"
        
        logger.info(f"[Task {task_id}] ✅ Conversion succeeded in {elapsed:.2f}s! ZIP size: {len(zip_bytes)} bytes.")

    except Exception as e:
        elapsed = time.time() - start_time
        err_msg = str(e)
        logger.error(f"[Task {task_id}] ❌ Conversion failed after {elapsed:.2f}s: {err_msg}", exc_info=True)
        TASKS[task_id]["status"] = "failure"
        TASKS[task_id]["error"] = err_msg
    finally:
        if temp_file_path.exists():
            try:
                os.remove(temp_file_path)
            except OSError:
                pass

@app.get("/")
def root():
    return {
        "message": "Docling VLM Service for LightRAG is running.", 
        "model_configured": MODEL_NAME,
        "endpoints": [
            "POST /convert",
            "POST /v1/convert/file/async",
            "GET /v1/status/poll/{task_id}",
            "GET /v1/result/{task_id}"
        ]
    }

@app.post("/convert")
async def convert_document(file: UploadFile = File(...)):
    """
    Upload a PDF file to extract its content to Markdown (Synchronous).
    """
    filename = file.filename or "uploaded_doc.pdf"
    if not filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = Path(temp_file.name)

        logger.info(f"📥 Received synchronous /convert request for '{filename}'")
        converter = get_vlm_doc_converter()
        result = converter.convert(temp_file_path)
        markdown_text = result.document.export_to_markdown()

        if not markdown_text.strip():
            logger.warning("⚠️ VLM returned empty markdown, using Standard Pipeline fallback...")
            converter = get_standard_doc_converter()
            result = converter.convert(temp_file_path)
            markdown_text = result.document.export_to_markdown()

        if temp_file_path.exists():
            os.remove(temp_file_path)

        # Translate any tables into narrative rules and append to Markdown
        narrative = translate_tables_to_narrative(markdown_text)
        if narrative:
            markdown_text = (
                f"{markdown_text}\n\n"
                f"## Interpretasi Aturan Tabel (Otomatis)\n\n"
                f"{narrative}"
            )

        return JSONResponse(content={
            "filename": filename,
            "status": "success",
            "model_used": MODEL_NAME,
            "markdown": markdown_text
        })

    except Exception as e:
        if 'temp_file_path' in locals() and temp_file_path.exists():
            os.remove(temp_file_path)
        logger.error(f"❌ Synchronous convert failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/convert/file/async")
async def convert_file_async(files: UploadFile = File(...)):
    """
    Docling Serve compatible async convert endpoint.
    """
    filename = files.filename or "uploaded_doc.pdf"
    if not filename.lower().endswith('.pdf'):
        logger.warning(f"Rejected non-PDF upload attempt: {filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(files.file, temp_file)
        temp_file_path = Path(temp_file.name)

    task_id = str(uuid.uuid4())
    TASKS[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "zip_bytes": None,
        "filename": None,
        "error": None
    }

    logger.info(f"📥 Received file conversion request: '{filename}' -> Created Task ID: {task_id}")

    loop = asyncio.get_running_loop()
    loop.run_in_executor(executor, process_conversion_task, task_id, temp_file_path, filename)

    return {"task_id": task_id, "status": "pending"}

@app.get("/v1/status/poll/{task_id}")
async def poll_task_status(task_id: str, wait: int = Query(default=0)):
    """
    Docling Serve compatible long-polling endpoint.
    If `wait` > 0, waits asynchronously until task finishes or timeout expires.
    """
    task = TASKS.get(task_id)
    if not task:
        logger.warning(f"Poll requested for non-existent task_id: {task_id}")
        raise HTTPException(status_code=404, detail="Task not found")

    if wait > 0 and task["status"] in ("pending", "started"):
        start_wait = time.time()
        logger.info(f"[Task {task_id}] 💤 Client long-polling (wait={wait}s)... Current status: {task['status']}")
        while time.time() - start_wait < wait:
            await asyncio.sleep(0.5)
            if task["status"] not in ("pending", "started"):
                break

    return {
        "task_id": task_id,
        "task_status": task["status"],
        "error_message": task["error"]
    }

@app.get("/v1/result/{task_id}")
async def get_task_result(task_id: str):
    """
    Docling Serve compatible result retrieval endpoint.
    Returns ZIP file containing converted document outputs.
    """
    task = TASKS.get(task_id)
    if not task:
        logger.warning(f"Result requested for non-existent task_id: {task_id}")
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task["status"] != "success":
        logger.warning(f"[Task {task_id}] Result requested but task status is '{task['status']}'")
        raise HTTPException(status_code=400, detail=f"Task status is {task['status']}: {task.get('error')}")

    zip_bytes = task.get("zip_bytes")
    if not zip_bytes:
        raise HTTPException(status_code=500, detail="ZIP bundle content missing.")

    filename = task.get("filename", "result.zip")
    logger.info(f"[Task {task_id}] 📤 Serving ZIP result bundle to LightRAG ({len(zip_bytes)} bytes)...")
    
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
