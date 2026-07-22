# Docling VLM Service with OpenRouter

Microservice berbasis FastAPI yang membungkus kapabilitas **Docling VLM Pipeline**. 
Service ini menerima file PDF, lalu menggunakan Vision LLM via OpenRouter (seperti `gemini-2.0-flash-001`, `llama-3.2-11b-vision-instruct`, dll) untuk mengekstrak isi PDF beserta gambar/tabel ke dalam format **Markdown yang bersih**.

## Prasyarat

- Python 3.10+
- Akun OpenRouter dan API Key (Dapatkan di [OpenRouter](https://openrouter.ai/))

## Setup Lokal (menggunakan `uv` atau `pip`)

1. **Masuk ke folder service**:
   ```bash
   cd docling_service
   ```

2. **Buat file `.env`**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` dan masukkan `OPENROUTER_API_KEY` kamu.

3. **Install dependencies**:
   Jika menggunakan `uv`:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```
   Atau dengan pip standar:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Jalankan Service**:
   ```bash
   python main.py
   ```
   Service akan berjalan di `http://0.0.0.0:8080`.

## Cara Penggunaan (API Endpoint)

### 1. Cek Status
```bash
curl http://localhost:8080/
```

### 2. Convert PDF ke Markdown
Gunakan endpoint `/convert` dengan form-data file PDF.
```bash
curl -X POST "http://localhost:8080/convert" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/dokumen.pdf"
```

Response-nya berupa JSON:
```json
{
  "filename": "dokumen.pdf",
  "status": "success",
  "model_used": "google/gemini-2.0-flash-001",
  "markdown": "# Judul Dokumen\n\nIsi dari dokumen yang diekstrak dengan sangat baik oleh VLM..."
}
```

## Integrasi dengan LightRAG

Setelah service ini menyala, kamu bisa menghubungkannya ke LightRAG. Cukup kirim file PDF ke service ini, lalu ambil field `markdown` dari response-nya, dan masukkan ke fungsi `insert` di LightRAG.

Contoh integrasi di script lain:
```python
import requests
from lightrag import LightRAG

# ... inisialisasi rag ...

pdf_path = "sample.pdf"
response = requests.post(
    "http://localhost:8080/convert",
    files={"file": open(pdf_path, "rb")}
)

if response.status_code == 200:
    markdown_content = response.json()["markdown"]
    # Masukkan ke LightRAG
    rag.insert(markdown_content)
else:
    print("Gagal convert PDF:", response.text)
```
