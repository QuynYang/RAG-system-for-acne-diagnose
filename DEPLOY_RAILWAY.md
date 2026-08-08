# Hướng dẫn deploy Acne Advisor AI lên Railway (A → Z)

> Repo: [QuynYang/RAG-system-for-acne-diagnose](https://github.com/QuynYang/RAG-system-for-acne-diagnose)  
> Thời gian ước tính: **2–4 giờ** lần đầu (build Docker ~10–20 phút, ingest KB ~30–90 phút).

---

## Mục lục

1. [Kiến trúc hệ thống](#1-kiến-trúc-hệ-thống)
2. [Chuẩn bị trước khi deploy](#2-chuẩn-bị-trước-khi-deploy)
3. [Push file deploy lên GitHub](#3-push-file-deploy-lên-github)
4. [Tạo tài khoản Railway & project](#4-tạo-tài-khoản-railway--project)
5. [Deploy backend (FastAPI)](#5-deploy-backend-fastapi)
6. [Thêm PostgreSQL & Redis](#6-thêm-postgresql--redis)
7. [Thiết lập Qdrant & Neo4j (managed cloud — khuyên dùng)](#7-thiết-lập-qdrant--neo4j-managed-cloud--khuyên-dùng)
8. [Cấu hình biến môi trường backend](#8-cấu-hình-biến-môi-trường-backend)
9. [Khởi tạo schema & ingest knowledge base](#9-khởi-tạo-schema--ingest-knowledge-base)
10. [Deploy frontend (React/Vite)](#10-deploy-frontend-reactvite)
11. [Kiểm tra sau deploy](#11-kiểm-tra-sau-deploy)
12. [Phương án B: tự host Qdrant/Neo4j trên Railway](#12-phương-án-b-tự-host-qdrantneo4j-trên-railway)
13. [Troubleshooting](#13-troubleshooting)
14. [Checklist bảo mật](#14-checklist-bảo-mật)

---

## 1. Kiến trúc hệ thống

Dự án **không phải 1 app đơn** — cần **6 service** khi chạy production:

```mermaid
flowchart TB
    subgraph Railway["Railway Project"]
        FE["Frontend\nReact/Vite + serve"]
        BE["Backend\nFastAPI + LangGraph"]
        PG[("PostgreSQL\nchat history")]
        RD[("Redis\ncache")]
    end

    subgraph Cloud["Managed Cloud (khuyên dùng)"]
        QD[("Qdrant Cloud\nvector DB")]
        NJ[("Neo4j Aura\nknowledge graph")]
    end

    subgraph External["External APIs"]
        GM["Google Gemini\nLLM + Embedding"]
        LP["LlamaParse\nPDF → Markdown"]
    end

    User((Người dùng)) --> FE
    FE -->|HTTPS /chat| BE
    BE --> PG
    BE --> RD
    BE --> QD
    BE --> NJ
    BE --> GM
    Ingest["Ingest (máy local)"] --> QD
    Ingest --> NJ
    Ingest --> GM
    Ingest --> LP
```

| Thành phần | Vai trò | Railway native? |
|---|---|---|
| **FastAPI** (`src/api/app.py`) | API chat, health, retrieval | ✅ Deploy Dockerfile |
| **PostgreSQL** | Lưu chat sessions/messages | ✅ Plugin Database 1-click |
| **Redis** | Semantic answer cache | ✅ Plugin Database 1-click |
| **Qdrant** | Hybrid dense + sparse BM25 retrieval | ❌ → Qdrant Cloud hoặc Docker image |
| **Neo4j** | Entity knowledge graph | ❌ → Neo4j Aura hoặc Docker image |
| **React/Vite** (`src/frontend/`) | Giao diện chat | ✅ Service riêng, Dockerfile |

### Luồng request chat (tóm tắt)

1. User gửi tin nhắn qua Frontend → `POST /chat`
2. Backend kiểm tra Redis cache
3. Hybrid retrieval: Qdrant (chunks + entities) + Neo4j (graph facts)
4. Rerank (rule-based trên Railway — không có GPU/model local)
5. Gemini generate câu trả lời
6. Lưu history vào PostgreSQL → trả về Frontend

### 3 điểm nghẽn quan trọng

| # | Vấn đề | Giải pháp |
|---|---|---|
| 1 | Reranker CUDA local (`C:/Models/...`) | Để `SEMANTIC_RERANK_MODEL_PATH=` rỗng → tự fallback rule-based |
| 2 | `sample_data/` (PDF) **không có trên GitHub** | Ingest chạy **từ máy local**, trỏ env vào DB production |
| 3 | Railway Postgres trả `postgresql://` | Code đã tự convert sang `postgresql+asyncpg://` |

---

## 2. Chuẩn bị trước khi deploy

### 2.1. Tài khoản & API keys cần có

| Dịch vụ | Link đăng ký | Dùng để |
|---|---|---|
| **GitHub** | github.com | Repo source |
| **Railway** | [railway.app](https://railway.app) | Hosting |
| **Google AI Studio** | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | `GOOGLE_API_KEY` (LLM + embedding) |
| **Qdrant Cloud** | [cloud.qdrant.io](https://cloud.qdrant.io) | Vector DB free tier |
| **Neo4j Aura** | [console.neo4j.io](https://console.neo4j.io) | Graph DB free tier |
| **LlamaCloud** (tuỳ chọn) | [cloud.llamaindex.ai](https://cloud.llamaindex.ai) | Parse PDF khi ingest |

### 2.2. Trên máy local cần có

- **Python 3.11+** + venv đã cài dependency (`pip install -r requirements.txt`)
- Thư mục **`sample_data/`** với PDF/JSON (file lớn, không commit Git — bạn phải có trên máy)
- **Node.js 20+** (cho Railway CLI, tuỳ chọn)
- **Git**

### 2.3. File deploy đã có trong repo

Sau bước push (mục 3), repo sẽ có:

```
RAG-system-for-acne-diagnose/
├── Dockerfile              ← Backend
├── .dockerignore
├── railway.json
└── src/frontend/
    └── Dockerfile          ← Frontend
```

---

## 3. Push file deploy lên GitHub

Mở **PowerShell** tại thư mục project:

```powershell
cd e:\code\KLTN\web-giang\RAG-system-for-acne-diagnose

git status
git add Dockerfile .dockerignore railway.json src/frontend/Dockerfile DEPLOY_RAILWAY.md
git add src/database/connection.py src/api/preflight.py scripts/init_schema.py scripts/init_chat_schema.py

git commit -m "chore: add Railway deploy config and production fixes"
git push origin main
```

---

## 4. Tạo tài khoản Railway & project

### Bước 4.1 — Đăng ký / đăng nhập

1. Mở trình duyệt → [https://railway.app](https://railway.app)
2. Bấm **Login** (góc trên phải)
3. Chọn **Login with GitHub**
4. GitHub hỏi quyền → bấm **Authorize Railway**

### Bước 4.2 — Tạo project mới

1. Trên Dashboard Railway → bấm **+ New Project** (hoặc **New**)
2. Chọn **Deploy from GitHub repo**
3. Nếu lần đầu: Railway yêu cầu **Configure GitHub App** → bấm **Configure GitHub App** → chọn repo `QuynYang/RAG-system-for-acne-diagnose` → **Save**
4. Quay lại Railway → chọn repo **`RAG-system-for-acne-diagnose`**
5. Railway tạo project + **1 service đầu tiên** (backend) và bắt đầu build

### Bước 4.3 — Đặt tên service backend

1. Bấm vào **service card** vừa tạo (thường tên = tên repo)
2. Bấm tên service ở đầu tab → đổi thành **`backend`** (dễ quản lý)
3. Bấm **Enter** để lưu

---

## 5. Deploy backend (FastAPI)

### Bước 5.1 — Kiểm tra Settings

1. Trong service **backend** → tab **Settings** (biểu tượng bánh răng)
2. Mục **Source**:
   - **Repository**: `QuynYang/RAG-system-for-acne-diagnose` ✅
   - **Branch**: `main` ✅
   - **Root Directory**: để **trống** (repo root, vì `Dockerfile` ở root)
3. Mục **Build**:
   - **Builder**: phải là **Dockerfile** (Railway đọc `railway.json`)
   - **Dockerfile Path**: `Dockerfile`
4. Mục **Deploy**:
   - **Healthcheck Path**: `/health` (từ `railway.json`)

> ⏱ Build lần đầu mất **10–20 phút** vì `sentence-transformers` kéo theo PyTorch (~2–3 GB).

### Bước 5.2 — Theo dõi build

1. Tab **Deployments** → bấm deployment đang **Building**
2. Xem **Build Logs** — đợi đến khi thấy **Build successful**
3. Nếu **Failed** → xem mục [Troubleshooting](#13-troubleshooting)

> Lúc này backend **chưa chạy được** vì chưa có Postgres/Redis/Qdrant/Neo4j — bình thường.

---

## 6. Thêm PostgreSQL & Redis

Trong **project view** (canvas hiển thị các service):

### Bước 6.1 — PostgreSQL

1. Bấm **+ Create** (hoặc **+ New** ở góc project)
2. Chọn **Database** → **Add PostgreSQL**
3. Railway tạo service **Postgres** — đợi status **Active**
4. Bấm service **Postgres** → tab **Variables** → ghi nhớ biến **`DATABASE_URL`**
   - Dạng: `postgresql://postgres:xxxxx@xxx.railway.internal:5432/railway`

### Bước 6.2 — Redis

1. Lại bấm **+ Create** → **Database** → **Add Redis**
2. Railway tạo service **Redis** — đợi **Active**
3. Tab **Variables** → ghi nhớ **`REDIS_URL`**

### Bước 6.3 — (Tuỳ chọn) Đổi tên service

- Postgres → đổi tên **`postgres`**
- Redis → đổi tên **`redis`**

---

## 7. Thiết lập Qdrant & Neo4j (managed cloud — khuyên dùng)

> Khuyên dùng cho project học thuật/portfolio: miễn phí, ổn định, không lo mất data khi redeploy.

### 7.1 — Qdrant Cloud

1. Mở [https://cloud.qdrant.io](https://cloud.qdrant.io) → **Sign Up** / **Log In**
2. Bấm **Create Cluster**
3. Chọn **Free tier** → chọn region gần bạn (vd: `us-east-1`)
4. Đặt tên cluster: `acne-advisor` → **Create**
5. Đợi cluster **Ready** (~2 phút)
6. Vào cluster → tab **API Keys** → **Create API Key** → copy key
7. Ghi lại:
   - **`QDRANT_URL`**: `https://xxxxxxxx.us-east-1-1.aws.cloud.qdrant.io` (tab Cluster → **Endpoint**)
   - **`QDRANT_API_KEY`**: key vừa tạo

### 7.2 — Neo4j Aura Free

1. Mở [https://console.neo4j.io](https://console.neo4j.io) → đăng ký / đăng nhập
2. Bấm **New Instance**
3. Chọn **AuraDB Free**
4. Instance name: `acne-advisor` → **Create**
5. **Download credentials** (hoặc copy password hiện ra) — **lưu ngay**, không xem lại được!
6. Ghi lại:
   - **`NEO4J_URI`**: `neo4j+s://xxxxxxxx.databases.neo4j.io`
   - **`NEO4J_USERNAME`**: `neo4j`
   - **`NEO4J_PASSWORD`**: password vừa tạo

---

## 8. Cấu hình biến môi trường backend

1. Bấm service **backend** → tab **Variables**
2. Bấm **+ New Variable** → **Raw Editor** (hoặc thêm từng biến)
3. Dán block dưới, **thay các giá trị `<...>`** bằng giá trị thật:

```env
# === App ===
APP_ENV=production
API_WORKERS=1
API_RELOAD=false
LOG_LEVEL=INFO
PREFLIGHT_REQUIRE_OLLAMA=false

# === CORS — cập nhật SAU khi có domain frontend (bước 10) ===
CORS_ALLOW_ORIGINS=https://<frontend-domain>.up.railway.app

# === PostgreSQL — tham chiếu service Railway ===
DATABASE_URL=${{postgres.DATABASE_URL}}

# === Redis ===
REDIS_URL=${{redis.REDIS_URL}}

# === Qdrant Cloud ===
QDRANT_URL=https://<cluster-id>.cloud.qdrant.io
QDRANT_API_KEY=<api-key-qdrant>
QDRANT_COLLECTION_NAME=acne_knowledge
CHUNK_QDRANT_COLLECTION_NAME=acne_knowledge
ENTITY_QDRANT_COLLECTION_NAME=acne_entities_v1

# === Neo4j Aura ===
NEO4J_URI=neo4j+s://<instance-id>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<password-neo4j>

# === Google Gemini (BẮT BUỘC) ===
GOOGLE_API_KEY=<google-api-key>
GOOGLE_MODEL=gemini-2.0-flash
EMBEDDING_PROVIDER=google
EMBEDDING_MODEL=models/gemini-embedding-2
EMBEDDING_DIMENSIONS=3072

# === LlamaParse (cần khi ingest PDF) ===
LLAMA_CLOUD_API_KEY=<llama-cloud-key>

# === Reranker — để trống, fallback rule-based ===
SEMANTIC_RERANK_MODEL_PATH=
SEMANTIC_RERANK_ALLOW_FALLBACK=true

# === Offline policy — tắt trên cloud ===
HF_HUB_OFFLINE=0
TRANSFORMERS_OFFLINE=0

# === Production safety ===
PHASE2_DEBUG_METADATA=false
CACHE_ENABLED=true
```

4. Bấm **Save** (hoặc **Deploy** nếu Railway hỏi redeploy)

### Giải thích cú pháp `${{service.VAR}}`

Railway tự nối biến giữa các service trong cùng project:

| Biến | Cú pháp | Ý nghĩa |
|---|---|---|
| Postgres URL | `${{postgres.DATABASE_URL}}` | Tự cập nhật khi Railway đổi host nội bộ |
| Redis URL | `${{redis.REDIS_URL}}` | Tương tự |

> **Lưu ý tên service**: `${{postgres.DATABASE_URL}}` phải khớp **tên service Postgres** trên canvas. Nếu bạn đặt tên khác (vd `Postgres`), dùng `${{Postgres.DATABASE_URL}}`.

### Bước 8.1 — Tạo public domain cho backend

1. Service **backend** → tab **Settings** → mục **Networking**
2. Bấm **Generate Domain**
3. Railway tạo URL dạng: `https://backend-production-xxxx.up.railway.app`
4. **Copy URL này** — dùng cho frontend (`VITE_API_URL`) và test `/health`

---

## 9. Khởi tạo schema & ingest knowledge base

> ⚠️ **Quan trọng**: `sample_data/` (PDF knowledge base) **không có trên GitHub** (`.gitignore`).  
> Bạn **phải chạy ingest từ máy local** có thư mục `sample_data/`.

### 9.1 — Khởi tạo schema (Postgres + Qdrant collections)

**Cách A — Railway CLI (khuyên dùng cho init schema):**

```powershell
# Cài Railway CLI (cần Node.js)
npm install -g @railway/cli

# Đăng nhập
railway login

# Link project (chọn đúng project vừa tạo)
cd e:\code\KLTN\web-giang\RAG-system-for-acne-diagnose
railway link

# Chọn service backend khi được hỏi
railway service

# Chạy init schema (inject env từ Railway)
railway run python scripts/init_schema.py
railway run python scripts/init_chat_schema.py
```

**Cách B — Chạy local với file `.env.production`:**

Tạo file `.env.production` (KHÔNG commit) copy các biến từ Railway Variables, rồi:

```powershell
cd e:\code\KLTN\web-giang\RAG-system-for-acne-diagnose
.\venv\Scripts\Activate.ps1

$env:DOTENV_PATH = ".env.production"   # hoặc copy thủ công vào .env tạm
python scripts/init_schema.py
python scripts/init_chat_schema.py
```

Kết quả mong đợi:
```
✅ Schema initialisation completed successfully.
✅ chat_sessions / chat_messages tables ready.
```

### 9.2 — Ingest knowledge base (chạy từ máy local)

Đảm bảo máy local có:
- Thư mục `sample_data/` với PDF (3 file theo README)
- `.env` trỏ **QDRANT_URL, NEO4J_URI, GOOGLE_API_KEY** vào production

```powershell
cd e:\code\KLTN\web-giang\RAG-system-for-acne-diagnose
.\venv\Scripts\Activate.ps1

# Copy biến production vào .env (hoặc dùng .env.production)
# QUAN TRỌNG: QDRANT_URL, NEO4J_URI phải trỏ cloud production

# Dry-run trước (không ghi DB)
python scripts/ingest_knowledge.py --source sample_data --dry-run --limit-files 1 --limit-chunks 5

# Ingest thật (30–90 phút, gọi Google Embedding + LlamaParse + Ollama graph nếu có)
python scripts/ingest_knowledge.py --source sample_data
```

> **Graph extraction** mặc định dùng **Ollama local**. Trên máy không có Ollama:
> - Cài Ollama + pull `qwen3:8b`, HOẶC
> - Chạy `--skip-graph-extraction` nếu đã có cache graph từ lần ingest trước

Sau ingest, kiểm tra Qdrant Cloud dashboard → collection `acne_knowledge` phải có **points > 0**.

### 9.3 — (Tuỳ chọn) Build entity index

```powershell
python scripts/build_entity_index.py --dry-run
python scripts/build_entity_graph.py --dry-run
# Chỉ chạy --no-dry-run khi đã hiểu tác động
```

---

## 10. Deploy frontend (React/Vite)

### Bước 10.1 — Thêm service frontend

1. Trong **project canvas** → bấm **+ Create**
2. Chọn **GitHub Repo**
3. Chọn lại repo **`RAG-system-for-acne-diagnose`**
4. Railway tạo service mới → đổi tên thành **`frontend`**

### Bước 10.2 — Cấu hình Settings

1. Service **frontend** → tab **Settings**
2. **Source → Root Directory**: nhập `src/frontend`
3. **Build → Builder**: **Dockerfile**
4. **Build → Dockerfile Path**: `Dockerfile` (tương đối root directory)

### Bước 10.3 — Biến môi trường frontend

Tab **Variables** → thêm:

```env
VITE_API_URL=https://<backend-domain>.up.railway.app
```

Thay `<backend-domain>` bằng domain backend đã generate ở bước 8.1.

> `VITE_API_URL` được bake vào build lúc Docker build — mỗi lần đổi URL backend phải **redeploy frontend**.

### Bước 10.4 — Generate domain frontend

1. **Settings → Networking → Generate Domain**
2. Copy URL frontend, vd: `https://frontend-production-yyyy.up.railway.app`

### Bước 10.5 — Cập nhật CORS backend

1. Quay lại service **backend** → **Variables**
2. Sửa `CORS_ALLOW_ORIGINS` = URL frontend vừa tạo (không slash cuối):
   ```
   CORS_ALLOW_ORIGINS=https://frontend-production-yyyy.up.railway.app
   ```
3. Save → Railway tự redeploy backend

### Bước 10.6 — Redeploy frontend

1. Service **frontend** → tab **Deployments**
2. Bấm **⋮** (menu) trên deployment mới nhất → **Redeploy** (nếu cần rebuild sau khi set `VITE_API_URL`)

---

## 11. Kiểm tra sau deploy

### 11.1 — Health check backend

Mở trình duyệt hoặc PowerShell:

```powershell
Invoke-RestMethod "https://<backend-domain>.up.railway.app/health" | ConvertTo-Json -Depth 5
```

Kết quả mong đợi:

```json
{
  "status": "ok",
  "service": "acne-advisor-api",
  "postgres": "ok",
  "qdrant": "ok",
  "neo4j": "ok",
  "redis": "ok",
  "ollama": "unavailable"
}
```

- `ollama: unavailable` là **bình thường** trên Railway (không có Ollama local)
- `status: degraded` nếu Qdrant chưa có data (points_count = 0) — cần chạy ingest

### 11.2 — Test chat trên UI

1. Mở URL frontend trên trình duyệt
2. Gõ câu hỏi thử: *"Adapalene dùng cho loại mụn nào?"*
3. Mở **DevTools** (F12) → tab **Network**:
   - Request phải gọi `https://<backend-domain>/chat`
   - Status **200**
4. Nếu lỗi CORS → kiểm tra lại `CORS_ALLOW_ORIGINS`

### 11.3 — Xem logs

- Service **backend** → tab **Deployments** → bấm deployment → **View Logs**
- Tìm warning `RerankerUnavailable` → bình thường (fallback rule-based)
- Lỗi `GOOGLE_API_KEY` → kiểm tra biến trên Railway

---

## 12. Phương án B: tự host Qdrant/Neo4j trên Railway

> Chỉ dùng nếu không muốn Qdrant Cloud / Neo4j Aura. Rủi ro: OOM, mất data nếu volume cấu hình sai.

### Qdrant trên Railway

1. **+ Create → Empty Service**
2. **Settings → Source → Deploy from Docker Image**
3. Image: `qdrant/qdrant:v1.18.0`
4. **Settings → Volumes** → Mount `/qdrant/storage`
5. **Settings → Networking** → expose port `6333`
6. Biến backend: `QDRANT_URL=http://<qdrant-service-name>.railway.internal:6333`

### Neo4j trên Railway

1. **+ Create → Empty Service**
2. Image: `neo4j:5`
3. Variables:
   ```
   NEO4J_AUTH=neo4j/<password-manh>
   NEO4J_PLUGINS=["apoc"]
   ```
4. Volume mount `/data`
5. Biến backend: `NEO4J_URI=bolt://<neo4j-service>.railway.internal:7687`

> Neo4j dễ bị **OOM kill** trên plan free/hobby. Nếu container restart liên tục → chuyển sang Aura Free.

---

## 13. Troubleshooting

| Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| Build backend **Failed** / timeout | PyTorch quá nặng | Đợi lâu hơn hoặc nâng plan Railway |
| `/health` → postgres **unavailable** | Chưa link `DATABASE_URL` | Dùng `${{postgres.DATABASE_URL}}`, redeploy |
| `/health` → qdrant **unavailable** | Sai URL/API key | Kiểm tra Qdrant Cloud endpoint + key |
| `/health` → neo4j **unavailable** | Sai URI hoặc password | Aura dùng `neo4j+s://`, không phải `bolt://` |
| `/health` status **degraded**, qdrant ok nhưng points=0 | Chưa ingest | Chạy `ingest_knowledge.py` từ local |
| Frontend gọi API **CORS error** | Sai `CORS_ALLOW_ORIGINS` | Thêm đúng domain frontend, redeploy backend |
| Frontend gọi `127.0.0.1:8000` | Sai `VITE_API_URL` | Set biến trên frontend service, **redeploy** |
| Chat trả lỗi Gemini | Thiếu/sai `GOOGLE_API_KEY` | Kiểm tra key trên AI Studio |
| `railway run` lỗi module | Chạy sai lệnh | Dùng `python scripts/init_schema.py` (không phải `-m scripts.init_schema`) |
| Ingest timeout | PDF lớn + API chậm | Chạy từ local, dùng `--limit-files 1` test trước |

---

## 14. Checklist bảo mật

- [ ] HTTPS: Railway tự cấp cho `*.up.railway.app`
- [ ] `CORS_ALLOW_ORIGINS` = domain frontend cụ thể, **không** dùng `*`
- [ ] `PHASE2_DEBUG_METADATA=false`
- [ ] Không commit `.env` / API keys vào Git
- [ ] Postgres/Redis **không** expose public port
- [ ] Qdrant/Neo4j dùng managed cloud (có auth) hoặc Railway internal network
- [ ] (Khuyên) Thêm rate limit / API key nếu public lâu dài
- [ ] Medical disclaimer hiển thị trên UI (đã có sẵn trong app)

---

## Tóm tắt thứ tự thực hiện

```
1. Push deploy files lên GitHub
2. Railway: New Project → GitHub repo → backend service
3. + PostgreSQL, + Redis
4. Tạo Qdrant Cloud + Neo4j Aura (managed)
5. Set biến môi trường backend + Generate domain
6. railway run init_schema + init_chat_schema
7. Ingest từ máy local (sample_data → cloud DBs)
8. + Frontend service (root: src/frontend) + VITE_API_URL
9. Generate domain frontend → cập nhật CORS
10. Test /health + chat UI
```

Chúc bạn deploy thành công! 🚀
