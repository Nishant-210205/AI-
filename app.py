"""
Streamlit User Interface (Cloud-Ready, Free-Tier Native & Claude AI Compatible)
Local & Cloud LLM-Based College Document Question Answering Assistant Using RAG
Project Management and Tracking Using Jira
"""

import os
import time
import streamlit as st
import pandas as pd
from rag_service import RAGService
from document_loader import load_pdf_with_metadata
from vector_store import is_placeholder_qdrant_url
from llm_service import get_secret

# Configure page layout and style
st.set_page_config(
    page_title="DocuMind AI - College Document Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.02rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .answer-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .source-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 8px 12px;
        margin-top: 8px;
        border-radius: 4px;
        font-size: 0.95rem;
    }
    .status-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state for runtime provider selection
if "llm_provider" not in st.session_state:
    st.session_state["llm_provider"] = get_secret("LLM_PROVIDER", "gemini").lower()


# Production Cloud Persistence Enforcement Guard
_v_prov = (get_secret("VECTOR_PROVIDER") or get_secret("VECTOR_STORE_PROVIDER") or "").lower()
_is_cloud = bool(
    os.environ.get("STREAMLIT_SHARING_HOST")
    or os.environ.get("STREAMLIT_SERVER_PORT")
    or get_secret("IS_STREAMLIT_CLOUD") == "true"
)

_q_url = get_secret("QDRANT_URL")
_q_key = get_secret("QDRANT_API_KEY")
_is_placeholder = is_placeholder_qdrant_url(_q_url) if _q_url else True

# If a non-placeholder Qdrant cluster is explicitly configured, validate format
if not _is_placeholder and _q_url and _q_key:
    try:
        validate_and_sanitize_qdrant_url(_q_url)
        sanitize_qdrant_api_key(_q_key)
    except ValueError as val_err:
        st.error(
            f"🛑 **Blocking Configuration Error: Invalid Qdrant Cloud Format**\n\n"
            f"**Details:** {val_err}\n\n"
            "**Action Required in Streamlit Secrets:**\n"
            "- Ensure `QDRANT_URL` is a plain HTTPS endpoint (e.g., `https://bc55f8d5-....cloud.qdrant.io`), not a Markdown link.\n"
            "- Ensure `QDRANT_API_KEY` is pasted on a single line without quotes, whitespace, or newlines."
        )
        st.stop()


@st.cache_resource(show_spinner=False)
def get_cached_rag(provider: str):
    rag = RAGService(llm_provider=provider)
    stats = rag.get_stats()
    if stats.get("total_chunks", 0) == 0 and os.path.exists("./data"):
        try:
            rag.index_directory("./data")
        except Exception as e:
            print(f"Warning during directory indexing: {e}")
    return rag



rag_service = get_cached_rag(st.session_state["llm_provider"])
health = rag_service.check_health()
stats = rag_service.get_stats()


# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/books.png", width=64)
    st.title("Cloud Settings")

    st.markdown("### 🤖 LLM Provider Selection")

    # Claude is only available when explicitly enabled via secrets AND key is set
    claude_allowed = get_secret("ENABLE_CLAUDE", "false").lower() == "true" and bool(get_secret("ANTHROPIC_API_KEY"))

    provider_map = {
        "Google Gemini (Free Cloud)": "gemini",
        "Groq Cloud (Free Fast)": "groq",
        "Mock (Testing / Offline)": "mock",
        "Local Ollama (Local Daemon Only)": "ollama",
    }
    if claude_allowed:
        provider_map["Anthropic Claude (Cloud)"] = "claude"

    provider_labels = list(provider_map.keys())
    current_label = next((k for k, v in provider_map.items() if v == st.session_state["llm_provider"]), provider_labels[0])
    selected_provider_label = st.selectbox("Active Provider:", provider_labels, index=provider_labels.index(current_label))

    new_provider = provider_map[selected_provider_label]
    if new_provider != st.session_state["llm_provider"]:
        st.session_state["llm_provider"] = new_provider
        st.rerun()

    # Provider specific information / status
    if new_provider == "gemini":
        gemini_model = get_secret("GEMINI_MODEL", "gemini-2.0-flash")
        st.caption(f"**Model:** `{gemini_model}` (Google AI Studio Free Tier)")
        current_g_key = get_secret("GEMINI_API_KEY", "")
        is_invalid = not current_g_key or current_g_key.startswith("your_") or current_g_key.startswith("AIzaSyYour")
        key_input = st.text_input(
            "🔑 Gemini API Key:",
            value="" if is_invalid else current_g_key,
            type="password",
            placeholder="AIzaSy...",
            help="Get a free key (no credit card required) at https://aistudio.google.com/apikey"
        )
        if key_input and key_input != current_g_key:
            st.session_state["GEMINI_API_KEY"] = key_input
            st.rerun()
        if is_invalid:
            st.info("💡 Enter your key above or in Secrets. Free at [Google AI Studio](https://aistudio.google.com/apikey)")
    elif new_provider == "groq":
        groq_model = get_secret("GROQ_MODEL", "llama-3.3-70b-versatile")
        st.caption(f"**Model:** `{groq_model}` (Groq Cloud Free Tier)")
        if not get_secret("GROQ_API_KEY"):
            st.warning("⚠️ `GROQ_API_KEY` not detected in Secrets/Env. Free key at [Groq Console](https://console.groq.com/)")
    elif new_provider == "claude":
        claude_model = get_secret("CLAUDE_MODEL", "claude-3-5-haiku-20241022")
        st.caption(f"**Model:** `{claude_model}` (Anthropic Claude)")
    elif new_provider == "ollama":
        st.caption("🖥️ **Local Ollama Daemon:** Requires local Ollama running on `localhost:11434` (Incompatible with cloud hosting)")
    elif new_provider == "mock":
        st.caption("🧪 **Deterministic Mock LLM:** Offline testing with zero network latency or costs")

    st.divider()

    st.markdown("### ⚙️ Cloud Health Diagnostics")
    if health["llm_online"]:
        st.success(f"🟢 {health['llm_service']} Online")
    else:
        st.warning(f"🟡 {health['llm_service']} Offline / Needs Key")

    st.caption(f"**Embedding:** `{health['embedding_service']}`")
    st.caption(f"**Vector Store:** `{health['vector_store_provider']}`")
    st.caption(f"**Indexed Chunks:** `{stats['total_chunks']}` across `{len(stats.get('unique_documents', []))}` docs")

    st.divider()

    st.markdown("### 📚 Indexed Documents")
    if stats.get("unique_documents"):
        for doc in stats["unique_documents"]:
            st.markdown(f"📄 `{doc}`")
    else:
        st.warning("No documents indexed.")

    col_idx1, col_idx2 = st.columns(2)
    with col_idx1:
        if st.button("🔄 Re-index All", use_container_width=True):
            with st.spinner("Re-indexing PDF documents in data/ ..."):
                res = rag_service.index_directory("./data", force_reindex=True)
                st.success(f"Indexed {res.get('indexed_chunks', 0)} chunks!")
                st.rerun()

    with col_idx2:
        if st.button("🗑️ Reset DB", use_container_width=True):
            rag_service.reset_index()
            st.warning("Vector database reset.")
            st.rerun()

    st.divider()

    st.markdown("### 📤 Ingest New PDF")
    st.caption("ℹ️ *Uploads are processed in-memory. Extracted chunks are stored in vector database payloads; raw PDFs are not saved to disk.*")
    uploaded_file = st.file_uploader("Upload a College PDF", type=["pdf"], key="pdf_uploader")
    if uploaded_file is not None:
        if st.button("📥 Index Uploaded PDF", use_container_width=True):
            with st.spinner(f"Processing {uploaded_file.name} in memory..."):
                file_bytes = uploaded_file.getvalue()
                res = rag_service.index_pdf_bytes(file_bytes, uploaded_file.name)
                if res.get("status") == "success":
                    st.success(f"Added {uploaded_file.name} ({res.get('indexed_chunks', 0)} chunks)")
                    st.rerun()
                elif res.get("status") == "warning":
                    st.warning(f"⚠️ {res.get('message')}")
                else:
                    st.error(f"❌ {res.get('message')}")

    st.divider()
    st.caption("🚀 **DocuMind AI** | Cloud RAG Assistant")


# ----------------- MAIN CONTENT -----------------
st.markdown('<div class="main-header">🎓 College Document Q&A Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Cloud-Native Retrieval-Augmented Generation (RAG) powered by <b>Google Gemini / FastEmbed</b> & Qdrant / ChromaDB | Tracked via Jira Scrum</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Ask Assistant",
    "☁️ Cloud Architecture & Diagnostics",
    "📊 Jira Scrum Project",
    "📑 Document Knowledge Base",
    "📈 Evaluation & Metrics"
])

# ----------------- TAB 1: ASK ASSISTANT -----------------
with tab1:
    st.markdown("#### Ask questions about college regulations, exams, placement, or campus conduct:")

    if stats.get("notice"):
        st.info(f"ℹ️ **Vector Database Notice:** {stats.get('notice')}")
    elif stats.get("warning"):
        st.warning(f"⚠️ **Vector Database Warning:** {stats.get('warning')}")
    elif stats.get("error"):
        st.warning(
            f"⚠️ **Vector Database Warning:** {stats.get('error')}\n\n"
            "*(Answers may be unavailable until Qdrant Cloud connectivity is restored.)*"
        )

    # Quick test question presets
    presets = [
        "Select a sample question...",
        "What is the minimum attendance required to appear for the semester examination?",
        "How many classes can I miss?",
        "How many credits are required for graduation?",
        "What is the procedure for examination revaluation?",
        "What are the rules for backlog examination?",
        "What is the eligibility criterion for campus placement?",
        "What documents are required for semester registration?",
        "What is the fine for overdue library books?",
        "What is the hostel fee?"  # Unanswerable test question (anti-hallucination test)
    ]

    selected_preset = st.selectbox("💡 Quick Test Presets:", presets, index=0)

    # Input query
    query_input = st.text_input(
        "Enter your question:",
        value=selected_preset if selected_preset != presets[0] else "",
        placeholder="e.g., What is the minimum attendance required?"
    )

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        ask_clicked = st.button("🔍 Ask Question", type="primary", use_container_width=True)

    if ask_clicked and query_input:
        start_time = time.time()
        with st.spinner(f"Retrieving context and querying {health['llm_service']}..."):
            result = rag_service.answer_question(query_input)
            elapsed_time = round(time.time() - start_time, 2)

        st.markdown("### 📝 Answer")
        if result.get("is_error"):
            st.error(f"❌ {result['answer']}")
            if any(term in result["answer"] for term in ("API key not valid", "INVALID_ARGUMENT", "Google Gemini API key not configured")):
                st.info(
                    "💡 **Action Required: Invalid or Placeholder Gemini Key**\n\n"
                    "Google rejected the provided key. You can fix this in seconds:\n\n"
                    "1. Get a 100% free key at [Google AI Studio](https://aistudio.google.com/apikey) (no credit card required).\n"
                    "2. Paste it directly in the left sidebar under **🔑 Gemini API Key** (or in Streamlit Cloud Secrets).\n"
                    "3. *(Alternatively, switch **Active Provider** in the left sidebar to **Mock (Testing / Offline)** to test the full RAG pipeline immediately without any API key).* "
                )
            elif "Anthropic API key is not configured" in result["answer"]:
                st.info("💡 **Setup Instructions:** Configure `ANTHROPIC_API_KEY` and set `ENABLE_CLAUDE=true` in Streamlit Cloud Secrets.")
            elif "Groq API key not configured" in result["answer"]:
                st.info("💡 **Setup Instructions:** Configure `GROQ_API_KEY` in Streamlit Cloud Secrets. Obtain a free key at [Groq Console](https://console.groq.com/).")
        elif result["is_refusal"]:
            st.warning(f"⚠️ **{result['answer']}**")
            st.caption("ℹ️ The requested information was not found in the indexed college documents. The system adheres to strict zero-hallucination policies.")
        else:
            st.markdown(f'<div class="answer-card">{result["answer"]}</div>', unsafe_allow_html=True)

            if result["sources"]:
                st.markdown("#### 📌 Verified Source Document(s):")
                for src in result["sources"]:
                    st.markdown(f'<div class="source-box">📄 <b>Source:</b> {src}</div>', unsafe_allow_html=True)

        st.caption(f"⏱️ Generated in {elapsed_time}s via `{health['llm_service']}` ({health['llm_model']}) & `{health['embedding_service']}`")

        # Retrieved Context Accordion
        with st.expander("🔍 View Retrieved Context & Similarity Details"):
            st.markdown(f"**Retrieved {len(result['retrieved_chunks'])} document chunk(s):**")
            for idx, chunk in enumerate(result["retrieved_chunks"], 1):
                st.markdown(f"**Chunk #{idx} | Source:** `{chunk.get('source')}` | **Page:** `{chunk.get('page')}` | **Similarity Score:** `{chunk.get('similarity')}`")
                st.info(chunk.get("text"))


# ----------------- TAB 2: CLOUD ARCHITECTURE & DIAGNOSTICS -----------------
with tab2:
    st.markdown("### ☁️ Cloud Platforms & Service Integrations")
    st.markdown("Status and configuration of the integrated cloud components:")

    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.markdown("""
        <div class="status-card">
            <h4>🌐 Hosting Platform</h4>
            <p><b>Platform:</b> Streamlit Community Cloud</p>
            <p><b>Status:</b> 🟢 Ready for Deploy</p>
            <p><b>Resource Limit:</b> 1 GB RAM / App</p>
            <p><b>Lifecycle:</b> Sleeps on inactivity</p>
        </div>
        """, unsafe_allow_html=True)

    with col_c2:
        llm_stat = "🟢 Connected" if health["llm_online"] else "🟡 Awaiting API Key"
        st.markdown(f"""
        <div class="status-card">
            <h4>🧠 Cloud LLM</h4>
            <p><b>Provider:</b> {health['llm_service']}</p>
            <p><b>Status:</b> {llm_stat}</p>
            <p><b>Model:</b> {health['llm_model']}</p>
            <p><b>Auth:</b> Streamlit Secrets / Env</p>
        </div>
        """, unsafe_allow_html=True)

    with col_c3:
        vdb_err = stats.get("error")
        vdb_status = "🟢 Operational" if not vdb_err else "🔴 Error / Unreachable"
        if stats.get("fallback_active"):
            vdb_status = "🟢 Operational (Resilient Fallback)"
        st.markdown(f"""
        <div class="status-card">
            <h4>🗄️ Vector Database</h4>
            <p><b>Provider:</b> {stats.get('provider')}</p>
            <p><b>Status:</b> {vdb_status}</p>
            <p><b>Storage:</b> Payloads & Embeddings</p>
            <p><b>Chunks:</b> {stats.get('total_chunks', 0)} Indexed</p>
        </div>
        """, unsafe_allow_html=True)
        if stats.get("notice"):
            st.info(f"ℹ️ {stats.get('notice')}")
        elif stats.get("warning"):
            st.warning(f"⚠️ {stats.get('warning')}")
        elif vdb_err:
            st.error(f"⚠️ **Vector Database Warning:** {vdb_err}")

    st.markdown("#### 📋 Integration Matrix")
    integration_matrix = [
        {"Platform": "A. Streamlit Community Cloud", "Role": "Production Web Hosting", "Status": "Configured & Deployable", "Cost": "₹0 (Community Plan)", "Limits & Terms": "1 GB RAM limit, sleeps after consecutive days of inactivity"},
        {"Platform": "B. Google Gemini API", "Role": "Cloud LLM (Default Provider)", "Status": "Integrated via google-genai SDK", "Cost": "₹0 (Free Tier)", "Limits & Terms": "15 RPM, 1M TPM, 1,500 RPD for Gemini 2.0 Flash; free tier data may be logged"},
        {"Platform": "C. Groq Cloud", "Role": "Secondary Cloud LLM", "Status": "Integrated via groq SDK", "Cost": "₹0 (Free Tier)", "Limits & Terms": "30 RPM, 6k-30k TPM depending on model"},
        {"Platform": "D. FastEmbed (ONNX)", "Role": "Semantic Vector Embeddings", "Status": "Integrated via fastembed", "Cost": "₹0 (In-container)", "Limits & Terms": "BAAI/bge-small-en-v1.5 (384-dim, 67MB ONNX, zero API calls)"},
        {"Platform": "E. Qdrant Cloud", "Role": "Semantic Vector Storage", "Status": "Integrated via qdrant-client", "Cost": "₹0 (Free Tier)", "Limits & Terms": "1 GB free cluster, no credit card required, suspends after 7 days inactivity"},
        {"Platform": "F. ChromaDB", "Role": "Local Vector Storage", "Status": "Active for Local Dev", "Cost": "₹0 (Local)", "Limits & Terms": "File-based storage in chroma_db/ for local testing & CI fallback"},
        {"Platform": "G. GitHub", "Role": "Source Control & CI/CD", "Status": "Connected", "Cost": "₹0 (Free)", "Limits & Terms": "Public repo: co2023smniyazi/DocuMind-AI (branch: cloud-scalable-rag)"},
        {"Platform": "H. Anthropic Claude (API)", "Role": "Optional Cloud LLM", "Status": "Mocked & Implemented (OFF by default)", "Cost": "Paid API (Requires Paid Key)", "Limits & Terms": "Requires ENABLE_CLAUDE=true and ANTHROPIC_API_KEY"},
        {"Platform": "I. Amazon Bedrock (Claude)", "Role": "Multi-Cloud Adapter", "Status": "Mocked & Implemented (OFF by default)", "Cost": "Paid AWS (Requires AWS Key)", "Limits & Terms": "Requires ENABLE_CLAUDE=true and ENABLE_BEDROCK_CLAUDE=true"},
        {"Platform": "J. Google Vertex AI (Claude)", "Role": "Multi-Cloud Adapter", "Status": "Mocked & Implemented (OFF by default)", "Cost": "Paid GCP (Requires GCP Project)", "Limits & Terms": "Requires ENABLE_CLAUDE=true and ENABLE_VERTEX_CLAUDE=true"},
        {"Platform": "K. MCP Tool Server", "Role": "Agent Protocol Gateway", "Status": "Implemented & Mocked (Not Deployed)", "Cost": "₹0 (Standalone code)", "Limits & Terms": "Exposes search_documents and ask_documents via Bearer auth"},
        {"Platform": "L. Jira Cloud", "Role": "Agile Scrum Management", "Status": "Backlog & Guide Ready", "Cost": "₹0 (Free Tier)", "Limits & Terms": "Free up to 10 users, 6 Epics, 4 Sprints, 25 Items"}
    ]
    st.dataframe(pd.DataFrame(integration_matrix), use_container_width=True)

    st.markdown("#### 🔒 Security & Secrets Configuration")
    st.markdown("""
    To configure secrets in Streamlit Community Cloud:
    1. Navigate to your app dashboard on [share.streamlit.io](https://share.streamlit.io).
    2. Click **App Settings > Secrets**.
    3. Add your secrets in TOML format:
    ```toml
    # Primary Free Cloud LLM (Google Gemini)
    GEMINI_API_KEY = "AIzaSy..."
    GEMINI_MODEL = "gemini-2.0-flash"
    LLM_PROVIDER = "gemini"

    # Optional Free Secondary LLM (Groq)
    GROQ_API_KEY = "gsk_..."
    GROQ_MODEL = "llama-3.3-70b-versatile"

    # Optional Qdrant Cloud Vector Store
    QDRANT_URL = "https://your-cluster-id.region.gcp.cloud.qdrant.io:6333"
    QDRANT_API_KEY = "your-qdrant-api-key"
    VECTOR_STORE_PROVIDER = "qdrant"

    # Optional Claude (OFF by default; budget rule: keep false unless approved)
    ENABLE_CLAUDE = "false"
    ANTHROPIC_API_KEY = "sk-ant-..."
    ```
    """)


# ----------------- TAB 3: JIRA SCRUM PROJECT -----------------
with tab3:
    import json
    st.markdown("### 📋 Jira Project Tracking (Project Key: RAG)")
    st.markdown("Live agile project state loaded directly from `jira_management/status.json`:")

    status_file = "./jira_management/status.json"
    jira_info = {}
    if os.path.exists(status_file):
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                jira_info = json.load(f)
        except Exception:
            pass

    # Dynamic Jira Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Project Name", jira_info.get("project_name", "College RAG Assistant"))
    col2.metric("Project Key", jira_info.get("project_key", "RAG"))
    col3.metric("Total Epics", f"{len(jira_info.get('epics', []))} Epics")
    col4.metric("Current Sprint", jira_info.get("current_sprint", "Sprint 4"))
    col5.metric("Story Points", f"{jira_info.get('total_story_points', 90)} SP")

    st.markdown("#### 🚀 Jira Workflow State")
    st.code("TO DO  ➡️  IN PROGRESS  ➡️  CODE REVIEW  ➡️  TESTING  ➡️  DONE", language="text")

    # Epics overview loaded dynamically from status.json (defaults to TO DO)
    epics_data = jira_info.get("epics", [
        {"epic_id": "EPIC-1", "name": "Document Ingestion & Extraction", "status": "TO DO", "story_points": 13},
        {"epic_id": "EPIC-2", "name": "Semantic Chunking & Vector Indexing", "status": "TO DO", "story_points": 18},
        {"epic_id": "EPIC-3", "name": "Cloud-Native Embedding & Vector Storage", "status": "TO DO", "story_points": 21},
        {"epic_id": "EPIC-4", "name": "LLM Integration, Fallback & Grounded Generation", "status": "TO DO", "story_points": 21},
        {"epic_id": "EPIC-5", "name": "Testing, Safe Refusal & Benchmark Evaluation", "status": "TO DO", "story_points": 18},
        {"epic_id": "EPIC-6", "name": "Cloud Deployment, CI/CD & Security", "status": "TO DO", "story_points": 18},
    ])
    st.dataframe(pd.DataFrame(epics_data), use_container_width=True)

    st.markdown("#### 📅 Sprint Breakdown (Tracked in `jira_management/jira_import.csv`)")
    sprint_tabs = st.tabs(["Sprint 1: Foundation", "Sprint 2: RAG Pipeline", "Sprint 3: Cloud & Claude", "Sprint 4: Evaluation & Demo"])

    with sprint_tabs[0]:
        st.markdown("""
        - `RAG-1` Define functional requirements (Story - 3 SP) `[TO DO]`
        - `RAG-2` Collect college documents (Task - 3 SP) `[TO DO]`
        - `RAG-3` Prepare 50 sample questions (Task - 5 SP) `[TO DO]`
        - `RAG-4` Setup LLM environment (Task - 2 SP) `[TO DO]`
        - `RAG-5` Run and test baseline LLM (Task - 3 SP) `[TO DO]`
        """)

    with sprint_tabs[1]:
        st.markdown("""
        - `RAG-6` Develop PDF text extractor (Story - 3 SP) `[TO DO]`
        - `RAG-7` Develop text chunking module (Story - 3 SP) `[TO DO]`
        - `RAG-8` Generate vector embeddings (Story - 5 SP) `[TO DO]`
        - `RAG-9` Create vector database (Story - 5 SP) `[TO DO]`
        - `RAG-10` Develop semantic retrieval (Story - 5 SP) `[TO DO]`
        """)

    with sprint_tabs[2]:
        st.markdown("""
        - `RAG-11` Integrate retrieved context with LLM (Story - 5 SP) `[TO DO]`
        - `RAG-12` Implement hallucination prevention (Story - 5 SP) `[TO DO]`
        - `RAG-13` Add source document citation (Story - 3 SP) `[TO DO]`
        - `RAG-14` Integrate Google Gemini Cloud LLM (Story - 5 SP) `[TO DO]`
        - `RAG-15` FastEmbed ONNX local embedder (Story - 5 SP) `[TO DO]`
        """)

    with sprint_tabs[3]:
        st.markdown("""
        - `RAG-16` Qdrant Cloud vector database integration (Story - 5 SP) `[TO DO]`
        - `RAG-17` Run automated 50-question benchmark (Story - 5 SP) `[TO DO]`
        - `RAG-18` Implement multi-metric evaluation (Story - 5 SP) `[TO DO]`
        - `RAG-19` Setup Streamlit Cloud deployment (Story - 3 SP) `[TO DO]`
        - `RAG-20` Cloud health check diagnostic page (Story - 3 SP) `[TO DO]`
        """)


# ----------------- TAB 4: DOCUMENT KNOWLEDGE BASE -----------------
with tab4:
    st.markdown("### 📑 College Document Knowledge Base")
    st.markdown("Inspect seed college documents bundled with the repository:")

    if os.path.exists("./data"):
        doc_files = [f for f in sorted(os.listdir("./data")) if f.endswith(".pdf")]
        for doc_file in doc_files:
            file_path = os.path.join("./data", doc_file)
            file_size_kb = round(os.path.getsize(file_path) / 1024, 1)

            with st.expander(f"📄 {doc_file} ({file_size_kb} KB)"):
                pages, summary = load_pdf_with_metadata(file_path)
                st.markdown(f"**Total Pages:** {summary['total_pages']} | **Total Characters:** {summary['total_chars']}")
                if summary.get("is_scanned"):
                    st.warning("⚠️ Scanned Image Document (no digital text layer).")
                for p in pages:
                    st.markdown(f"**Page {p['page']} Text Preview:**")
                    st.text(p['text'])


# ----------------- TAB 5: EVALUATION & METRICS -----------------
with tab5:
    st.markdown("### 📈 System Evaluation & Metrics (STEP 21-23, 44-47)")
    st.markdown("Comprehensive evaluation across **50 benchmark questions** categorized into Direct, Paraphrased, and Unanswerable queries.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Category A (Direct)", "20 Questions", "Retrieval Testing")
    col2.metric("Category B (Paraphrased)", "20 Questions", "Semantic Testing")
    col3.metric("Category C (Unanswerable)", "10 Questions", "Safe Refusal Testing")

    st.markdown("#### 🎯 Evaluation Rubric")
    st.markdown("""
    | Metric | Meaning | Target Score |
    |---|---|---|
    | **Correctness** | Is the answer factually supported by the document? | 5 / 5 |
    | **Relevance** | Does it answer the specific question asked? | 5 / 5 |
    | **Groundedness** | Is it strictly based on retrieved context (no hallucinations)? | 5 / 5 |
    | **Clarity** | Is it understandable and grammatically well-formed? | 5 / 5 |
    | **Completeness** | Is sufficient information provided? | 5 / 5 |
    """)

    if os.path.exists("./evaluation_report.md"):
        st.markdown("#### 📄 Latest Benchmark Report")
        with open("./evaluation_report.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("Run `python evaluate.py` to generate the latest benchmark report.")

