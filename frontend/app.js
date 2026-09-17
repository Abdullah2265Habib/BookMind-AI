/**
 * BookMind-AI: Frontend Application Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  // State
  let currentBookId = null;
  let booksData = [];
  let currentSettings = {
    provider: "local",
    gemini_api_key: "",
    openai_api_key: "",
    groq_api_key: ""
  };
  let activePollingJob = null;
  let chatCitationsStore = {}; // Store citations by id for modal preview

  // DOM Elements
  const dropZone = document.getElementById("drop-zone");
  const pdfFileInput = document.getElementById("pdf-file-input");
  const presetPaperBtn = document.getElementById("preset-paper-btn");
  const pipelineProgressCard = document.getElementById("pipeline-progress-card");
  const progressBarFill = document.getElementById("progress-bar-fill");
  const pipelineStatusText = document.getElementById("pipeline-status-text");
  const pipelinePct = document.getElementById("pipeline-pct");
  const booksList = document.getElementById("books-list");
  const booksCount = document.getElementById("books-count");

  // Tabs
  const tabBtnChat = document.getElementById("tab-btn-chat");
  const tabBtnInspector = document.getElementById("tab-btn-inspector");
  const tabChat = document.getElementById("tab-chat");
  const tabInspector = document.getElementById("tab-inspector");
  const summaryBookTitle = document.getElementById("summary-book-title");

  // Chat
  const chatMessagesContainer = document.getElementById("chat-messages-container");
  const chatForm = document.getElementById("chat-form");
  const chatTextarea = document.getElementById("chat-textarea");
  const bannerTitle = document.getElementById("banner-title");
  const bannerMeta = document.getElementById("banner-meta");
  const clearChatBtn = document.getElementById("clear-chat-btn");
  const promptsChips = document.getElementById("prompts-chips");

  // Inspector
  const metricPages = document.getElementById("metric-pages");
  const metricDocType = document.getElementById("metric-doc-type");
  const metricChunks = document.getElementById("metric-chunks");
  const metricImages = document.getElementById("metric-images");
  const metricLang = document.getElementById("metric-lang");
  const chaptersTree = document.getElementById("chapters-tree");
  const chunksBrowserList = document.getElementById("chunks-browser-list");
  const chunksCountPill = document.getElementById("chunks-count-pill");

  // Modals
  const citationModal = document.getElementById("citation-modal");
  const citationModalTitle = document.getElementById("citation-modal-title");
  const citationModalMeta = document.getElementById("citation-modal-meta");
  const citationModalContent = document.getElementById("citation-modal-content");
  const citationModalClose = document.getElementById("citation-modal-close");
  const citationModalOk = document.getElementById("citation-modal-ok");

  const settingsModal = document.getElementById("settings-modal");
  const openSettingsBtn = document.getElementById("open-settings-btn");
  const settingsModalClose = document.getElementById("settings-modal-close");
  const settingsCancelBtn = document.getElementById("settings-cancel-btn");
  const settingsSaveBtn = document.getElementById("settings-save-btn");
  const providerSelect = document.getElementById("provider-select");
  const geminiApiKeyInput = document.getElementById("gemini-api-key");
  const openaiApiKeyInput = document.getElementById("openai-api-key");
  const groqApiKeyInput = document.getElementById("groq-api-key");
  const providerLabel = document.getElementById("provider-label");

  // --- Initial Load ---
  async function init() {
    loadLocalSettings();
    await fetchStatus();
    await refreshBooks();
    setupEventListeners();
  }

  function loadLocalSettings() {
    try {
      const saved = localStorage.getItem("bookmind_settings");
      if (saved) {
        currentSettings = JSON.parse(saved);
        updateProviderLabel();
      }
    } catch (e) {
      console.warn("Failed to load local settings", e);
    }
  }

  function saveLocalSettings() {
    localStorage.setItem("bookmind_settings", JSON.stringify(currentSettings));
    updateProviderLabel();
  }

  function updateProviderLabel() {
    const labels = {
      gemini: "Google Gemini 2.0",
      openai: "OpenAI GPT-4o-mini",
      groq: "Groq LLaMA 3.3",
      local: "Local Grounded CPU"
    };
    providerLabel.textContent = labels[currentSettings.provider] || "Local Grounded CPU";
  }

  async function fetchStatus() {
    try {
      const res = await fetch("/api/status");
      if (!res.ok) return;
      const data = await res.json();
      if (!currentSettings.provider || currentSettings.provider === "local") {
        if (data.has_gemini_key) currentSettings.provider = "gemini";
        else if (data.has_openai_key) currentSettings.provider = "openai";
        else if (data.has_groq_key) currentSettings.provider = "groq";
        updateProviderLabel();
      }
    } catch (err) {
      console.warn("Status fetch error", err);
    }
  }

  // --- Books Management ---
  async function refreshBooks(selectBookId = null) {
    try {
      const res = await fetch("/api/books");
      if (!res.ok) return;
      booksData = await res.json();
      renderBooksList();

      if (selectBookId) {
        selectBook(selectBookId);
      } else if (booksData.length > 0 && !currentBookId) {
        selectBook(booksData[0].book_id);
      }
    } catch (err) {
      console.error("Failed to refresh books", err);
    }
  }

  function renderBooksList() {
    booksCount.textContent = `${booksData.length} Book${booksData.length === 1 ? "" : "s"}`;
    if (booksData.length === 0) {
      booksList.innerHTML = `
        <div class="empty-state">
          <p>No books indexed yet. Upload a PDF or load the sample paper above.</p>
        </div>
      `;
      return;
    }

    booksList.innerHTML = booksData.map(book => {
      const isSelected = book.book_id === currentBookId;
      return `
        <div class="book-card ${isSelected ? "selected" : ""}" data-id="${book.book_id}">
          <h4 class="book-card-title">${escapeHtml(book.title || "Untitled Book")}</h4>
          <div class="book-card-meta">
            <span class="book-meta-pill">${book.total_pages} Pages</span>
            <span class="book-meta-pill">${book.document_type || "text-based"}</span>
            <span class="book-meta-pill">${book.total_chunks || 0} Chunks</span>
          </div>
        </div>
      `;
    }).join("");

    // Add click handlers
    document.querySelectorAll(".book-card").forEach(card => {
      card.addEventListener("click", () => {
        const id = card.getAttribute("data-id");
        selectBook(id);
      });
    });
  }

  async function selectBook(bookId) {
    currentBookId = bookId;
    const book = booksData.find(b => b.book_id === bookId);
    if (!book) return;

    // Update active highlight in UI
    document.querySelectorAll(".book-card").forEach(c => {
      c.classList.toggle("selected", c.getAttribute("data-id") === bookId);
    });

    summaryBookTitle.textContent = book.title || "Selected Book";
    bannerTitle.textContent = book.title || "Book Assistant Ready";
    bannerMeta.textContent = `${book.total_pages} pages • ${book.document_type} • ${book.total_chunks || 0} semantic chunks • Strictly grounded answers`;

    // Load Inspector Data
    loadInspectorData(bookId);
  }

  async function loadInspectorData(bookId) {
    try {
      const res = await fetch(`/api/books/${bookId}`);
      if (!res.ok) return;
      const data = await res.json();
      const meta = data.metadata;

      metricPages.textContent = meta.total_pages || "-";
      metricDocType.textContent = meta.document_type || "text-based";
      metricChunks.textContent = meta.total_chunks || "-";
      metricImages.textContent = meta.total_images || "0";
      metricLang.textContent = (meta.primary_language || "en").toUpperCase();

      chunksCountPill.textContent = `${meta.total_chunks || 0} Chunks`;

      // Render TOC / Chapters
      const toc = meta.table_of_contents || [];
      if (toc.length > 0) {
        chaptersTree.innerHTML = toc.map(item => `
          <div class="chapter-node" style="padding-left: ${item.level * 12 + 10}px">
            <span class="chapter-node-title">${escapeHtml(item.title)}</span>
            <span class="chapter-node-page">Page ${item.page}</span>
          </div>
        `).join("");
      } else {
        chaptersTree.innerHTML = `
          <div class="chapter-node">
            <span class="chapter-node-title">${escapeHtml(meta.title)} (Full Document Structure)</span>
            <span class="chapter-node-page">1-${meta.total_pages}</span>
          </div>
        `;
      }

      // Render Chunks Browser
      const chunks = data.chunks_sample || [];
      if (chunks.length > 0) {
        chunksBrowserList.innerHTML = chunks.map(c => `
          <div class="chunk-card-item">
            <div class="chunk-card-meta">
              <span>[${c.header_context || "Page " + c.page_number}]</span>
              <span>${c.char_count} chars</span>
            </div>
            <p class="chunk-card-snippet">${escapeHtml(c.content.slice(0, 180))}...</p>
          </div>
        `).join("");
      } else {
        chunksBrowserList.innerHTML = `<p class="empty-hint">No chunks available.</p>`;
      }
    } catch (err) {
      console.warn("Failed to load inspector data", err);
    }
  }

  // --- Pipeline Processing & Upload ---
  function startJobPolling(jobId) {
    pipelineProgressCard.classList.remove("hidden");
    progressBarFill.style.width = "5%";
    pipelinePct.textContent = "5%";
    pipelineStatusText.textContent = "Processing PDF on 4 CPU Cores...";

    if (activePollingJob) clearInterval(activePollingJob);

    activePollingJob = setInterval(async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        if (!res.ok) return;
        const job = await res.json();

        const pct = job.progress || 10;
        progressBarFill.style.width = `${pct}%`;
        pipelinePct.textContent = `${pct}%`;
        pipelineStatusText.textContent = job.stage || "Processing...";

        // Update stage indicators
        updateStageBullets(pct);

        if (job.status === "completed") {
          clearInterval(activePollingJob);
          progressBarFill.style.width = "100%";
          pipelinePct.textContent = "100%";
          pipelineStatusText.textContent = "Knowledge base ready!";

          setTimeout(async () => {
            pipelineProgressCard.classList.add("hidden");
            await refreshBooks(job.book_id);
          }, 1200);
        } else if (job.status === "failed") {
          clearInterval(activePollingJob);
          pipelineStatusText.textContent = `Failed: ${job.error || "Unknown error"}`;
        }
      } catch (e) {
        console.error("Job polling error", e);
      }
    }, 800);
  }

  function updateStageBullets(pct) {
    const s1 = document.getElementById("stage-1");
    const s2 = document.getElementById("stage-2");
    const s3 = document.getElementById("stage-3");
    const s4 = document.getElementById("stage-4");

    s1.className = "stage-item " + (pct >= 40 ? "done" : pct >= 15 ? "active" : "");
    s2.className = "stage-item " + (pct >= 60 ? "done" : pct >= 45 ? "active" : "");
    s3.className = "stage-item " + (pct >= 80 ? "done" : pct >= 65 ? "active" : "");
    s4.className = "stage-item " + (pct >= 100 ? "done" : pct >= 85 ? "active" : "");
  }

  async function handleFileUpload(file) {
    if (!file || !file.name.toLowerCase().endsWith(".pdf")) {
      alert("Please select a valid PDF file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      pipelineProgressCard.classList.remove("hidden");
      pipelineStatusText.textContent = "Uploading PDF file...";
      progressBarFill.style.width = "10%";
      pipelinePct.textContent = "10%";

      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }

      const data = await res.json();
      startJobPolling(data.job_id);
    } catch (err) {
      alert(`Upload error: ${err.message}`);
      pipelineProgressCard.classList.add("hidden");
    }
  }

  async function loadPresetPaper() {
    try {
      presetPaperBtn.disabled = true;
      presetPaperBtn.textContent = "Launching 4-Core Ingestion...";
      const res = await fetch("/api/process-preset", { method: "POST" });
      if (!res.ok) throw new Error("Failed to process preset PDF");
      const data = await res.json();
      startJobPolling(data.job_id);
    } catch (err) {
      alert(err.message);
    } finally {
      presetPaperBtn.disabled = false;
      presetPaperBtn.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="btn-icon">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
          <polyline points="14 2 14 8 20 8"></polyline>
          <line x1="16" y1="13" x2="8" y2="13"></line>
          <line x1="16" y1="17" x2="8" y2="17"></line>
        </svg>
        Load Pre-uploaded 6G Networks Paper
      `;
    }
  }

  // --- Chat Interactions ---
  async function submitQuery(queryText) {
    const query = (queryText || chatTextarea.value).trim();
    if (!query) return;

    if (!currentBookId) {
      alert("Please select or upload a book first.");
      return;
    }

    chatTextarea.value = "";
    chatTextarea.style.height = "auto";

    // Append User Message
    appendMessage("user", query);

    // Append Assistant Loading Indicator
    const assistantMsgId = `msg_${Date.now()}`;
    appendLoadingAssistantMessage(assistantMsgId);

    try {
      const payload = {
        book_id: currentBookId,
        query: query,
        provider: currentSettings.provider,
        api_key: getActiveApiKey(),
        model: null
      };

      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to generate answer");
      }

      const result = await res.json();
      replaceLoadingAssistantMessage(assistantMsgId, result.answer, result.citations || []);
    } catch (err) {
      replaceLoadingAssistantMessage(
        assistantMsgId,
        `⚠️ **Error:** Unable to retrieve answer: ${err.message}`,
        []
      );
    }
  }

  function getActiveApiKey() {
    if (currentSettings.provider === "gemini") return currentSettings.gemini_api_key;
    if (currentSettings.provider === "openai") return currentSettings.openai_api_key;
    if (currentSettings.provider === "groq") return currentSettings.groq_api_key;
    return "";
  }

  function appendMessage(role, text) {
    const bubble = document.createElement("div");
    bubble.className = `message-bubble message-${role}`;
    bubble.innerHTML = `
      <div class="bubble-avatar">
        <span>${role === "user" ? "YOU" : "AI"}</span>
      </div>
      <div class="bubble-body">
        <div class="bubble-content">
          <p>${escapeHtml(text)}</p>
        </div>
      </div>
    `;
    chatMessagesContainer.appendChild(bubble);
    chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
  }

  function appendLoadingAssistantMessage(msgId) {
    const bubble = document.createElement("div");
    bubble.className = "message-bubble message-assistant";
    bubble.id = msgId;
    bubble.innerHTML = `
      <div class="bubble-avatar">
        <span>AI</span>
      </div>
      <div class="bubble-body">
        <div class="bubble-content">
          <div style="display:flex; align-items:center; gap:8px;">
            <span class="status-spinner"></span>
            <span>Searching 4-Core Knowledge Base & Reranking Passages...</span>
          </div>
        </div>
      </div>
    `;
    chatMessagesContainer.appendChild(bubble);
    chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
  }

  function replaceLoadingAssistantMessage(msgId, answerText, citations) {
    const msgEl = document.getElementById(msgId);
    if (!msgEl) return;

    // Save citations to store for modal previews
    citations.forEach((cite, idx) => {
      const citeId = `${msgId}_cite_${idx}`;
      cite._id = citeId;
      chatCitationsStore[citeId] = cite;
    });

    // Format markdown text and convert citation patterns [Page X] or [Sec Y, Page X] to interactive pills
    const formattedHtml = formatAnswerWithCitations(answerText, citations);

    // Build Verified Citations Tray
    let citationsTrayHtml = "";
    if (citations && citations.length > 0) {
      const pillsHtml = citations.map(c => `
        <button class="citation-pill" data-cite-id="${c._id}">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:12px; height:12px;">
            <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
            <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
          </svg>
          ${c.header_context || "Page " + c.page}
        </button>
      `).join("");

      citationsTrayHtml = `
        <div class="citations-tray">
          <span class="citations-label">Verified Citations:</span>
          ${pillsHtml}
        </div>
      `;
    }

    msgEl.innerHTML = `
      <div class="bubble-avatar">
        <span>AI</span>
      </div>
      <div class="bubble-body">
        <div class="bubble-content">
          ${formattedHtml}
        </div>
        ${citationsTrayHtml}
      </div>
    `;

    // Attach click events to citation pills
    msgEl.querySelectorAll(".citation-pill").forEach(pill => {
      pill.addEventListener("click", () => {
        const citeId = pill.getAttribute("data-cite-id");
        openCitationModal(citeId);
      });
    });

    chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
  }

  function formatAnswerWithCitations(rawText, citations) {
    // 1. Basic Markdown conversion
    let html = rawText
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    // Headers
    html = html.replace(/^### (.*$)/gim, "<h3>$1</h3>");
    html = html.replace(/^## (.*$)/gim, "<h2>$1</h2>");
    html = html.replace(/^# (.*$)/gim, "<h1>$1</h1>");

    // Bold & Italic
    html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*(.*?)\*/g, "<em>$1</em>");

    // Blockquotes
    html = html.replace(/^\> (.*$)/gim, "<blockquote>$1</blockquote>");

    // Bullet points
    html = html.replace(/^\- (.*$)/gim, "<li>$1</li>");
    html = html.replace(/(<li>.*<\/li>)/gims, "<ul>$1</ul>");

    // Convert brackets like [Page X] or [Ch. Y, Page X] into clickable pills if matching citations
    html = html.replace(/\[([^\]]*(?:Page|Sec|Ch)[^\]]*)\]/gi, (match, p1) => {
      // Find matching citation
      const matchCite = citations.find(c =>
        match.toLowerCase().includes(`page ${c.page}`) ||
        (c.section && match.toLowerCase().includes(c.section.toLowerCase()))
      );
      if (matchCite && matchCite._id) {
        return `<button class="citation-pill" data-cite-id="${matchCite._id}">📄 ${p1}</button>`;
      }
      return `<span class="citation-pill">📄 ${p1}</span>`;
    });

    // Paragraphs
    html = html.split("\n\n").map(p => {
      if (p.startsWith("<h") || p.startsWith("<ul>") || p.startsWith("<blockquote>")) return p;
      return `<p>${p.replace(/\n/g, "<br>")}</p>`;
    }).join("");

    return html;
  }

  function openCitationModal(citeId) {
    const cite = chatCitationsStore[citeId];
    if (!cite) return;

    citationModalTitle.textContent = cite.header_context || `Page ${cite.page} Citation`;
    citationModalMeta.innerHTML = `
      <span class="tag">Page: ${cite.page_range || cite.page}</span>
      <span class="tag">Chapter: ${cite.chapter || "N/A"}</span>
      <span class="tag">Section: ${cite.section || "N/A"}</span>
      <span class="tag">RRF Score: ${cite.rrf_score || 0}</span>
    `;
    citationModalContent.textContent = cite.snippet || "No passage excerpt available.";
    citationModal.classList.remove("hidden");
  }

  // --- Event Listeners ---
  function setupEventListeners() {
    // Drop Zone
    dropZone.addEventListener("click", () => pdfFileInput.click());
    pdfFileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
      }
    });

    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
    dropZone.addEventListener("dragleave", () => {
      dropZone.classList.remove("dragover");
    });
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
      if (e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });

    // Preset button
    presetPaperBtn.addEventListener("click", loadPresetPaper);

    // Tabs
    tabBtnChat.addEventListener("click", () => {
      tabBtnChat.classList.add("active");
      tabBtnInspector.classList.remove("active");
      tabChat.classList.add("active");
      tabInspector.classList.remove("active");
    });

    tabBtnInspector.addEventListener("click", () => {
      tabBtnInspector.classList.add("active");
      tabBtnChat.classList.remove("active");
      tabInspector.classList.add("active");
      tabChat.classList.remove("active");
    });

    // Chat form
    chatForm.addEventListener("submit", (e) => {
      e.preventDefault();
      submitQuery();
    });

    chatTextarea.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        submitQuery();
      }
    });

    // Auto-expand textarea
    chatTextarea.addEventListener("input", () => {
      chatTextarea.style.height = "auto";
      chatTextarea.style.height = Math.min(chatTextarea.scrollHeight, 120) + "px";
    });

    // Clear Chat
    clearChatBtn.addEventListener("click", () => {
      chatMessagesContainer.innerHTML = `
        <div class="message-bubble message-assistant">
          <div class="bubble-avatar"><span>AI</span></div>
          <div class="bubble-body">
            <div class="bubble-content">
              <p>Chat cleared. Ask any new question about the active book!</p>
            </div>
          </div>
        </div>
      `;
    });

    // Prompt Chips
    promptsChips.querySelectorAll(".chip").forEach(chip => {
      chip.addEventListener("click", () => {
        const prompt = chip.getAttribute("data-prompt");
        submitQuery(prompt);
      });
    });

    // Modals
    citationModalClose.addEventListener("click", () => citationModal.classList.add("hidden"));
    citationModalOk.addEventListener("click", () => citationModal.classList.add("hidden"));

    openSettingsBtn.addEventListener("click", () => {
      providerSelect.value = currentSettings.provider || "gemini";
      geminiApiKeyInput.value = currentSettings.gemini_api_key || "";
      openaiApiKeyInput.value = currentSettings.openai_api_key || "";
      groqApiKeyInput.value = currentSettings.groq_api_key || "";
      settingsModal.classList.remove("hidden");
    });

    settingsModalClose.addEventListener("click", () => settingsModal.classList.add("hidden"));
    settingsCancelBtn.addEventListener("click", () => settingsModal.classList.add("hidden"));

    settingsSaveBtn.addEventListener("click", async () => {
      currentSettings.provider = providerSelect.value;
      currentSettings.gemini_api_key = geminiApiKeyInput.value.trim();
      currentSettings.openai_api_key = openaiApiKeyInput.value.trim();
      currentSettings.groq_api_key = groqApiKeyInput.value.trim();

      saveLocalSettings();

      try {
        await fetch("/api/settings", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            provider: currentSettings.provider,
            gemini_api_key: currentSettings.gemini_api_key,
            openai_api_key: currentSettings.openai_api_key,
            groq_api_key: currentSettings.groq_api_key
          })
        });
      } catch (err) {
        console.warn("Failed to persist settings on server", err);
      }

      settingsModal.classList.add("hidden");
    });
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  init();
});
