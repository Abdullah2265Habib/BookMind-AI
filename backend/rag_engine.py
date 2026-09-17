import os
import re
import httpx
from typing import List, Dict, Any, Optional
from backend.vector_store import BookVectorStore
from backend.config import (
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    GROQ_API_KEY,
    OPENAI_BASE_URL,
    GROQ_BASE_URL
)

class RAGEngine:
    """
    RAG Engine that connects the book knowledge base with LLM generators.
    Enforces strict grounding and verified page, chapter, and section citations.
    """

    def __init__(self):
        pass

    def retrieve_context(self, book_store: BookVectorStore, query: str) -> List[Dict[str, Any]]:
        """Retrieves and reranks top relevant passages for the query."""
        return book_store.search(query=query)

    def _build_system_prompt(self, book_title: str) -> str:
        return (
            f"You are the dedicated AI Assistant for the book/paper: '{book_title}'.\n"
            "Your role is to provide accurate, comprehensive, and strictly grounded answers "
            "based ONLY on the provided excerpts from the document.\n\n"
            "MANDATORY CITATION RULES:\n"
            "1. Every factual statement, finding, technique, or conclusion must include an inline citation "
            "specifying the chapter/section and page number, in the format: [Page X] or [Sec. Y, Page X] or [Ch. Z, Page X].\n"
            "2. Rely solely on the facts stated in the context passages. Do not fabricate details or assume outside knowledge.\n"
            "3. If the provided context does not contain enough information to answer a question, clearly explain what is missing "
            "and suggest what related topics are covered in the book.\n"
            "4. Maintain a clear, professional, and well-structured formatting with bullet points and bold highlights."
        )

    def _format_context_passages(self, passages: List[Dict[str, Any]]) -> str:
        formatted = []
        for i, p in enumerate(passages):
            header = p.get("header_context", f"Page {p.get('page_number', 1)}")
            formatted.append(f"--- PASSAGE {i+1} [{header}] ---\n{p['content']}\n")
        return "\n".join(formatted)

    async def generate_gemini(self, prompt: str, system_prompt: str, api_key: str, model: str = "gemini-2.0-flash") -> str:
        """Call Google Gemini REST API."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"{system_prompt}\n\n{prompt}"}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 2048
            }
        }
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                raise Exception(f"Gemini API error ({resp.status_code}): {resp.text}")
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            return "No response received from Gemini."

    async def generate_openai_compatible(
        self,
        prompt: str,
        system_prompt: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini"
    ) -> str:
        """Call OpenAI or Groq / Ollama compatible endpoint."""
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                raise Exception(f"API error ({resp.status_code}): {resp.text}")
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def generate_local_grounded(self, query: str, book_title: str, passages: List[Dict[str, Any]]) -> str:
        """
        CPU Grounded Synthesizer:
        If the user has not yet configured an API key, this module extracts,
        synthesizes, and cites direct evidence from the retrieved chunks.
        """
        if not passages:
            return f"I couldn't find any relevant information in **{book_title}** regarding '{query}'."

        tokens = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', query.lower()))
        response_lines = [
            f"### Grounded Findings from *{book_title}*\n",
            f"Based on the semantic analysis and hybrid retrieval of **{book_title}**, here are the most relevant findings:\n"
        ]

        for i, p in enumerate(passages):
            page = p.get("page_number", 1)
            sec = p.get("section", "Overview")
            chap = p.get("chapter", "")
            content = p.get("content", "")

            # Extract highest scoring sentences matching the query
            sentences = re.split(r'(?<=[.?!])\s+', content)
            matched_sentences = []
            for s in sentences:
                s_clean = s.strip()
                s_words = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', s_clean.lower()))
                overlap = len(tokens.intersection(s_words))
                if overlap > 0 and len(s_clean) > 25:
                    matched_sentences.append((overlap, s_clean))

            matched_sentences.sort(key=lambda x: x[0], reverse=True)
            best_snippet = " ".join(s[1] for s in matched_sentences[:3]) if matched_sentences else content[:300] + "..."

            cite_label = f"[Page {page}]"
            if sec and sec != "Overview":
                cite_label = f"[{sec}, Page {page}]"

            response_lines.append(f"**{i+1}. {sec}** {cite_label}")
            response_lines.append(f"> {best_snippet}\n")

        response_lines.append(
            "\n> 💡 *Note: You are currently running in Local Grounded CPU Mode. "
            "To unlock full conversational synthesis and reasoning, add your Gemini, OpenAI, or Groq API key in Model Settings.*"
        )

        return "\n".join(response_lines)

    def generate_custom_llm_response(self, query: str, book_title: str, passages: List[Dict[str, Any]]) -> str:
        """
        Generate a response using the custom from-scratch BookMindGPT model.
        Constructs a context-aware prompt from RAG-retrieved passages and
        runs autoregressive generation through the trained model.
        """
        try:
            from backend.custom_llm import generate_response, get_model_status

            status = get_model_status()
            if status.get("status") != "trained":
                return (
                    "⚠️ **Custom LLM Not Yet Trained**\n\n"
                    "The BookMindGPT model has not been trained yet. "
                    "Please go to the **Custom LLM** tab, upload training PDFs, "
                    "and train the model before using it for chat.\n\n"
                    "> 💡 *Tip: Upload relevant book PDFs to the training folder and click 'Train Model'.*"
                )

            # Build a context-enriched prompt from retrieved passages
            context_parts = []
            for i, p in enumerate(passages[:3]):  # Use top 3 passages to fit in context
                header = p.get("header_context", f"Page {p.get('page_number', 1)}")
                context_parts.append(f"[{header}]: {p['content'][:300]}")

            context_text = "\n".join(context_parts)
            full_prompt = f"Book: {book_title}\n{context_text}\nQuestion: {query}\nAnswer:"

            generated = generate_response(
                prompt=full_prompt,
                max_new_tokens=200,
                temperature=0.7,
                top_k=40,
            )

            if not generated or not generated.strip():
                generated = "(The model generated an empty response. It may need more training data.)"

            # Format response with citations
            response = f"### BookMindGPT Response\n\n"
            response += f"*Generated by the custom from-scratch LLM trained on your book corpus:*\n\n"
            response += generated.strip() + "\n\n"

            # Add passage references
            if passages:
                response += "---\n**Referenced Context:**\n"
                for i, p in enumerate(passages[:3]):
                    page = p.get("page_number", 1)
                    sec = p.get("section", "General")
                    response += f"- [{sec}, Page {page}]\n"

            response += (
                "\n> 💡 *This response was generated entirely by the custom BookMindGPT model "
                "built from scratch (no external APIs). The model quality depends on the "
                "amount and quality of training data provided.*"
            )

            return response

        except FileNotFoundError as e:
            return (
                f"⚠️ **Custom LLM Error:** {str(e)}\n\n"
                "Please train the model first from the Custom LLM tab."
            )
        except Exception as e:
            return f"⚠️ **Custom LLM Generation Error:** {str(e)}"

    async def answer(
        self,
        book_store: BookVectorStore,
        query: str,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main RAG query processing method:
        1. Retrieves & reranks context passages
        2. Dispatches to selected LLM provider (or local synthesizer)
        3. Formats citations and returns grounded response
        """
        passages = self.retrieve_context(book_store, query)
        book_title = book_store.metadata.get("title", "the document")

        # Determine provider and key
        active_provider = provider or "gemini"
        active_key = api_key or ""

        if not active_key:
            if active_provider == "gemini":
                active_key = os.environ.get("GEMINI_API_KEY", "")
            elif active_provider == "openai":
                active_key = os.environ.get("OPENAI_API_KEY", "")
            elif active_provider == "groq":
                active_key = os.environ.get("GROQ_API_KEY", "")

        system_prompt = self._build_system_prompt(book_title)
        formatted_context = self._format_context_passages(passages)
        user_prompt = f"DOCUMENT EXCERPTS:\n{formatted_context}\n\nUSER QUESTION:\n{query}"

        answer_text = ""
        used_provider = active_provider

        try:
            if active_provider == "custom_llm":
                # Use the custom from-scratch BookMindGPT model
                used_provider = "custom_llm"
                answer_text = self.generate_custom_llm_response(query, book_title, passages)
            elif active_provider == "gemini" and active_key:
                gemini_model = model or "gemini-2.0-flash"
                answer_text = await self.generate_gemini(user_prompt, system_prompt, active_key, gemini_model)
            elif active_provider == "openai" and active_key:
                openai_model = model or "gpt-4o-mini"
                answer_text = await self.generate_openai_compatible(
                    user_prompt, system_prompt, active_key, OPENAI_BASE_URL, openai_model
                )
            elif active_provider == "groq" and active_key:
                groq_model = model or "llama-3.3-70b-versatile"
                answer_text = await self.generate_openai_compatible(
                    user_prompt, system_prompt, active_key, GROQ_BASE_URL, groq_model
                )
            else:
                # Fallback to local grounded CPU synthesizer
                used_provider = "local"
                answer_text = self.generate_local_grounded(query, book_title, passages)
        except Exception as e:
            # Graceful fallback to local grounded synthesizer on network/API failure
            used_provider = "local_fallback"
            answer_text = (
                f"*API Connection Notice: {str(e)}*\n\n" +
                self.generate_local_grounded(query, book_title, passages)
            )

        # Parse citations for UI pills
        citations = []
        for p in passages:
            citations.append({
                "chunk_id": p.get("chunk_id"),
                "page": p.get("page_number", 1),
                "page_range": p.get("page_range", str(p.get("page_number", 1))),
                "chapter": p.get("chapter", ""),
                "section": p.get("section", ""),
                "header_context": p.get("header_context", ""),
                "rrf_score": p.get("rrf_score", 0),
                "snippet": p.get("content", "")[:280] + "..."
            })

        return {
            "answer": answer_text,
            "provider": used_provider,
            "citations": citations,
            "passages_count": len(passages)
        }

