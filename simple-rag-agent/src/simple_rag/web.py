"""Local-only browser interface. Configure documents when starting the server."""
import argparse
import json
import os
from pathlib import Path
import secrets
from http.server import BaseHTTPRequestHandler, HTTPServer
from .agent import RAGAgent
from .documents import load_documents
from .models import OpenAIModel
from .retrieval import TfidfRetriever

PAGE = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Document Q&A</title><style>
body{font:16px system-ui;max-width:760px;margin:4rem auto;padding:0 1.25rem;color:#172033;background:#f7f8fb}main{background:white;border:1px solid #dce1ea;border-radius:12px;padding:2rem}textarea{box-sizing:border-box;width:100%;min-height:110px;padding:12px;font:inherit;border:1px solid #aeb9c8;border-radius:6px}button{background:#244ccb;color:white;border:0;border-radius:6px;padding:10px 18px;font:inherit;cursor:pointer}button:disabled{opacity:.6}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;line-height:1.5}small{color:#526077}.source{border-top:1px solid #ddd;padding-top:12px}
</style></head><body><main><h1>Ask your documents</h1><p>Ask a question about the documents configured when this app started.</p><form id="form"><label for="question">Your question</label><p><textarea id="question" maxlength="4000" required></textarea></p><button id="submit">Ask</button></form><p id="status" role="status"></p><section id="results" hidden><h2>Answer</h2><pre id="answer"></pre><h2>Retrieved sources</h2><div id="sources"></div></section><small>Questions and retrieved excerpts are sent to the configured model provider.</small></main><script>
const token = "__TOKEN__";
document.getElementById('form').addEventListener('submit', async e => {
 e.preventDefault(); const button=document.getElementById('submit'), status=document.getElementById('status'), results=document.getElementById('results');
 button.disabled=true; status.textContent='Searching documents and generating an answer…'; results.hidden=true;
 try { const response=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json','X-Local-Token':token},body:JSON.stringify({question:document.getElementById('question').value})}); const data=await response.json(); if(!response.ok) throw new Error(data.error);
 document.getElementById('answer').textContent=data.answer; const sources=document.getElementById('sources'); sources.replaceChildren();
 if(!data.sources.length) sources.textContent='No matching sources.';
 for(const source of data.sources){const item=document.createElement('div');item.className='source';const heading=document.createElement('strong');heading.textContent=source.source+' · chunk '+source.index+' · score '+source.score.toFixed(3);const text=document.createElement('pre');text.textContent=source.text;item.append(heading,text);sources.append(item)}
 results.hidden=false;status.textContent=''; } catch(error){status.textContent=error.message || 'Request failed.'} finally{button.disabled=false}
});</script></body></html>'''


def make_handler(agent, token):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, body, content_type="application/json; charset=utf-8"):
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/":
                self.respond(404, '{"error":"Not found"}')
                return
            self.respond(200, PAGE.replace("__TOKEN__", token), "text/html; charset=utf-8")

        def do_POST(self):
            if self.path != "/ask":
                self.respond(404, '{"error":"Not found"}')
                return
            if not secrets.compare_digest(self.headers.get("X-Local-Token", ""), token):
                self.respond(403, '{"error":"Forbidden"}')
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 20000:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
                question = payload.get("question")
                if not isinstance(question, str):
                    raise ValueError("Question must be text")
                result = agent.ask(question)
                body = {"answer": result.text, "sources": [{"source": c.source, "index": c.index, "score": c.score, "text": c.text} for c in result.sources]}
                self.respond(200, json.dumps(body))
            except (ValueError, json.JSONDecodeError):
                self.respond(400, json.dumps({"error": "Enter a question between 1 and 4000 characters."}))
            except Exception:
                self.respond(502, json.dumps({"error": "Unable to generate an answer. Check your model configuration and connection."}))

        def log_message(self, format, *args):
            pass  # Do not log questions or document content.

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Run local document Q&A")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--file", action="append", required=True)
    parser.add_argument("--model", default=os.getenv("RAG_MODEL", "gpt-4o-mini"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    documents = load_documents(args.root, args.file)
    agent = RAGAgent(TfidfRetriever(documents), OpenAIModel(args.model, os.getenv("OPENAI_BASE_URL")), args.top_k)
    server = HTTPServer(("127.0.0.1", args.port), make_handler(agent, secrets.token_urlsafe(32)))
    print(f"Open http://127.0.0.1:{server.server_port} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
