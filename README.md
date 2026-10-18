# Bridge AI

A local-first technical assistant built with Python, Streamlit, Ollama,
and ChromaDB. Use general conversation, your own PDF knowledge library,
or an uploaded PDF as the answer scope.

## Features

- General, Library, Uploaded, and Hybrid answer modes.
- PDF text extraction, local embedding generation, and incremental library sync.
- Persistent vector storage and retrieved-source filenames/page references.
- Selectable assistant profiles and local chat models.
- Ollama and Anthropic provider adapters; OpenAI/Azure adapters are not implemented.

Ollama is the default provider. Anthropic requires separate credentials and
paid API usage. Model output and source attribution are not guaranteed correct.

## Requirements

- **Python 3.13** for this snapshot. Python 3.14 is not a validated environment.
- Ollama installed and running, with sufficient memory for the chosen models.
- Windows setup below uses the Python launcher to select 3.13 explicitly.

The default chat model is `qwen3:4b-instruct-2507-q4_K_M`; `qwen2.5:3b` is an
alternative. Embeddings use `nomic-embed-text`. Check `ollama list` first.
If the default models are not installed, these optional commands download them:

```powershell
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama pull nomic-embed-text
```

Model licenses and hardware requirements apply separately from this code license.

## Local setup (Windows / PowerShell)

From the project directory:

```powershell
py -3.13 -m venv .venv313
.\.venv313\Scripts\python.exe -m pip install -r requirements.txt
.\.venv313\Scripts\python.exe -m streamlit run app.py --server.address=127.0.0.1
```

Use a fresh environment directory if an existing one was created with another
Python version. Do not use an unqualified `python -m venv` that may select 3.14.
No activation or global Python/package changes are required. Open the local URL
printed by Streamlit; stop it with Ctrl+C.

Ollama defaults to the local computer. The `OLLAMA_HOST` process environment
variable controls its endpoint for chat and embeddings; choosing a non-local
endpoint changes where that content is sent. For deliberate Anthropic use, set
`ANTHROPIC_API_KEY` in the process environment before startup and select that
provider in the UI. The application does not automatically load `.env` files.
Never put actual credentials in source code or commit them.

## Add your own knowledge library

An empty library is supported; no PDF corpus is distributed. To add documents,
create category folders under `docs/knowledge` and place text-based PDFs inside:

```text
docs/
  knowledge/
    architecture/
      your-document.pdf
```

The application synchronizes the library for a new Streamlit session. Use
Refresh Knowledge Library after adding or removing files. Use only documents
you are authorized to process. PDFs and runtime data are ignored by Git.

## Answer modes

| Mode | Evidence scope |
| --- | --- |
| General | Model knowledge; no document retrieval |
| Library | PDFs in the local knowledge library |
| Uploaded | The active uploaded PDF |
| Hybrid | Library and active uploaded PDF, when available |

Profiles control the response perspective independently of the answer mode.
Uploaded mode requires an active upload. Check the Model dropdown: an existing
session may retain the prior valid model selection. Clear Chat clears messages,
not model or upload selection. A fresh session may require re-uploading a PDF.

## Local data and privacy

The default local Ollama setup runs inference through your local Ollama server.
Vectors and knowledge metadata are stored under `data/`. Uploaded-document
content may remain in the vector store after a browser session ends.

Selecting a remote provider sends the assembled request, including conversation
and retrieved excerpts, to that provider. Review its terms/data handling first.
Do not commit API keys, private documents, indexes, chat data, or environments.

## Source-only evaluation utilities

`src/evaluation` contains optional evaluation utilities, but this snapshot does
**not** distribute tests, evaluation case datasets, or document fixtures. The
default evaluation file paths therefore do not exist in a fresh checkout.
Running those utilities requires your own schema-compatible case files; the
retrieval evaluator also needs your library and configured embedding model.
`requirements-dev.txt` includes pytest as a development convenience, not a claim
that this snapshot ships a test suite or a reproducible answer-quality score.

## Limitations and security status

- Experimental personal project. Answers can contain factual errors,
  unsupported details, or incorrect attribution. Verify important technical
  claims and generated code against authoritative sources.
- Displayed sources show retrieved material; they do not prove each claim.
- Text-based PDF extraction only: no OCR. Fixed character windows within pages
  can split technical explanations, tables, or code.
- Automatic model routing, fallback, and web tools are not implemented.
- **Dependency vulnerability findings remain unresolved.** This project has
  not been security-audited or hardened for public hosting. Publishing its
  source is not a claim that a public-facing deployment is safe.
- The MIT license does not cover third-party dependencies, model weights,
  or user-supplied documents; their terms apply separately.

## License

MIT License. Copyright (c) 2026 Arvind Gupta. See [LICENSE](LICENSE).
