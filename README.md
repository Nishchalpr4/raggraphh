# Graph RAG Pipeline

Text chunks in → Knowledge graph out. Works for any company.

## How It Works

```
Chunks  →  LLM extracts entities/relations  →  Dedup across chunks  →  Graph JSON + Mermaid diagram
```

1. You give it text chunks (from IPO docs, annual reports, news, etc.)
2. An LLM (Groq) reads each chunk and pulls out entities, relationships, and events
3. Duplicate entities across chunks get merged automatically
4. You get two output files:
   - `graph_output.json` — full graph data
   - `graph_output.mmd` — Mermaid diagram (paste into any viewer)

## Quick Start

```bash
pip install requests
```

Set your Groq API key (free at [console.groq.com](https://console.groq.com)):

```bash
# Windows
$env:GROQ_API_KEY = "your_key_here"

# Mac/Linux
export GROQ_API_KEY="your_key_here"
```

Run:

```bash
python main.py
```

## Use Your Own Data

```python
from main import build_graph, render_mermaid, retrieve_chunks

chunks = [
    {"id": "chunk_1", "text": "Your company text here..."},
    {"id": "chunk_2", "text": "More text here..."},
]

graph = build_graph(chunks)
print(render_mermaid(graph))
```

## How the Graph Is Structured

The graph follows this hierarchy — top to bottom:

```
Company
  └── operates → Platform
                    ├── offers → Service → executed_via → Product → Feature
                    ├── supported_by → Partner        (external: restaurants, delivery fleet)
                    └── enabled_by → Capability       (internal: tech, analytics, membership)
```

Plus:
- **AcquisitionTarget** — acquired companies, linked to which service they expanded
- **Event** — one node per year, all milestones grouped
- **Source** — tracks which chunk each fact came from

## How Dedup Works

Same entity can appear differently across chunks ("Instamart" vs "Swiggy Instamart"). The resolver merges them with 3 checks:

| Check | Rule | Example |
|-------|------|---------|
| **Type match** | Only merge same-type entities | "Swiggy" (Company) ≠ "Swiggy IPO Prospectus" (Source) |
| **Name overlap** | Substring match, but shorter must be ≥50% of longer | "Instamart" in "Swiggy Instamart" → 56% ✅ |
| **Fuzzy match** | String similarity ≥ 0.82 | "Membership Programme" ↔ "Membership Program" → 0.94 ✅ |

If text says "X branded as Y" → X merges into Y automatically.

## Tech

- **Python 3.10+** — single file, no frameworks
- **Groq API** — LLM extraction (llama-3.3-70b-versatile, free tier)
- **difflib** — fuzzy name matching

## License

MIT
