# Graph RAG Pipeline

A general-purpose **Chunk → Knowledge Graph** builder that extracts entities, relationships, and events from text chunks and produces a Graph RAG-ready knowledge graph with Mermaid visualization.

## What It Does

Feed it text chunks (from prospectuses, annual reports, press releases, etc.) and it produces:

1. **Graph JSON** — nodes, edges, and provenance metadata
2. **Mermaid TD** — paste-ready flowchart diagram
3. **Chunk Retrieval** — keyword-based retrieval for RAG queries

## Architecture

```
Text Chunks → LLM Extraction → Entity Resolution → Graph Assembly → Output
                  (Groq)          (Fuzzy Dedup)       (Nodes+Edges)    (JSON + Mermaid)
```

### Pipeline Stages

| Stage | Function | Description |
|-------|----------|-------------|
| Extract | `extract()` | Sends each chunk to Groq LLM with a structured prompt |
| Resolve | `EntityResolver` | Fuzzy deduplication across chunks (type-aware, substring-safe) |
| Build | `build_graph()` | Assembles nodes, edges, events with cross-chunk merging |
| Render | `render_mermaid()` | Generates sectioned Mermaid TD output |
| Retrieve | `retrieve_chunks()` | Keyword overlap scoring for chunk retrieval |

## Ontology

### Node Types

`Company` · `Platform` · `Service` · `Product` · `Feature` · `Capability` · `Partner` · `AcquisitionTarget` · `Event` · `Source` · `Location` · `Person`

### Controlled Relations

`operates` · `offers` · `supported_by` · `enabled_by` · `includes` · `executed_via` · `integrated_surface` · `acquired` · `has_event` · `launched` · `described_in` · `feature` · `benefits_across` · `expanded_surface` · `located_in` · `subsidiary_of` · `invested_in` · `partnered_with`

### Structural Patterns

```
Company ──operates──▶ Platform
Platform ──offers──▶ Service
Platform ──supported_by──▶ Partner
Platform ──enabled_by──▶ Capability
Service ──executed_via──▶ Product
Service ──feature──▶ Feature
Capability ──benefits_across──▶ Service
Company ──acquired──▶ AcquisitionTarget
Company ──has_event──▶ Event
Event ──launched──▶ Product/Service
```

## Key Features

- **General-purpose** — no hardcoded entity names; works for any company or domain
- **Reproducible** — feed new chunks from a different company and the graph builds correctly
- **Dynamic entity resolution** — fuzzy matching with type-gating prevents false merges
- **Cross-chunk event dedup** — same-year events from different chunks merge into one node
- **branded_as merging** — if text says "X branded as Y", entities unify automatically
- **Attribute-driven edges** — `integrated_with` attributes auto-generate `integrated_surface` edges
- **Self-edge blocking** — prevents self-referential edges
- **Rate limit handling** — auto-retries on Groq API 429 errors with parsed wait times

## Setup

### Requirements

- Python 3.10+
- Groq API key ([get one free](https://console.groq.com))

### Install

```bash
pip install requests
```

### Configure

Set your API key as an environment variable:

```bash
# Linux / macOS
export GROQ_API_KEY="your_groq_api_key_here"

# Windows PowerShell
$env:GROQ_API_KEY = "your_groq_api_key_here"
```

Or create a `.env` file (git-ignored):

```
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

### Run with default test chunks

```bash
python main.py
```

### Use with your own chunks

```python
from main import build_graph, render_mermaid, retrieve_chunks
import json

chunks = [
    {"id": "chunk_1", "text": "Your text here..."},
    {"id": "chunk_2", "text": "More text here..."},
]

graph = build_graph(chunks)

# Save JSON
with open("graph_output.json", "w") as f:
    json.dump(graph.to_json(), f, indent=2)

# Save Mermaid
with open("graph_output.mmd", "w") as f:
    f.write(render_mermaid(graph))

# Query chunks
results = retrieve_chunks("What acquisitions were made?", chunks)
```

## Output Files

| File | Description |
|------|-------------|
| `graph_output.json` | Full graph with nodes, edges, attributes, and provenance |
| `graph_output.mmd` | Mermaid TD diagram — paste into any Mermaid renderer |

## Example Mermaid Output

The pipeline produces sectioned Mermaid diagrams with:

- **Core Entities** — Company, Platform, Services, Partners, Capabilities
- **Products / Offerings** — Products and Features with attributes
- **Acquisitions** — Acquisition targets with year and stake
- **Events / Timeline** — Year-grouped events with tags (Launch, Acquisition, Milestone, etc.)
- **Sources** — Provenance tracking back to source chunks
- **Edge groups** — Core Structure, Product Mapping, Acquisitions, Timeline, Provenance

## Tech Stack

- **Python 3** — single-file implementation
- **Groq API** — LLM extraction (llama-3.3-70b-versatile)
- **difflib** — fuzzy entity matching (SequenceMatcher)

## License

MIT
