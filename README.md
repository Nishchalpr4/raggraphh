# Graph RAG Pipeline

Text chunks in → Knowledge graph out. Works for any company.

---

## 1. What This Does

You give it paragraphs of text about a company. It builds a structured knowledge graph — entities, relationships, events, and provenance — and outputs it as JSON and a Mermaid diagram.

**Input:** Text chunks (from IPO prospectus, annual report, press release, etc.)

**Output:**
| File | What's inside |
|------|--------------|
| `graph_output.json` | Every entity, edge, attribute, and which chunk it came from |
| `graph_output.mmd` | A Mermaid flowchart — paste into any Mermaid viewer to see the graph |

---

## 2. How the Pipeline Works

```
Step 1          Step 2              Step 3             Step 4
─────────       ──────────────      ─────────────      ──────────
Text Chunks  →  LLM reads each  →  Merge duplicates → Save JSON
                chunk & extracts    across chunks      + Mermaid
                entities, edges,
                events
```

### Step by step:

1. **Extract** — Each chunk is sent to Groq LLM (llama-3.3-70b). The LLM returns structured JSON: entities with types, relationships between them, and year-grouped events.

2. **Resolve** — The same entity can appear differently across chunks ("Instamart" in one, "Swiggy Instamart" in another). The `EntityResolver` deduplicates them (details below).

3. **Build** — Entities become graph nodes. Relationships become edges. Events from the same year across different chunks merge into a single node.

4. **Output** — The final graph is saved as JSON (for code) and Mermaid (for visualization).

---

## 3. How the Graph Is Structured (Ontology)

The graph models companies the way they describe themselves. Five layers:

### Layer 1 — The Business

```
Company ──operates──▶ Platform
```

- **Company** = the legal/business entity ("Swiggy", "Amazon")
- **Platform** = the system the company built and runs

*Why separate them?* Partners and capabilities attach to the Platform, not to the legal entity. Documents say "our platform offers..." not "our company offers..."

### Layer 2 — What the Platform Offers

```
Platform ──offers──▶ Service ──executed_via──▶ Product
                                                  └──▶ Feature
```

- **Service** = broad business categories ("Food Delivery", "Grocery", "Cloud Compute")
- **Product** = named/branded offerings that deliver a service ("Instamart" delivers Grocery)
- **Feature** = a mode or sub-capability, not standalone ("Bolt" = 10-min mode of Food Delivery)

*Why 3 levels?* "Food Delivery" (the category) ≠ "Swiggy Instamart" (the app) ≠ "Bolt" (the speed mode). Collapsing them loses structure.

### Layer 3 — What Powers the Platform

```
Platform ──supported_by──▶ Partner        (external)
Platform ──enabled_by──▶ Capability       (internal)
```

- **Partner** = external support categories (restaurant partners, delivery fleet, merchants)
- **Capability** = internal enablers (technology, analytics, fulfilment, membership programs)

*Why split?* Documents always distinguish "our partners" from "our technology". One relation for both would make the graph unreadable.

### Layer 4 — Growth & History

```
Company ──acquired──▶ AcquisitionTarget ──expanded_surface──▶ Service
Company ──has_event──▶ Event ──launched──▶ Product
```

- **AcquisitionTarget** = companies that were bought, linked to which Service they expanded
- **Event** = one node per year. All milestones from the same year merge into one node (10 year-nodes, not 50 event-nodes)

### Layer 5 — Provenance

```
Entity ──described_in──▶ Source (chunk)
```

Every entity links back to which text chunk mentioned it. This is what makes it **RAG-ready** — you can trace any graph answer back to the original text.

### All Node Types

| Type | What it represents |
|------|--------------------|
| Company | The main business entity |
| Platform | The unified system/product ecosystem |
| Service | Broad business category (food delivery, grocery) |
| Product | Named/branded offering |
| Feature | Mode or sub-capability of a service/product |
| Capability | Internal enabler (tech, analytics, membership) |
| Partner | External support category (delivery fleet, restaurants) |
| AcquisitionTarget | Acquired company |
| Event | Year-grouped milestone |
| Source | Document/chunk provenance |
| Location | City, country, region |
| Person | Named individual (CEO, founder) |

### All Relations

| Relation | Meaning |
|----------|---------|
| operates | Company runs a Platform |
| offers | Platform provides a Service |
| executed_via | Service is delivered through a Product |
| feature | Service has a Feature (mode) |
| supported_by | Platform is backed by a Partner (external only) |
| enabled_by | Platform is powered by a Capability (internal only) |
| includes | Entity contains a sub-entity |
| benefits_across | Capability spans multiple Services |
| acquired | Company bought an AcquisitionTarget |
| expanded_surface | Acquisition expanded a Service area |
| has_event | Company has a timeline Event |
| launched | Event produced a Product/Service |
| described_in | Entity is mentioned in a Source chunk |
| integrated_surface | Product is integrated with another Product |
| located_in | Entity is in a Location |
| invested_in | Company invested in another entity |
| partnered_with | Company partnered with another entity |
| subsidiary_of | Entity is a subsidiary |
| related_to | Fallback for any other relationship |

---

## 4. How Entity Deduplication Works

Chunks are processed independently, so the same entity can appear with different names:

- Chunk 5: "Instamart"
- Chunk 7: "Swiggy Instamart"

The `EntityResolver` merges them using **3 safety checks**:

### Check 1 — Type Gate

> Only merge entities of the **same type**.

"Swiggy" (Company) will never merge with "Swiggy Limited IPO Prospectus" (Source) — different types, blocked.

### Check 2 — Substring + Length Ratio

> If one name contains the other, merge only if the shorter name is ≥ 50% of the longer name's length.

| Short | Long | Ratio | Result |
|-------|------|-------|--------|
| "Instamart" (9) | "Swiggy Instamart" (16) | 56% | ✅ Merged |
| "Swiggy" (6) | "Swiggy Limited IPO Prospectus" (30) | 20% | ❌ Blocked |

### Check 3 — Fuzzy Similarity

> For non-substring cases, uses string similarity (SequenceMatcher). Threshold: 0.82.

| Name A | Name B | Score | Result |
|--------|--------|-------|--------|
| "Membership Programme" | "Membership Program" | 0.94 | ✅ Merged |
| "Food Delivery" | "Dining Out" | 0.30 | ❌ Not merged |

### Extra Rules

- When two names merge → the **longer** (more descriptive) name becomes canonical
- If an entity has attribute `branded_as=Y` → the entity merges into Y (e.g. "Membership" → "Swiggy One")
- Same-year events from different chunks → merged into **one** Event node

---

## 5. Quick Start

### Install

```bash
pip install requests
```

### Set API Key

Get a free Groq key at [console.groq.com](https://console.groq.com), then:

```bash
# Windows PowerShell
$env:GROQ_API_KEY = "your_key_here"

# Mac / Linux
export GROQ_API_KEY="your_key_here"
```

Or create a `.env` file in the project folder (git-ignored):

```
GROQ_API_KEY=your_key_here
```

### Run

```bash
python main.py
```

---

## 6. Use Your Own Data

```python
from main import build_graph, render_mermaid, retrieve_chunks
import json

chunks = [
    {"id": "chunk_1", "text": "Your company text here..."},
    {"id": "chunk_2", "text": "More text here..."},
]

# Build graph
graph = build_graph(chunks)

# Save outputs
with open("graph_output.json", "w") as f:
    json.dump(graph.to_json(), f, indent=2)

with open("graph_output.mmd", "w") as f:
    f.write(render_mermaid(graph))

# Retrieve relevant chunks for a query
results = retrieve_chunks("What acquisitions were made?", chunks)
```

---

## 7. Tech Stack

| Component | What it does |
|-----------|-------------|
| **Python 3.10+** | Single file, no frameworks needed |
| **Groq API** | LLM extraction using llama-3.3-70b-versatile (free tier) |
| **difflib** | Fuzzy name matching for entity dedup (SequenceMatcher) |
| **requests** | Only external dependency |

---

## License

MIT
