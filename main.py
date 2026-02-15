"""
Graph RAG Pipeline – General-purpose Chunk → Knowledge Graph builder.

Works for ANY company / domain.  No hardcoded entity names.
Feed it text chunks and it produces:
  (A) Graph JSON  (nodes + edges + provenance)
  (B) Mermaid TD  (paste-ready)
  (C) Simple chunk retrieval by keyword query
"""

import os
import re
import json
import time
import requests
from collections import defaultdict
from difflib import SequenceMatcher
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# ============================================================
# CONFIG  (reads from env var or .env file)
# ============================================================

def _load_env():
    """Load .env file if present (no extra dependencies needed)."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise RuntimeError("Set GROQ_API_KEY as an environment variable or in a .env file")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ============================================================
# ONTOLOGY  (domain-agnostic)
# ============================================================

NODE_TYPES = {
    "Company", "Platform", "Service", "Product", "Feature",
    "Capability", "Partner", "AcquisitionTarget", "Event",
    "Source", "Location", "Person", "Regulation",
}

CONTROLLED_RELATIONS = {
    "operates", "offers", "supported_by", "enabled_by",
    "includes", "executed_via", "integrated_surface",
    "acquired", "has_event", "launched", "described_in",
    "feature", "events_experiences_surface", "benefits_across",
    "expanded_surface", "related_to", "located_in",
    "subsidiary_of", "invested_in", "partnered_with",
}

RELATION_NORMALIZATION = {
    # Structure
    "built": "operates", "runs": "operates", "manages": "operates",
    "provides": "offers", "delivers": "offers",
    "supports": "supported_by", "backed_by": "supported_by",
    "powered_by": "enabled_by", "underpinned_by": "enabled_by",
    "underpinned by": "enabled_by",
    "partners with": "supported_by", "partners_with": "supported_by",
    # Launch / expansion
    "expanded_into": "launched", "rolled_out": "launched",
    "released": "launched", "introduced": "launched",
    "expands": "launched", "expanded": "launched",
    "scaled": "launched", "pioneered": "launched",
    # Hierarchy
    "is part of": "includes", "is_part_of": "includes",
    "part_of": "includes", "has": "includes",
    "branded_as": "includes", "consists_of": "includes",
    # Acquisition
    "acquisition_of": "acquired", "bought": "acquired",
    "merged_with": "acquired",
    # Events
    "completed": "has_event",
    # Location
    "headquartered_in": "located_in", "based_in": "located_in",
    "incorporated_in": "located_in",
}

# ============================================================
# LLM TYPE ALIASES  (normalises LLM output variations)
# ============================================================

TYPE_ALIASES = {
    "Membership": "Capability",
    "Document": "Source",
    "Organization": "Company",
    "Organisation": "Company",
    "City": "Location",
    "Place": "Location",
    "Region": "Location",
    "Country": "Location",
    "Geography": "Location",
    "Program": "Capability",
    "Programme": "Capability",
    "Subsidiary": "Company",
    "Brand": "Product",
    "Regulation": "Source",
}

# ============================================================
# DYNAMIC ENTITY RESOLUTION  (fuzzy dedup, no hardcoding)
# ============================================================

class EntityResolver:
    """
    Maintains a registry of canonical entity names.
    When a new name arrives, it checks for near-duplicates
    (e.g. "Instamart" vs "Swiggy Instamart") and merges them.
    Only merges entities with COMPATIBLE types.
    """

    # Types that can be merged with each other
    COMPATIBLE_TYPES = {
        frozenset({"Company"}),
        frozenset({"Platform"}),
        frozenset({"Service"}),
        frozenset({"Product", "Feature"}),
        frozenset({"Capability"}),
        frozenset({"Partner"}),
        frozenset({"AcquisitionTarget"}),
        frozenset({"Location"}),
        frozenset({"Person"}),
        frozenset({"Source"}),
    }

    def __init__(self, similarity_threshold: float = 0.82):
        self.threshold = similarity_threshold
        # canonical_name → node_type
        self.registry: Dict[str, str] = {}
        # lowered alias → canonical_name
        self._alias_map: Dict[str, str] = {}

    def _normalise_key(self, name: str) -> str:
        return re.sub(r"\s+", " ", name.lower().strip())

    def _similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def _types_compatible(self, type_a: str, type_b: str) -> bool:
        """Check if two types can logically be merged."""
        if type_a == type_b:
            return True
        for group in self.COMPATIBLE_TYPES:
            if type_a in group and type_b in group:
                return True
        return False

    def _is_meaningful_substring(self, short: str, long: str) -> bool:
        """
        True when one is a suffix/core of the other and they are close enough.
        Guards against false merges like 'Swiggy' ↔ 'Swiggy Limited IPO Prospectus'.
        The short name must be at least 60% of the long name's length.
        """
        s = short.lower().strip()
        l = long.lower().strip()
        if len(s) < 3 or len(l) < 3:
            return False
        # Must be a real substring
        if s not in l and l not in s:
            return False
        shorter, longer = (s, l) if len(s) <= len(l) else (l, s)
        # Require the shorter to be at least 50% of the longer
        if len(shorter) / len(longer) < 0.50:
            return False
        return True

    def resolve(self, raw_name: str, node_type: str) -> Tuple[str, str]:
        """
        Returns (canonical_name, resolved_type).
        If the name closely matches an existing entity OF A COMPATIBLE TYPE,
        reuse that. Otherwise register as new.
        """
        # Normalise type first
        if node_type not in NODE_TYPES:
            node_type = TYPE_ALIASES.get(node_type, "Product")

        key = self._normalise_key(raw_name)

        # Exact hit
        if key in self._alias_map:
            canon = self._alias_map[key]
            return canon, self.registry[canon]

        # Fuzzy / substring match against existing (type-gated)
        best_match = None
        best_score = 0.0
        for existing_key, canon in self._alias_map.items():
            existing_type = self.registry.get(canon, "Product")
            if not self._types_compatible(node_type, existing_type):
                continue  # never merge across incompatible types

            if self._is_meaningful_substring(key, existing_key):
                score = 0.92
            else:
                score = self._similarity(key, existing_key)

            if score > best_score:
                best_score = score
                best_match = canon

        if best_match and best_score >= self.threshold:
            # Prefer the longer (more descriptive) name as canonical
            if len(raw_name.strip()) > len(best_match):
                old_canon = best_match
                new_canon = raw_name.strip()
                for k, v in list(self._alias_map.items()):
                    if v == old_canon:
                        self._alias_map[k] = new_canon
                self.registry[new_canon] = self.registry.pop(old_canon, node_type)
                self._alias_map[key] = new_canon
                return new_canon, self.registry[new_canon]
            else:
                self._alias_map[key] = best_match
                return best_match, self.registry[best_match]

        # New entity
        canon = raw_name.strip()
        self.registry[canon] = node_type
        self._alias_map[key] = canon
        return canon, node_type


# ============================================================
# CANONICAL ID SYSTEM
# ============================================================

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text


def canonical_id(name: str, node_type: str) -> str:
    return f"{normalize_text(node_type)}_{normalize_text(name)}"


# ============================================================
# GRAPH CLASS
# ============================================================

class Graph:
    def __init__(self):
        self.nodes: Dict[str, dict] = {}
        self.edges: list = []
        self.edge_set: set = set()
        self.node_provenance: Dict[str, set] = defaultdict(set)

    def add_node(self, name: str, node_type: str,
                 attributes: Optional[dict] = None,
                 source: Optional[str] = None) -> str:
        # Normalise LLM type variations
        if node_type not in NODE_TYPES:
            node_type = TYPE_ALIASES.get(node_type, "Product")

        node_id = canonical_id(name, node_type)

        if node_id not in self.nodes:
            self.nodes[node_id] = {
                "id": node_id,
                "label": name,
                "type": node_type,
                "attributes": attributes or {},
            }
        if attributes:
            self.nodes[node_id]["attributes"].update(
                {k: v for k, v in attributes.items() if v}
            )
        if source:
            self.node_provenance[node_id].add(source)
        return node_id

    def _norm_rel(self, relation: str) -> str:
        relation = relation.lower().strip()
        return RELATION_NORMALIZATION.get(relation, relation)

    def add_edge(self, src_id: str, relation: str, tgt_id: str,
                 source: Optional[str] = None):
        if src_id == tgt_id:          # block self-referential edges
            return
        relation = self._norm_rel(relation)
        if relation not in CONTROLLED_RELATIONS:
            relation = "related_to"
        key = (src_id, relation, tgt_id)
        if key in self.edge_set:
            return
        self.edge_set.add(key)
        self.edges.append({
            "source": src_id,
            "relation": relation,
            "target": tgt_id,
            "provenance": [source] if source else [],
        })

    def to_json(self) -> dict:
        return {
            "nodes": sorted(self.nodes.values(), key=lambda x: x["id"]),
            "edges": sorted(
                self.edges,
                key=lambda x: (x["source"], x["relation"], x["target"])
            ),
        }


# ============================================================
# LLM EXTRACTION  (general-purpose prompt)
# ============================================================

SYSTEM_PROMPT = """You are a precise information extraction system for building knowledge graphs from company documents (prospectuses, annual reports, press releases, etc.).

Extract structured JSON with this exact schema:

{
  "entities": [
    {
      "name": "...",
      "type": "Company|Platform|Service|Product|Feature|Capability|Partner|AcquisitionTarget|Location|Person|Source",
      "attributes": {"key": "value"}
    }
  ],
  "relations": [
    {"source": "entity name", "relation": "verb", "target": "entity name"}
  ],
  "events": [
    {
      "year": "YYYY",
      "description": "concise summary of everything that happened that year",
      "tags": ["Launch","Acquisition","Milestone","Fundraise","Corporate","Scale","Listing","Pioneer"],
      "event_relations": [
        {"relation": "launched|acquired", "target": "entity name"}
      ]
    }
  ]
}

=== CRITICAL: ENTITY DEDUPLICATION ===
- Use ONE consistent name for each entity across the entire output.
- If the company is called both "X" and "X Limited", pick ONE name (prefer the shorter common name).
- If a platform is described in different words, pick ONE name for it.
- Do NOT create separate entities for the same thing described differently.
- If a concept is described as "branded as Y" or "known as Y", use the BRANDED NAME (Y) as the entity name. Do NOT create both the generic concept AND the branded version — emit ONLY the branded name.
- Membership benefits ("10-minute delivery", "restaurant reservations", etc.) are NOT separate Services.
  They describe what the membership offers across existing Service categories. Do NOT extract them as individual Service entities.
  Instead, list them as a comma-separated string in the membership entity's "benefits_across" attribute.

ENTITY TYPING RULES (general, not company-specific):
- The main company being described → "Company" (use its common/short name)
- Companies that were ACQUIRED by the main company → "AcquisitionTarget" (add attribute acquired_year)
- A unified/consumer platform described as a distinct concept → "Platform" (use ONE Platform if text describes one platform)
- Broad service CATEGORIES (e.g. food delivery, dining, grocery, logistics) → "Service" (only top-level categories, not sub-features)
- Named products, apps, branded offerings with proper names → "Product"
- A sub-feature or mode of an existing service/product (e.g. a fast-delivery mode) → "Feature"
- Technology, analytics, membership programs, fulfilment capabilities → "Capability"
- Partner categories (restaurant partners, delivery fleet, merchants) → "Partner"
- City / country / region names → "Location" (NEVER "Company")
- Named individuals (CEO, founder, etc.) → "Person"
- Document sources (prospectus, annual report, filings) → "Source" (NOT "Regulation")

ATTRIBUTE RULES – capture EXACTLY what the text states, no reinterpretation:
- launch_year, scale, integrated_with, stake, member_count
- For Company: Strategy (verbatim from text), Culture (verbatim from text, e.g. "innovation-led" NOT "customer convenience"), founded_year
- For Platform: Principle, ValueProps (use exact terms from text)
- For Platform: do NOT invent Capability values — only use capability names explicitly listed in the text
- For AcquisitionTarget: acquired_year, stake
- For Product: if the text says a product is "integrated with" another product, add attribute integrated_with=<other product name>

RELATION RULES – use ONLY these verbs:
operates, offers, supported_by, enabled_by, includes,
executed_via, integrated_surface, acquired, has_event, launched,
described_in, feature, events_experiences_surface, benefits_across,
expanded_surface, related_to, located_in, subsidiary_of,
invested_in, partnered_with

CRITICAL RELATION CONSTRAINTS:
- supported_by is ONLY for Partner entities. NEVER use supported_by for Capability entities.
- enabled_by is for Capability entities. Platform enabled_by Capability.
- Map each product to the SERVICE it actually delivers (e.g. a grocery product → grocery service, NOT food delivery service).
- If a Feature is a mode of a Service (e.g. fast-delivery mode of food delivery), emit: Service feature Feature.
- If an AcquisitionTarget expanded a service area, emit: AcquisitionTarget expanded_surface Service.
- If a Product is integrated with another Product, emit: Product integrated_surface Product.

KEY STRUCTURAL PATTERNS (apply when the text supports them):
- Company operates Platform
- Platform offers Service  (route through Platform when it exists, NOT directly from Company)
- Platform supported_by Partner  (ONLY Partners)
- Platform enabled_by Capability  (ONLY Capabilities)
- Service executed_via Product  (the product is the vehicle/app for delivering that service)
- Service includes Product  (the product is part of a service category)
- Service integrated_surface Product  (a product is an integrated surface of a service)
- Service feature Feature  (a feature/mode of a service, e.g. fast-delivery mode)
- Capability benefits_across Service  (membership spanning services)
- Capability includes Product  (products bundled under a capability, e.g. credit card, quick bites)
- Company acquired AcquisitionTarget
- AcquisitionTarget expanded_surface Service  (acquisition expanded a service area, e.g. dining)
- Company located_in Location  (extract city names mentioned as locations)

EVENT RULES:
- Group ALL things from the SAME YEAR into ONE event entry
- Include relevant tags: Corporate, Launch, Pioneer, Fundraise, Milestone, Scale, Acquisition, Listing
- In event_relations use "launched" for launches and "acquired" for acquisitions
- Do NOT create events with year=None. Only extract events that have a clear year.

IMPORTANT:
- Do NOT invent facts not present in the text
- Use neutral verbs; do not over-claim
- Extract city names mentioned in the text as Location entities
- Return ONLY valid JSON – no markdown fences, no commentary"""


def call_groq(chunk_text: str) -> dict:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": chunk_text},
        ],
        "temperature": 0,
    }

    resp = requests.post(GROQ_URL, headers=headers, json=payload)

    # Handle rate limiting with retry
    retries = 0
    while resp.status_code == 429 and retries < 5:
        # Parse wait time from error message
        err_text = resp.text
        wait_match = re.search(r"try again in (\d+)m([\d.]+)s", err_text)
        if wait_match:
            wait_secs = int(wait_match.group(1)) * 60 + float(wait_match.group(2))
        else:
            wait_secs = 60 * (retries + 1)
        wait_secs = min(wait_secs + 5, 1200)  # cap at 20 min, add buffer
        print(f"    Rate limited. Waiting {wait_secs:.0f}s before retry...")
        time.sleep(wait_secs)
        resp = requests.post(GROQ_URL, headers=headers, json=payload)
        retries += 1

    if resp.status_code != 200:
        raise Exception(f"Groq API error {resp.status_code}: {resp.text}")

    content = resp.json()["choices"][0]["message"]["content"]

    # Strip markdown fences if the LLM wraps output
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except Exception:
        raise Exception("LLM did not return valid JSON:\n" + content)


# ============================================================
# EXTRACT  –  per-chunk extraction
# ============================================================

def extract(chunk_text: str) -> dict:
    """Extract entities, relations, and events from a single chunk."""
    return call_groq(chunk_text)


# ============================================================
# BUILD_GRAPH  –  assemble graph from multiple chunks
# ============================================================

def build_graph(chunks: List[Dict]) -> Graph:
    graph = Graph()
    resolver = EntityResolver(similarity_threshold=0.82)
    global_name_to_id: Dict[str, str] = {}
    year_event_ids: Dict[str, str] = {}   # year → event node_id (dedup)

    for chunk in chunks:
        source_id = chunk["id"]
        text = chunk["text"]
        print(f"  Extracting {source_id} ...")

        extraction = extract(text)
        local_name_to_id: Dict[str, str] = {}

        # -- Source node --
        src_node_id = graph.add_node(
            source_id, "Source",
            attributes={"chunk_id": source_id},
        )

        # -- Entities --
        for ent in extraction.get("entities", []):
            raw_name = ent["name"]
            ent_type = ent.get("type", "Product")
            attrs = ent.get("attributes", {})

            # Resolve via dynamic dedup
            canon_name, resolved_type = resolver.resolve(raw_name, ent_type)
            node_id = graph.add_node(canon_name, resolved_type,
                                     attributes=attrs, source=source_id)

            local_name_to_id[raw_name] = node_id
            local_name_to_id[canon_name] = node_id
            global_name_to_id[raw_name] = node_id
            global_name_to_id[canon_name] = node_id

            graph.add_edge(node_id, "described_in", src_node_id)

        # -- branded_as dedup: merge generic + branded into one entity --
        for ent in extraction.get("entities", []):
            attrs = ent.get("attributes", {})
            branded = attrs.get("branded_as", "")
            if branded:
                raw_name = ent["name"]
                branded_canon, branded_type = resolver.resolve(branded, ent.get("type", "Product"))
                generic_canon, _ = resolver.resolve(raw_name, ent.get("type", "Product"))
                generic_id = local_name_to_id.get(generic_canon) or global_name_to_id.get(generic_canon)
                branded_id = local_name_to_id.get(branded_canon) or global_name_to_id.get(branded_canon)
                if generic_id and branded_id and generic_id != branded_id:
                    # Rewire all edges from generic → branded
                    for edge in graph.edges:
                        if edge["source"] == generic_id:
                            edge["source"] = branded_id
                        if edge["target"] == generic_id:
                            edge["target"] = branded_id
                    # Merge attributes into branded node
                    if generic_id in graph.nodes and branded_id in graph.nodes:
                        graph.nodes[branded_id]["attributes"].update(
                            {k: v for k, v in graph.nodes[generic_id]["attributes"].items() if v}
                        )
                    # Remove generic node
                    graph.nodes.pop(generic_id, None)
                    # Update maps
                    for name, nid in list(local_name_to_id.items()):
                        if nid == generic_id:
                            local_name_to_id[name] = branded_id
                    for name, nid in list(global_name_to_id.items()):
                        if nid == generic_id:
                            global_name_to_id[name] = branded_id

        # -- Relations --
        for rel in extraction.get("relations", []):
            s, t = rel["source"], rel["target"]
            sid = local_name_to_id.get(s) or global_name_to_id.get(s)
            tid = local_name_to_id.get(t) or global_name_to_id.get(t)

            # Try resolved names if raw didn't match
            if not sid:
                cn, _ = resolver.resolve(s, "Product")
                sid = local_name_to_id.get(cn) or global_name_to_id.get(cn)
            if not tid:
                cn, _ = resolver.resolve(t, "Product")
                tid = local_name_to_id.get(cn) or global_name_to_id.get(cn)

            if sid and tid:
                graph.add_edge(sid, rel["relation"], tid, source_id)

        # -- Events (year-grouped, deduplicated across chunks) --
        for ev in extraction.get("events", []):
            year = str(ev.get("year", ""))
            if not year or year == "None":
                continue
            desc = ev.get("description", f"Events of {year}")
            tags = ev.get("tags", [])

            event_attrs = {"year": year}
            if tags:
                event_attrs["tags"] = ", ".join(tags)

            if year in year_event_ids:
                # Merge into existing event node for this year
                event_id = year_event_ids[year]
                existing = graph.nodes[event_id]
                # Merge description
                old_desc = existing["label"]
                if desc not in old_desc:
                    merged_desc = f"{old_desc}; {desc}"
                    existing["label"] = merged_desc
                # Merge tags
                if tags:
                    old_tags = existing["attributes"].get("tags", "")
                    new_tags = set(old_tags.split(", ")) | set(tags)
                    existing["attributes"]["tags"] = ", ".join(sorted(new_tags))
                graph.node_provenance[event_id].add(source_id)
            else:
                event_label = f"{desc} ({year})"
                event_id = graph.add_node(event_label, "Event",
                                          attributes=event_attrs, source=source_id)
                year_event_ids[year] = event_id

            # Link Company → has_event → Event
            for nid, node in graph.nodes.items():
                if node["type"] == "Company":
                    graph.add_edge(nid, "has_event", event_id, source_id)
                    break

            # Event → launched/acquired → entity
            for er in ev.get("event_relations", []):
                tgt = er["target"]
                tid = local_name_to_id.get(tgt) or global_name_to_id.get(tgt)
                if not tid:
                    cn, _ = resolver.resolve(tgt, "Product")
                    tid = local_name_to_id.get(cn) or global_name_to_id.get(cn)
                if tid:
                    graph.add_edge(event_id, er["relation"], tid, source_id)

    # -- Post-processing: attribute-driven edge injection --
    for node_id, node in list(graph.nodes.items()):
        attrs = node.get("attributes", {})
        integrated_with = attrs.get("integrated_with", "")
        if integrated_with:
            # Find the target entity
            target_canon, _ = resolver.resolve(integrated_with, "Product")
            target_id = global_name_to_id.get(target_canon)
            if target_id and target_id != node_id:
                graph.add_edge(node_id, "integrated_surface", target_id)

    return graph


# ============================================================
# RENDER_MERMAID  –  graph JSON → Mermaid TD text
# ============================================================

def _mermaid_safe(text: str) -> str:
    return text.replace('"', "'").replace("#", "&#35;")


def render_mermaid(graph: Graph) -> str:
    gj = graph.to_json()
    lines = ["graph TD"]

    # ── Node sections ──
    by_type: Dict[str, list] = defaultdict(list)
    for node in gj["nodes"]:
        by_type[node["type"]].append(node)

    sections = [
        ("Core Entities",        ["Company", "Platform", "Service", "Partner", "Capability"]),
        ("Products / Offerings", ["Product", "Feature"]),
        ("Acquisitions",         ["AcquisitionTarget"]),
        ("Events / Timeline",    ["Event"]),
        ("Sources",              ["Source"]),
        ("Locations",            ["Location"]),
        ("People",               ["Person"]),
        ("Regulations",          ["Regulation"]),
    ]

    for sec_name, types in sections:
        has = any(by_type.get(t) for t in types)
        if not has:
            continue
        lines.append(f"    %% =========================")
        lines.append(f"    %% {sec_name}")
        lines.append(f"    %% =========================")
        for ntype in types:
            for node in by_type.get(ntype, []):
                nid = node["id"]
                label_parts = [f'{node["type"]}: {node["label"]}']
                attrs = {k: v for k, v in node.get("attributes", {}).items()
                         if k != "chunk_id" and v}
                if attrs:
                    attr_str = "; ".join(f"{k}={v}" for k, v in attrs.items())
                    label_parts.append(f"<br/><b>Attributes</b>: {attr_str}")
                full = _mermaid_safe("".join(label_parts))
                lines.append(f'    {nid}["{full}"]')

    # ── Edge sections ──
    rel_groups = [
        ("Core Structure",   {"operates", "offers", "supported_by", "enabled_by"}),
        ("Product Mapping",  {"executed_via", "includes", "integrated_surface",
                              "feature", "events_experiences_surface", "benefits_across"}),
        ("Acquisitions",     {"acquired", "expanded_surface"}),
        ("Timeline",         {"has_event", "launched"}),
        ("Provenance",       {"described_in"}),
    ]

    covered = set()
    for group_name, rels in rel_groups:
        group_edges = [e for e in gj["edges"] if e["relation"] in rels]
        if not group_edges:
            continue
        lines.append(f"    %% =========================")
        lines.append(f"    %% {group_name}")
        lines.append(f"    %% =========================")
        for e in group_edges:
            lines.append(f'    {e["source"]} -->|{e["relation"]}| {e["target"]}')
        covered |= rels

    # Remaining
    remaining = [e for e in gj["edges"] if e["relation"] not in covered]
    if remaining:
        lines.append("    %% Other")
        for e in remaining:
            lines.append(f'    {e["source"]} -->|{e["relation"]}| {e["target"]}')

    return "\n".join(lines)


# ============================================================
# RETRIEVE_CHUNKS  –  simple keyword retrieval
# ============================================================

def retrieve_chunks(query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
    """
    Retrieve the most relevant chunks for a user query.
    Uses keyword overlap scoring (works without embeddings).
    """
    query_tokens = set(re.findall(r"\w+", query.lower()))
    scored = []
    for chunk in chunks:
        text_tokens = set(re.findall(r"\w+", chunk["text"].lower()))
        if not query_tokens:
            continue
        overlap = len(query_tokens & text_tokens)
        score = overlap / len(query_tokens)
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k] if _ > 0]


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    chunks = [
        {
            "id": "chunk_5",
            "text": (
                "Swiggy is a new-age, consumer-first technology company offering users an easy-to-use "
                "unified platform, structured as a consumer convenience platform. This platform "
                "encompasses four primary service categories: Food delivery, Dining out and Events, "
                "Grocery and Household items delivery, and Other hyperlocal services. Supporting these "
                "services are four partner categories: Restaurant partners, Merchant partners, Brand "
                "partners, and Delivery partners. Underpinning the entire ecosystem are four "
                "foundational capabilities: Technology, Analytics, Fulfilment, and Membership, with the "
                "Membership component branded as \"Swiggy one.\" The source of this information is the "
                "Swiggy Limited IPO Prospectus."
            ),
        },
        {
            "id": "chunk_6",
            "text": (
                "Swiggy positions itself as a pioneer in high-frequency hyperlocal commerce categories, "
                "driven by an innovation-led culture that is described as an integral part of its DNA, "
                "encouraging constant ideation, experimentation, and iteration. The company "
                "emphasizes a convenience-first approach, faster delivery times, product quality, "
                "assortments, and personalized recommendations. Its primary app and underlying "
                "reusable tech stack enable quick and low-cost innovations. Swiggy was a pioneer in "
                "Food Delivery in 2014 and in Quick Commerce in 2020. It has a successful track record "
                "of scaling up businesses, exemplified by Instamart, which scaled to 124 cities as of 31 "
                "March 2025 from 2 cities in less than 5 years of launch. Strategic acquisitions, such as "
                "Dineout, have expanded its platform capabilities. The company offers a membership "
                "programme providing benefits across its offerings, including 10-minute food delivery, "
                "food delivery from restaurants, quick delivery of groceries and household items, "
                "restaurant reservations and payments, events and experiences, a co-branded credit "
                "card, and Quick Bites."
            ),
        },
        {
            "id": "chunk_7",
            "text": (
                "The urban convenience platform was built by adding adjacent services over a decade, "
                "beginning with the incorporation of the company in 2013 under the name Bengaluru, "
                "followed by the launch of the Food Delivery business in 2014. In 2015, the company "
                "completed its first major fundraise. The platform expanded in 2019 with the expansion "
                "of the food delivery business to cover over 500 cities. In 2020, Swiggy Instamart and "
                "Swiggy Genie were launched. The year 2021 saw the launch of the membership "
                "program, Swiggy One. In 2022, the company acquired and integrated Dineout, "
                "expanded Swiggy Instamart to cover 25 cities with over 400 dark stores and over 8,400 "
                "SKUs, and launched Swiggy Minis. In 2023, the company launched the Swiggy-HDFC "
                "Bank co-branded credit card, acquired a 100% stake in Lynk, and launched Swiggy "
                "Mall, now integrated with Instamart. In 2024, the Swiggy One membership base crossed "
                "5.7 million members, the company completed its public listing, launched Bolt for 10-"
                "minute food delivery, and launched Swiggy Scenes for events and experiences. In "
                "2025, the platform surpassed 120 million transacted users and launched SNACC and "
                "Pyng. The source of this information is the Swiggy Limited IPO Prospectus, Annual "
                "Report FY 2024-25."
            ),
        },
    ]

    print("=" * 60)
    print("  Graph RAG Pipeline – Building Knowledge Graph")
    print("=" * 60)

    graph = build_graph(chunks)

    # ── Save Graph JSON ──
    graph_json_str = json.dumps(graph.to_json(), indent=2)
    with open("graph_output.json", "w", encoding="utf-8") as f:
        f.write(graph_json_str)
    print("\n  Saved: graph_output.json")

    # ── Save Mermaid ──
    mermaid_text = render_mermaid(graph)
    with open("graph_output.mmd", "w", encoding="utf-8") as f:
        f.write(mermaid_text)
    print("  Saved: graph_output.mmd")

    # ── Print outputs ──
    print("\n" + "=" * 60)
    print("  GRAPH JSON")
    print("=" * 60)
    print(graph_json_str)

    print("\n" + "=" * 60)
    print("  MERMAID (paste-ready)")
    print("=" * 60)
    print(mermaid_text)

    # ── Demo: chunk retrieval ──
    print("\n" + "=" * 60)
    print("  CHUNK RETRIEVAL DEMO")
    print("=" * 60)
    demo_query = "What acquisitions did the company make?"
    print(f"  Query: \"{demo_query}\"")
    results = retrieve_chunks(demo_query, chunks)
    for r in results:
        print(f"    -> {r['id']}: {r['text'][:100]}...")
