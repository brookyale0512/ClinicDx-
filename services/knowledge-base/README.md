# ClinicDx Knowledge Base

A lightweight local HTTP daemon serving clinical knowledge retrieval.

## Indexes

| Index | Source | Search |
|---|---|---|
| `who_knowledge.mv2` | WHO Africa clinical guidelines | Lexical (BM25) |
| `wikimed.mv2` | WikiMed medical reference | Lexical (BM25) |

Built with [memvid](https://github.com/Oaynerad/memvid).

## Running

```bash
pip install memvid-sdk

# Point to your index files
export KB_INDEX_DIR=/path/to/kb/indexes

# Start on port 4276 (default)
python -m kb.daemon 4276
```

## API

### `GET /health`
Returns `{"ok": true}`.

### `GET /stats`
Returns index statistics.

### `POST /search`
```json
{
  "query": "malaria diagnosis criteria",
  "k": 3,
  "snippet_chars": 800,
  "source_mode": "auto",
  "threshold": 0.0,
  "who_first_policy": false,
  "who_failover_threshold": 5.0
}
```

Response:
```json
{
  "ok": true,
  "query": "malaria diagnosis criteria",
  "hit": {
    "score": 18.5,
    "title": "...",
    "content": "...",
    "source": "WHO Guidelines",
    "uri": "",
    "frame_id": "42"
  },
  "source_used": "WHO Guidelines",
  "latency_ms": 12.3
}
```

## Client Usage (Python)

```python
from kb.client import query_kb_http

hit = query_kb_http("malaria treatment Africa", daemon_url="http://localhost:4276")
if hit:
    print(hit["content"])
```
