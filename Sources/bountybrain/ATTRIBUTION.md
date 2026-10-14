# Data attribution

BountyBrain redistributes nothing. `python fetch_data.py` downloads each
dataset from its origin into `data/raw/`, which is gitignored. This file
records what is used, under what terms, and how.

The machine-readable version of this list is
[`bountybrain/datasets/registry.py`](bountybrain/datasets/registry.py), which
carries the licence and attribution for every source alongside its fetch URL.

---

### MITRE CWE — Common Weakness Enumeration
- <https://cwe.mitre.org/>
- Free to use with attribution, under the MITRE Terms of Use.
- CWE™ is a trademark of The MITRE Corporation.
- **Used for:** the canonical weakness vocabulary, its ChildOf/PeerOf/CanPrecede
  hierarchy, alternate terms, consequences, detection methods, mitigations, and
  observed CVE examples.

### MITRE CAPEC — Common Attack Pattern Enumeration and Classification
- <https://capec.mitre.org/>
- Free to use with attribution, under the MITRE Terms of Use.
- CAPEC™ is a trademark of The MITRE Corporation.
- **Used for:** attack-pattern nodes, execution flows, prerequisites, required
  skill levels, mitigations, and the official CAPEC→CWE mapping.

### ProjectDiscovery nuclei-templates
- <https://github.com/projectdiscovery/nuclei-templates> — MIT Licence
- **Used for:** detection-signature nodes and their CWE/CVSS/EPSS/CPE metadata,
  the technology↔weakness bridge, and gold labels for the classification
  evaluation.

### Exploit Database
- <https://gitlab.com/exploit-database/exploitdb> — GPL-2.0
- Maintained by OffSec.
- **Used for:** public exploit availability per CVE. Only the published CSV
  index is read; exploit code is not downloaded or executed.

### pentester.land — List of Bug Bounty Writeups
- <https://pentester.land/>
- **Used for:** disclosed-writeup records — titles, programs, curated
  vulnerability labels, bounty amounts, dates.
- **Only the published JSON metadata feed is read** — a single request. The
  6,421 linked articles live on third-party blogs and are never crawled. No
  article text is stored or redistributed.

### PayloadsAllTheThings
- <https://github.com/swisskyrepo/PayloadsAllTheThings> — MIT Licence
- **Used for:** practitioner technique and payload prose in the retrieval corpus.

### OWASP Web Security Testing Guide
- <https://github.com/OWASP/wstg> — CC-BY-SA-4.0
- **Used for:** structured testing procedures in the retrieval corpus.
- Note the share-alike term: WSTG content reproduced downstream must carry the
  same licence.

### Bugcrowd Vulnerability Rating Taxonomy
- <https://github.com/bugcrowd/vulnerability-rating-taxonomy> — Apache-2.0
- **Used for:** P1–P5 payout priority per weakness class, via the official
  VRT→CWE mapping.

---

### Models

- `BAAI/bge-small-en-v1.5` (MIT) — passage and query embeddings.
- `BAAI/bge-reranker-base` (MIT) — cross-encoder reranking.
- `Qwen/Qwen2.5-14B-Instruct` (Apache-2.0) — the default local generation model,
  served with vLLM. Swappable via `LLM_MODEL`.

### This project

The BountyBrain source code is MIT licensed. See [LICENSE](LICENSE).
