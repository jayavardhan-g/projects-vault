You are writing a tutorial for someone learning bug bounty hunting, on the
topic: **{topic}**.

You have two inputs: numbered source excerpts retrieved from a local corpus,
and structured facts from a knowledge graph built on MITRE CWE and CAPEC.

**Citation rules — these matter more than style:**
- Every technical claim must cite a source by number, like `[3]`.
- Cite only the numbers that appear below. Never invent a citation.
- If the sources do not support a claim, leave the claim out. An incomplete
  tutorial is fine; a confident wrong one is not.
- Do not invent payloads, CVE numbers, bounty amounts, or program names.

Knowledge graph facts (use for framing; still cite sources for claims):
{kg_context}

Sources:
{sources}

Structure:

1. **What it is** — one plain-language paragraph.
2. **Why it happens** — the underlying flaw, in beginner terms.
3. **How to find it** — concrete techniques, each cited.
4. **A worked example** — walk through one real case from the sources, cited.
5. **How to test it safely** — stress that this belongs in a lab or on a
   target explicitly in scope for a program you are authorised to test.
6. **Key takeaways** — three bullets.

Keep the vocabulary simple and the voice active. Return markdown only.
