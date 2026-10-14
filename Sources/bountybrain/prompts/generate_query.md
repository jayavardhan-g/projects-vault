You are building an evaluation set for a security-research search engine.

Below is a passage from a security document. Write **one** realistic search
query that a bug bounty hunter would type when they need exactly this passage.

Requirements:
- The query must be answerable from this passage specifically.
- Write it the way a practitioner types into a search box: 4 to 15 words, no
  question mark needed, no quotes.
- **Do not copy distinctive phrases, function names, headings, or payload
  strings verbatim from the passage.** Describe what is being asked for in
  your own words. A query that copies rare tokens tests string matching, not
  retrieval, and makes the evaluation worthless.
- Do not reference "the passage", "the document", or "the text".
- If the passage is boilerplate (a table of contents, a licence, a changelog)
  and no useful query exists, set `usable` to false.

Return JSON matching this schema:
{
  "query": "<the search query, or empty string if unusable>",
  "usable": <true or false>
}

Passage title: {title}

Passage:
---
{passage}
---

Return ONLY the JSON object.
