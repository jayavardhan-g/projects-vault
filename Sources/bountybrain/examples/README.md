# Example recon output

These are **synthetic** files, hand-written to exercise the parsers and the
planner. They describe a fictional target (`acme.example`) and were not
produced by scanning anything.

They exist so you can try the Recon Planner without running recon first:

```bash
streamlit run bountybrain/ui/streamlit_app.py
# -> Recon Planner -> upload examples/httpx.jsonl
```

or from the command line:

```bash
python -c "
from bountybrain.recon.parsers import parse_recon_file
from bountybrain.recon.planner import plan_from_findings, render_markdown
f = parse_recon_file('examples/httpx.jsonl')
print(render_markdown(plan_from_findings(f)))
"
```

| file | tool | what it exercises |
|---|---|---|
| `httpx.jsonl` | httpx `-json -tech-detect` | technology fingerprints → product matching |
| `subfinder.txt` | subfinder | plain subdomain list |
| `nuclei.jsonl` | nuclei `-json` | CVE findings → direct weakness lookup |

For real use, point the planner at your own output — against targets that are
explicitly in scope for a program you are authorised to test.
