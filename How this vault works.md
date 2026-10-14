---
type: guide
tags: [meta]
---
# How this vault works

A knowledge base for the projects in `/data/jayavardhan/projects/`. To open it in Obsidian, choose File → Open vault and pick
`/data/jayavardhan/projects/vault`. Everything is plain Markdown. The thesis work is in a separate vault at
`/data/jayavardhan/mtp/vault`.

## Note types

| folder | one note per | answers |
|---|---|---|
| `Projects/` | project | what it is, what was built, the key numbers, state, how to run it, where the docs are |
| `Lessons/` | kind of mistake or practice | traps that cost time in more than one project |
| `Infrastructure/` | topic | environments, repos and git state |
| `Log/` | day | short dated entries |
| `Sources/<project>/` | — | **copies** of each project's docs (`.md`) and result figures (`.png`), with the original folder layout kept |

Start at [[Home]]. The chronology is in [[Timeline]]. Unfinished work is in [[Open threads]].

## Conventions

- **Numbers come from each project's own docs** (`README.md`, `RESULTS.md`, `PROGRESS.md`). A note quotes the
  1–3 numbers that matter and links to the source for the full tables. If a number changes, update the
  source first and then the note.
- **Negative results stay in.** Cut features and confounded results are recorded so they are not redone.
- Paths in a project note are relative to that project's folder, `/data/jayavardhan/projects/<folder>/`,
  unless they are absolute.
- To add a project, copy `Templates/Project.md` into `Projects/`, add a row to [[Home]] and an entry to [[Timeline]],
  and add a `Log/` entry.
- **`Sources/` is a snapshot** (taken 2026-10-14), not a live link. Each project note links to its copies under "Docs in this vault".
  Left out on purpose: PDFs, data/CSV/JSON, code, `init/`, and the `CLAUDE.md` / `.claude/` files (except sysml's assignment brief).
  If a project's docs change, refresh its copy (`rsync` only adds and updates, it never deletes):

  ```bash
  P=bountybrain   # the project folder name
  rsync -a --prune-empty-dirs --exclude='.git/' --exclude='.claude/' --exclude='.pytest_cache/' --exclude='.ruff_cache/' \
    --exclude='__pycache__/' --exclude='data/' --exclude='_to_delete/' --exclude='RAFT/' --exclude='CLAUDE*.md' \
    --exclude='*.pdf' --include='*/' --include='*.md' --include='*.png' --exclude='*' \
    /data/jayavardhan/projects/$P/ /data/jayavardhan/vaults/projects/Sources/$P/
  ```
- Each project's `CLAUDE.md` and `PROGRESS.md` are the detailed hand-over notes. This vault is the index on top of them.
