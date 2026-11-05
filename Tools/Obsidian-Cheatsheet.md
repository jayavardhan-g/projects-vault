# Obsidian Cheatsheet

Vim keys for this vault. `<Space>` is the leader, same as nvim, and most keys do what
they do in LazyVim. Search this note with `<Space>sb`, jump between sections with the
outline (`gO` or `<Space>oo`), and come back to wherever you were with `Ctrl+O`.

Keys live in `.obsidian/.obsidian.vimrc` (plus a few cleared defaults in
`.obsidian/hotkeys.json`). Restart Obsidian after editing them.

## Find & Search

### Find a note
- `<Space><Space>`, `<Space>ff`, `<Space>fr`, `Ctrl+P` - quick switcher (recent notes first)

### Search text
- `<Space>/`, `<Space>sg` - Omnisearch across the whole vault
- `<Space>sb` - Omnisearch inside this note
- `<Space>sr` - search & replace in this note
- `Ctrl+Shift+F` - Obsidian's search sidebar
- `/pattern` then `n` / `N` - search in this note (match stays centered)
- `*` / `#` - search the word under the cursor
- `<Esc>` - clear search highlight

### Jump anywhere on screen
- `s` then 2 characters, then the label shown - Lightspeed jump (nvim's flash)
- `f` / `t` / `F` / `T` `{char}`, then `;` / `,` - jump on this line

## Moving Around

### Lines and scrolling
- `j` / `k` - move by visual (wrapped) line
- `Ctrl+D` / `Ctrl+U` - half page down / up, cursor stays centered
- `zz` - center the cursor line

### Headings and outline
- `[[` / `]]` - previous / next heading
- `gO`, `<Space>oo`, `<Space>cs` - outline sidebar

### History
- `Ctrl+O` / `Ctrl+I` - back / forward through notes you've visited

## Links

### Follow links
- `gd`, `gf`, `gx` - follow the link under the cursor
- `go` - open it in a new tab
- `gD` - open it in a split
- `gr` - backlinks (which notes link here)
- `<Space>oB` - toggle backlinks at the bottom of the note

### Make links
- `[[` in insert mode - pick a note to link
- `gsaw` - wrap the word or selection in `[[ ]]`
- `Alt+P` - paste the clipboard URL as a link over the selection / word

## Tabs, Splits & Panes

### Tabs
- `H` / `L`, `[b` / `]b`, `gt` / `gT`, `Ctrl+Tab` / `Ctrl+Shift+Tab` - previous / next tab
- `Ctrl+T` - new tab
- `<Space>bd`, `<Space>wd`, `Ctrl+W`, `:q`, `:wq` - close tab
- `<Space>bo` - close all other tabs
- `<Space>bu` - reopen the last closed tab
- `<Space>bp` - pin / unpin tab

### Splits and panes
- `<Space>|`, `:vs` - split right
- `<Space>-`, `:sp` - split below
- `Ctrl+H` / `Ctrl+J` / `Ctrl+K` / `Ctrl+L` - focus pane left / down / up / right

### Sidebars
- `<Space>e`, `Ctrl+B` - left sidebar (file explorer)
- `<Space>E`, `Ctrl+Shift+B` - right sidebar (backlinks, outline, tags, git)

## Editing

### Delete, yank, paste
- `x` - delete a character without yanking it
- `y` / `p` - yank / paste use the system clipboard (like nvim locally)
- `<Space>y` / `<Space>Y` - yank / yank to end of line (same thing, nvim muscle memory)

### Move lines
- `Alt+J` / `Alt+K` - move the line down / up

### Surround
Wraps the selection, or the word under the cursor. Add only - there's no delete / replace.
- `gsa"` `gsa'` ``gsa` `` - quotes / backticks
- `gsa(` `gsa[` `gsa{` - brackets
- `gsab` - `**bold**`
- `gsai` - `*italic*`
- `gsam` - `$math$`
- `gsaw` - `[[wiki link]]`

### Markdown
- `<Space>x` - tick / untick a checkbox
- `<Space>cl` - cycle the line: plain → `-` bullet → `1.` numbered → `[ ]` checkbox
- `<Space>>` - toggle blockquote
- `<Space>um` - toggle reading view / live preview
- `:w` - save now (Obsidian autosaves anyway)

### Folding
- `za`, `zo`, `zc` - toggle the fold under the cursor
- `zm` / `zr` - fold one heading level more / less
- `zM` / `zR` - fold / unfold everything
- `<Space>op` - fold the properties (frontmatter) block

## Notes & Files

- `<Space>fn` - new note
- `<Space>fN` - new folder
- `<Space>cr` - rename this note
- `<Space>m`, `Ctrl+Shift+M` - move this note to another folder
- `<Space>fd` - delete this note (asks first, goes to trash)
- `<Space>fe` - reveal this note in the file explorer
- `<Space>ce` - extract the heading under the cursor into a new note (leaves a link)
- `<Space>oi` - insert a template
- `<Space>ba` - bookmark this note
- `<Space>om` - bookmarks list
- `<Space>oP` - properties panel

## Obsidian Panels

- `<Space>od` - today's daily note
- `[d` / `]d` - previous / next daily note
- `<Space>ob` - backlinks sidebar
- `<Space>ot` - tags
- `<Space>og` / `<Space>oG` - graph / local graph of this note

## Git

- `<Space>gg` - source control view
- `<Space>gl` - history view
- `<Space>gd` - diff view
- `<Space>gb` - toggle blame on lines
- `<Space>gc` - commit everything, pull and push
- `]h` / `[h` - next / previous change (hunk)
- `<Space>ghs` / `<Space>ghr` / `<Space>ghp` - stage / reset / preview the hunk

## App

- `Ctrl+Shift+P` - command palette (every Obsidian command)
- `<Space>ul` - toggle line numbers
- `<Space>h` - this cheatsheet
- Ex commands: `:w` save, `:q` / `:wq` close tab, `:vs` / `:sp` split, `:nohl`

## Differences from nvim

- Surround is add-only: no `gsd` / `gsr`.
- `s` jumps in normal mode only; in visual mode it's plain substitute. No `S`.
- Obsidian hotkeys cleared to make room for vim keys: `Ctrl+H` (search & replace → `<Space>sr`),
  `Ctrl+K` (insert link → `[[`), `Ctrl+L` (checkbox → `<Space>x`), `Ctrl+D` (delete
  paragraph), `Ctrl+O` / `Ctrl+I` (quick switcher, italic), `Ctrl+B` (bold → `gsab`).
- `n` / `N` / `Ctrl+D` / `Ctrl+U` keep the cursor centered; nvim doesn't.
