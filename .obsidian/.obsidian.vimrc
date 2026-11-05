" Mirrors ~/.config/nvim (LazyVim + custom keymaps.lua) wherever Obsidian has
" an equivalent action. Leader = Space, same as nvim.

" Have j and k navigate visual lines rather than logical ones (LazyVim does
" the same for j/k without a count)
nmap j gj
nmap k gk

" Esc clears search highlights, like LazyVim. Any ex command clears them in
" this vim engine, so :nohl is enough. , stays native (repeat f/t backward).
nmap <Esc> :nohl<CR>
nmap <F9> :nohl<CR>

" Plain y/p use the system clipboard - same as nvim locally (LazyVim's
" clipboard=unnamedplus). <Space>y / <Space>Y kept too for muscle memory.
set clipboard=unnamed

" x deletes a character without yanking it (nvim keymaps.lua: x -> "_x)
nnoremap x "_x

" Keep the cursor line centered after half-page scrolls and search jumps.
" Must be nnoremap: with nmap, the engine's recursion guard swallows the
" inner n/N/<C-d>/<C-u>, leaving only zz. Ctrl+D needs Obsidian's default
" Ctrl+D (delete paragraph) hotkey cleared in hotkeys.json.
nnoremap <C-d> <C-d>zz
nnoremap <C-u> <C-u>zz
nnoremap n nzz
nnoremap N Nzz

" Vim ex aliases - :w saves, :q / :wq close the tab, :vs / :sp split right /
" below. The engine's built-in :w calls a CodeMirror save hook Obsidian never
" sets, so it's a silent no-op without this.
exmap w obcommand editor:save-file
exmap q obcommand workspace:close
exmap wq obcommand workspace:close
exmap vs obcommand workspace:split-vertical
exmap sp obcommand workspace:split-horizontal

" Go back and forward with Ctrl+O and Ctrl+I
" (requires clearing Obsidian's default Quick switcher / Toggle italic
" hotkeys from Ctrl+O / Ctrl+I in Settings -> Hotkeys first)
exmap back obcommand app:go-back
nmap <C-o> :back<CR>
exmap forward obcommand app:go-forward
nmap <C-i> :forward<CR>

" --- FLASH (s) ---
" nvim has flash.nvim on s. Closest Obsidian match is the Jump to link
" plugin's Lightspeed jump: type 2 chars, then the label shown on the match.
" Normal mode only - visual s stays native (substitute selection).
exmap flash obcommand mrj-jump-to-link:activate-lightspeed-jump
nunmap s
nmap s :flash<CR>

" --- SURROUND (gsa, like nvim's mini.surround) ---
" Wraps the selection, or the word under the cursor. Add-only: the plugin's
" surround can't delete/replace, so there's no gsd/gsr.
exmap surround_math surround $ $
exmap surround_wiki surround [[ ]]
exmap surround_double_quotes surround " "
exmap surround_single_quotes surround ' '
exmap surround_backticks surround ` `
exmap surround_brackets surround ( )
exmap surround_square_brackets surround [ ]
exmap surround_curly_brackets surround { }
exmap surround_italic surround * *
exmap surround_bold surround ** **

" NOTE: must use 'map' and not 'nmap'
" [[ and ]] deliberately left as the plugin's own defaults (jump to
" previous/next heading) -- wiki-link-wrap lives on gsaw instead.
map gsa" :surround_double_quotes<CR>
map gsa' :surround_single_quotes<CR>
map gsa` :surround_backticks<CR>
map gsa( :surround_brackets<CR>
map gsa[ :surround_square_brackets<CR>
map gsa{ :surround_curly_brackets<CR>
map gsam :surround_math<CR>
map gsai :surround_italic<CR>
map gsab :surround_bold<CR>
map gsaw :surround_wiki<CR>

" --- APP ACTIONS (matches browser / VSCode muscle memory) ---
exmap newtab obcommand workspace:new-tab
nmap <C-t> :newtab<CR>

exmap nexttab obcommand workspace:next-tab
nmap <C-Tab> :nexttab<CR>
exmap prevtab obcommand workspace:previous-tab
nmap <C-S-Tab> :prevtab<CR>

exmap close obcommand workspace:close
nunmap <C-w>
nmap <C-w> :close<CR>
imap <C-w> <Esc>:close<CR>

exmap cmdpalette obcommand command-palette:open
nmap <C-S-p> :cmdpalette<CR>

exmap quickswitch obcommand switcher:open
nmap <C-p> :quickswitch<CR>

exmap toggleleftsidebar obcommand app:toggle-left-sidebar
nmap <C-b> :toggleleftsidebar<CR>
exmap togglerightsidebar obcommand app:toggle-right-sidebar
nmap <C-S-b> :togglerightsidebar<CR>

exmap search obcommand global-search:open
nmap <C-S-f> :search<CR>

exmap movefile obcommand file-explorer:move-file
nmap <C-S-m> :movefile<CR>
" <Space>m restored from the original pre-2026-09-20 leader block (mapped
" below, after unmap <Space>)

" --- LEADER (Space) - same keys as LazyVim ---
" Bare <Space> already has a built-in full match (Space -> l, move right,
" real vim behavior) -- the dispatcher fires a full match immediately and
" never checks for a longer partial match once one exists, so bare Space
" must be unmapped first (same reason 's' gets unmapped above) or these
" longer sequences can never complete.
unmap <Space>

" Splits: Space then - (below) or | (right)
exmap vsplit obcommand workspace:split-vertical
nmap <Space>| :vsplit<CR>
exmap hsplit obcommand workspace:split-horizontal
nmap <Space>- :hsplit<CR>

" Move note to another folder
nmap <Space>m :movefile<CR>

" Find files / recent -> quick switcher (it lists recent notes first)
nmap <Space><Space> :quickswitch<CR>
nmap <Space>ff :quickswitch<CR>
nmap <Space>fr :quickswitch<CR>
exmap newfile obcommand file-explorer:new-file
nmap <Space>fn :newfile<CR>
exmap newfolder obcommand file-explorer:new-folder
nmap <Space>fN :newfolder<CR>
" Reveal this note in the explorer (nvim: <leader>fe explorer)
exmap revealfile obcommand file-explorer:reveal-active-file
nmap <Space>fe :revealfile<CR>
" Delete this note (Obsidian asks first; goes to trash)
exmap deletefile obcommand app:delete-file
nmap <Space>fd :deletefile<CR>

" Grep -> Omnisearch picker (vault / this note)
exmap grep obcommand omnisearch:show-modal
nmap <Space>/ :grep<CR>
nmap <Space>sg :grep<CR>
exmap grepbuffer obcommand omnisearch:show-modal-infile
nmap <Space>sb :grepbuffer<CR>
" Search & replace in this note (LazyVim <leader>sr; replaces freed Ctrl+H)
exmap searchreplace obcommand editor:open-search-replace
nmap <Space>sr :searchreplace<CR>

" Explorer (left) / other sidebar (right: backlinks, outline, tags, git) -
" e/E pair like nvim's two explorer keys
nmap <Space>e :toggleleftsidebar<CR>
nmap <Space>E :togglerightsidebar<CR>

" Buffers / windows -> tabs / panes
nmap <Space>bd :close<CR>
nmap <Space>wd :close<CR>
exmap closeothers obcommand workspace:close-others
nmap <Space>bo :closeothers<CR>
" Pin tab (LazyVim <leader>bp), reopen closed tab
exmap togglepin obcommand workspace:toggle-pin
nmap <Space>bp :togglepin<CR>
exmap reopentab obcommand workspace:undo-close-pane
nmap <Space>bu :reopentab<CR>
" Bookmarks (harpoon-style): add this note / list on <Space>om
exmap bookmarkadd obcommand bookmarks:bookmark-current-view
nmap <Space>ba :bookmarkadd<CR>

" Yank to system clipboard (nvim: "+y)
nmap <Space>y "+y
vmap <Space>y "+y
nmap <Space>Y "+y$

" Rename note (nvim: <leader>cr = LSP rename)
exmap rename obcommand workspace:edit-file-title
nmap <Space>cr :rename<CR>
" Extract the heading under the cursor (+ its content) into a new note,
" leaving a [[link]] behind
exmap extractheading obcommand note-composer:extract-heading
nmap <Space>ce :extractheading<CR>

" Cheatsheet note (nvim: <leader>h). Command comes from the "Hotkeys for
" specific files" plugin - its id is the note's path, and obcommand splits on
" spaces, so the path must not contain any.
exmap cheatsheet obcommand obsidian-hotkeys-for-specific-files:Tools/Obsidian-Cheatsheet.md
nmap <Space>h :cheatsheet<CR>

" Toggles
exmap togglelinenumbers obcommand editor:toggle-line-numbers
nmap <Space>ul :togglelinenumbers<CR>
" Reading view / live preview (LazyVim markdown extra: <leader>um)
exmap togglepreview obcommand markdown:toggle-preview
nmap <Space>um :togglepreview<CR>

" Markdown editing: tick checkbox, toggle blockquote (line or selection)
exmap toggletask obcommand editor:toggle-checklist-status
map <Space>x :toggletask<CR>
exmap toggleblockquote obcommand editor:toggle-blockquote
map <Space>> :toggleblockquote<CR>
" Cycle line/selection: plain -> bullet -> numbered -> checkbox -> plain
exmap cyclelist obcommand editor:cycle-list-checklist
map <Space>cl :cyclelist<CR>

" --- OBSIDIAN (<Space>o - group is free in nvim) ---
exmap dailynote obcommand daily-notes
nmap <Space>od :dailynote<CR>
exmap prevdaily obcommand daily-notes:goto-prev
nmap [d :prevdaily<CR>
exmap nextdaily obcommand daily-notes:goto-next
nmap ]d :nextdaily<CR>

exmap backlinks obcommand backlink:open
nmap <Space>ob :backlinks<CR>
exmap docbacklinks obcommand backlink:toggle-backlinks-in-document
nmap <Space>oB :docbacklinks<CR>
" Outline also on gO (nvim: document symbols) and <Space>cs (LazyVim symbols)
exmap outline obcommand outline:open
nmap <Space>oo :outline<CR>
nmap <Space>cs :outline<CR>
exmap tags obcommand tag-pane:open
nmap <Space>ot :tags<CR>
exmap graph obcommand graph:open
nmap <Space>og :graph<CR>
exmap localgraph obcommand graph:open-local
nmap <Space>oG :localgraph<CR>

exmap template obcommand templates:insert-template
nmap <Space>oi :template<CR>
exmap bookmarklist obcommand bookmarks:open
nmap <Space>om :bookmarklist<CR>
" Properties: op folds the frontmatter block, oP opens the properties panel
exmap foldprops obcommand editor:toggle-fold-properties
nmap <Space>op :foldprops<CR>
exmap propspanel obcommand properties:open-local
nmap <Space>oP :propspanel<CR>

" Git (obsidian-git): lazygit view, blame, hunks
exmap gitview obcommand obsidian-git:open-git-view
nmap <Space>gg :gitview<CR>
exmap gitblame obcommand obsidian-git:toggle-line-author-info
nmap <Space>gb :gitblame<CR>
exmap nexthunk obcommand obsidian-git:next-hunk
nmap ]h :nexthunk<CR>
exmap prevhunk obcommand obsidian-git:prev-hunk
nmap [h :prevhunk<CR>
" Hunk actions on LazyVim's gitsigns keys (<leader>gh...)
exmap stagehunk obcommand obsidian-git:stage-hunk
nmap <Space>ghs :stagehunk<CR>
exmap resethunk obcommand obsidian-git:reset-hunk
nmap <Space>ghr :resethunk<CR>
exmap previewhunk obcommand obsidian-git:preview-hunk
nmap <Space>ghp :previewhunk<CR>
exmap githistory obcommand obsidian-git:open-history-view
nmap <Space>gl :githistory<CR>
exmap gitdiff obcommand obsidian-git:open-diff-view
nmap <Space>gd :gitdiff<CR>
" Commit everything, pull and push (obsidian-git "Commit-and-sync")
exmap gitsync obcommand obsidian-git:push
nmap <Space>gc :gitsync<CR>

" --- LINKS ---
" gd / gx follow the link under the cursor (nvim: go to definition / open URI)
exmap followlink obcommand editor:follow-link
nmap gd :followlink<CR>
nmap gx :followlink<CR>
nmap gf :followlink<CR>
" go: link in a new tab, gD: link in a split
exmap linknewtab obcommand editor:open-link-in-new-leaf
nmap go :linknewtab<CR>
exmap linksplit obcommand editor:open-link-in-new-split
nmap gD :linksplit<CR>
" gr: backlinks (LazyVim gr = references), gO: outline (nvim document symbols)
nmap gr :backlinks<CR>
nmap gO :outline<CR>

" --- WINDOW FOCUS (Ctrl+hjkl, like LazyVim) ---
" Requires Obsidian's default Ctrl+H (search & replace), Ctrl+K (insert
" link) and Ctrl+L (toggle checklist) hotkeys to be cleared - Obsidian
" hotkeys fire before vimrc maps.
exmap focusLeft obcommand editor:focus-left
exmap focusRight obcommand editor:focus-right
exmap focusTop obcommand editor:focus-top
exmap focusBottom obcommand editor:focus-bottom
nmap <C-h> :focusLeft<CR>
nmap <C-l> :focusRight<CR>
nmap <C-k> :focusTop<CR>
nmap <C-j> :focusBottom<CR>

" --- MOVE LINES (Alt+j/k, like LazyVim) ---
exmap linedown obcommand editor:swap-line-down
exmap lineup obcommand editor:swap-line-up
nmap <A-j> :linedown<CR>
nmap <A-k> :lineup<CR>

" --- FOLDING ---
exmap togglefold obcommand editor:toggle-fold
nmap zo :togglefold<CR>
nmap zc :togglefold<CR>
nmap za :togglefold<CR>

" zm / zr: fold one heading level more / less
exmap foldmore obcommand editor:fold-more
nmap zm :foldmore<CR>
exmap foldless obcommand editor:fold-less
nmap zr :foldless<CR>

exmap unfoldall obcommand editor:unfold-all
nmap zR :unfoldall<CR>

exmap foldall obcommand editor:fold-all
nmap zM :foldall<CR>

" --- TAB NAVIGATION (H/L and [b/]b like LazyVim's buffers, plus gt/gT) ---
exmap tabnext obcommand workspace:next-tab
exmap tabprev obcommand workspace:previous-tab
nmap H :tabprev<CR>
nmap L :tabnext<CR>
nmap [b :tabprev<CR>
nmap ]b :tabnext<CR>
nmap gt :tabnext<CR>
nmap gT :tabprev<CR>

" --- HYPERLINKS ---
" Paste clipboard URL over selected text / word under cursor
map <A-p> :pasteinto<CR>
