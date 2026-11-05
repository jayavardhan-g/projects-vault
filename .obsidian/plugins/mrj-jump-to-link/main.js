'use strict';

var obsidian = require('obsidian');
var state = require('@codemirror/state');
var view = require('@codemirror/view');

class Settings {
    constructor() {
        // Defaults as in Vimium extension for browsers
        this.letters = 'sadfjklewcmpgh';
        this.jumpToAnywhereRegex = '\\b\\w{3,}\\b';
        this.lightspeedCaseSensitive = false;
        this.jumpToLinkIfOneLinkOnly = true;
        this.lightspeedJumpToStartOfWord = true;
        this.lightspeedCharacterCount = 2;
    }
}

class MarkWidget extends view.WidgetType {
    constructor(mark, type, matchedEventKey) {
        super();
        this.mark = mark;
        this.type = type;
        this.matchedEventKey = matchedEventKey;
    }
    eq(other) {
        return other.mark === this.mark && other.matchedEventKey == this.matchedEventKey;
    }
    toDOM() {
        // The window level `createDiv` returns a DETACHED element, while the Node
        // level helpers append to the node they are called on - so
        // `activeDocument.createDiv()` would throw. A widget has to be detached.
        const cls = ["jl", "jl-" + this.type, "jl-inline", "popover"];
        if (this.matchedEventKey && this.mark.toUpperCase().startsWith(this.matchedEventKey.toUpperCase())) {
            cls.push('matched');
        }
        const wrapper = createDiv({ cls });
        wrapper.createSpan({ text: this.mark });
        return wrapper;
    }
    ignoreEvent() {
        return false;
    }
}

class MarkPlugin {
    constructor(_view) {
        this.links = [];
        this.matchedEventKey = undefined;
        this.links = [];
        this.matchedEventKey = undefined;
        this.decorations = view.Decoration.none;
    }
    setLinks(links) {
        this.links = links;
        this.matchedEventKey = undefined;
    }
    clean() {
        this.links = [];
        this.matchedEventKey = undefined;
    }
    filterWithEventKey(eventKey) {
        if (eventKey.length != 1)
            return;
        this.links = this.links.filter(v => {
            return v.letter.length == 2 && v.letter[0].toUpperCase() == eventKey.toUpperCase();
        });
        this.matchedEventKey = eventKey;
    }
    get visible() {
        return this.links.length > 0;
    }
    update(_update) {
        const widgets = this.links.map((x) => view.Decoration.widget({
            widget: new MarkWidget(x.letter, x.type, this.matchedEventKey),
            side: 1,
        }).range(x.index));
        this.decorations = view.Decoration.set(widgets);
    }
}

/**
 * Get only visible content
 * @param cmEditor
 * @returns Letter offset and visible content as a string
 */
function getVisibleLineText(cmEditor) {
    const scrollInfo = cmEditor.getScrollInfo();
    const { line: from } = cmEditor.coordsChar({ left: 0, top: 0 }, 'page');
    const { line: to } = cmEditor.coordsChar({ left: scrollInfo.left, top: scrollInfo.top + scrollInfo.height });
    const indOffset = cmEditor.indexFromPos({ ch: 0, line: from });
    const strs = cmEditor.getRange({ ch: 0, line: from }, { ch: 0, line: to + 1 });
    return { indOffset, strs };
}
/**
 *
 * @param alphabet - Letters which used to produce hints
 * @param numLinkHints - Count of needed links
 */
function getLinkHintLetters(alphabet, numLinkHints) {
    const alphabetUppercase = alphabet.toUpperCase();
    let prefixCount = Math.ceil((numLinkHints - alphabetUppercase.length) / (alphabetUppercase.length - 1));
    // ensure 0 <= prefixCount <= alphabet.length
    prefixCount = Math.max(prefixCount, 0);
    prefixCount = Math.min(prefixCount, alphabetUppercase.length);
    const prefixes = ['', ...Array.from(alphabetUppercase.slice(0, prefixCount))];
    const linkHintLetters = [];
    for (let i = 0; i < prefixes.length; i++) {
        const prefix = prefixes[i];
        for (let j = 0; j < alphabetUppercase.length; j++) {
            if (linkHintLetters.length < numLinkHints) {
                const letter = alphabetUppercase[j];
                if (prefix === '') {
                    if (!prefixes.contains(letter)) {
                        linkHintLetters.push(letter);
                    }
                }
                else {
                    linkHintLetters.push(prefix + letter);
                }
            }
            else {
                break;
            }
        }
    }
    return linkHintLetters;
}
function getMDHintLinks(content, offset, letters) {
    var _a;
    // expecting either [[Link]] or [[Link|Title]]
    const regExInternal = /\[\[(.+?)(\|.+?)?]]/g;
    // expecting [Title](../example.md)
    const regExMdInternal = /\[[^[\]]+?\]\(((\.\.|\w|\d).+?)\)/g;
    // expecting [Title](file://link), [Title](https://link) or any other [Jira-123](jira://bla-bla) link
    const regExExternal = /\[[^[\]]+?\]\((.+?:\/\/.+?)\)/g;
    // expecting http://hogehoge or https://hogehoge
    const regExUrl = /( |\n|^)(https?:\/\/[^ \n]+)/g;
    const indexes = new Set();
    const linksWithIndex = [];
    let regExResult;
    const addLinkToArray = (link) => {
        if (indexes.has(link.index))
            return;
        indexes.add(link.index);
        linksWithIndex.push(link);
    };
    while ((regExResult = regExInternal.exec(content)) !== null) {
        const linkText = (_a = regExResult[1]) === null || _a === void 0 ? void 0 : _a.trim();
        addLinkToArray({ index: regExResult.index + offset, type: 'internal', linkText });
    }
    // External Link above internal, to prefer type external over interal in case of a dupe
    while ((regExResult = regExExternal.exec(content)) !== null) {
        const linkText = regExResult[1];
        addLinkToArray({ index: regExResult.index + offset, type: 'external', linkText });
    }
    while ((regExResult = regExMdInternal.exec(content)) !== null) {
        const linkText = regExResult[1];
        addLinkToArray({ index: regExResult.index + offset, type: 'internal', linkText });
    }
    while ((regExResult = regExUrl.exec(content)) !== null) {
        const linkText = regExResult[2];
        addLinkToArray({ index: regExResult.index + offset + 1, type: 'external', linkText });
    }
    const linkHintLetters = getLinkHintLetters(letters, linksWithIndex.length);
    const linksWithLetter = [];
    linksWithIndex
        .sort((x, y) => x.index - y.index)
        .forEach((linkHint, i) => {
        linksWithLetter.push({ letter: linkHintLetters[i], ...linkHint });
    });
    return linksWithLetter.filter(link => link.letter);
}
function createWidgetElement(content, type) {
    return createDiv({ cls: ['jl', 'jl-' + type, 'popover'], text: content });
}
function displaySourcePopovers(cmEditor, linkKeyMap) {
    const drawWidget = (cmEditor, linkHint) => {
        const pos = cmEditor.posFromIndex(linkHint.index);
        cmEditor.addWidget(pos, createWidgetElement(linkHint.letter, linkHint.type), false, 'over');
    };
    linkKeyMap.forEach(x => drawWidget(cmEditor, x));
}

/**
 * Returns the range of the document which is visible on screen, trimming the
 * lines CodeMirror renders above the viewport when that information is available.
 */
function getVisibleRange(cmEditor) {
    var _a, _b, _c;
    let { from } = cmEditor.viewport;
    const { to } = cmEditor.viewport;
    const viewState = cmEditor.viewState;
    const pixelOffsetTop = (_a = viewState === null || viewState === void 0 ? void 0 : viewState.pixelViewport) === null || _a === void 0 ? void 0 : _a.top;
    if (pixelOffsetTop) {
        const lines = (_b = viewState.viewportLines) !== null && _b !== void 0 ? _b : [];
        from = (_c = lines.filter(line => line.top > pixelOffsetTop)[0]) === null || _c === void 0 ? void 0 : _c.from;
    }
    return { index: from, content: cmEditor.state.sliceDoc(from, to) };
}

class CM6LinkProcessor {
    constructor(editor, alphabet) {
        this.getSourceLinkHints = () => {
            const { letters } = this;
            const { index, content } = this.getVisibleLines();
            return getMDHintLinks(content, index, letters);
        };
        this.cmEditor = editor;
        this.letters = alphabet;
    }
    init() {
        return this.getSourceLinkHints();
    }
    getVisibleLines() {
        return getVisibleRange(this.cmEditor);
    }
}

function extractRegexpBlocks(content, offset, regexp, letters, caseSensitive) {
    const regExUrl = caseSensitive ? new RegExp(regexp, 'g') : new RegExp(regexp, 'ig');
    let linksWithIndex = [];
    let regExResult;
    while ((regExResult = regExUrl.exec(content))) {
        const linkText = regExResult[1];
        linksWithIndex.push({
            index: regExResult.index + offset,
            type: "regex",
            linkText,
        });
    }
    const linkHintLetters = getLinkHintLetters(letters, linksWithIndex.length);
    const linksWithLetter = [];
    linksWithIndex
        .sort((x, y) => x.index - y.index)
        .forEach((linkHint, i) => {
        linksWithLetter.push({
            letter: linkHintLetters[i],
            ...linkHint,
        });
    });
    return linksWithLetter.filter(link => link.letter);
}

class CM6RegexProcessor extends CM6LinkProcessor {
    constructor(editor, alphabet, regexp, caseSensitive) {
        super(editor, alphabet);
        this.regexp = regexp;
        this.caseSensitive = caseSensitive;
    }
    init() {
        const { letters, regexp } = this;
        const { index, content } = this.getVisibleLines();
        return extractRegexpBlocks(content, index, regexp, letters, this.caseSensitive);
    }
}

class LegacyRegexpProcessor {
    constructor(cmEditor, regexp, alphabet, caseSensitive) {
        this.cmEditor = cmEditor;
        this.regexp = regexp;
        this.letters = alphabet;
        this.caseSensitive = caseSensitive;
    }
    init() {
        const [content, offset] = this.getVisibleContent();
        const links = this.getLinks(content, offset);
        this.display(links);
        return links;
    }
    getVisibleContent() {
        const { cmEditor } = this;
        const { indOffset, strs } = getVisibleLineText(cmEditor);
        return [strs, indOffset];
    }
    getLinks(content, offset) {
        const { regexp, letters } = this;
        return extractRegexpBlocks(content, offset, regexp, letters, this.caseSensitive);
    }
    display(links) {
        const { cmEditor } = this;
        displaySourcePopovers(cmEditor, links);
    }
}

class LegacySourceLinkProcessor {
    constructor(editor, alphabet) {
        this.getSourceLinkHints = (cmEditor) => {
            const { letters } = this;
            const { indOffset, strs } = getVisibleLineText(cmEditor);
            return getMDHintLinks(strs, indOffset, letters);
        };
        this.cmEditor = editor;
        this.letters = alphabet;
    }
    init() {
        const { cmEditor } = this;
        const linkHints = this.getSourceLinkHints(cmEditor);
        displaySourcePopovers(cmEditor, linkHints);
        return linkHints;
    }
}

function getPreviewLinkHints(previewViewEl, letters) {
    const anchorEls = previewViewEl.querySelectorAll('a, .metadata-link-inner');
    const embedEls = previewViewEl.querySelectorAll('.internal-embed');
    const linkHints = [];
    anchorEls.forEach((anchorEl, _i) => {
        var _a;
        if (checkIsPreviewElOnScreen(previewViewEl, anchorEl)) {
            return;
        }
        const linkType = anchorEl.classList.contains('internal-link')
            ? 'internal'
            : 'external';
        const linkText = linkType === 'internal'
            ? (_a = anchorEl.dataset['href']) !== null && _a !== void 0 ? _a : anchorEl.href
            : anchorEl.href;
        let offsetParent = anchorEl.offsetParent;
        let top = anchorEl.offsetTop;
        let left = anchorEl.offsetLeft;
        while (offsetParent) {
            if (offsetParent == previewViewEl) {
                offsetParent = undefined;
            }
            else {
                top += offsetParent.offsetTop;
                left += offsetParent.offsetLeft;
                offsetParent = offsetParent.offsetParent;
            }
        }
        linkHints.push({
            linkElement: anchorEl,
            letter: '',
            linkText: linkText,
            type: linkType,
            top: top,
            left: left,
        });
    });
    embedEls.forEach((embedEl, _i) => {
        const linkText = embedEl.getAttribute('src');
        const linkEl = embedEl.querySelector('.markdown-embed-link');
        if (linkText && linkEl) {
            if (checkIsPreviewElOnScreen(previewViewEl, linkEl)) {
                return;
            }
            let offsetParent = linkEl.offsetParent;
            let top = linkEl.offsetTop;
            let left = linkEl.offsetLeft;
            while (offsetParent) {
                if (offsetParent == previewViewEl) {
                    offsetParent = undefined;
                }
                else {
                    top += offsetParent.offsetTop;
                    left += offsetParent.offsetLeft;
                    offsetParent = offsetParent.offsetParent;
                }
            }
            linkHints.push({
                linkElement: linkEl,
                letter: '',
                linkText: linkText,
                type: 'internal',
                top: top,
                left: left,
            });
        }
    });
    const sortedLinkHints = linkHints.sort((a, b) => {
        if (a.top > b.top) {
            return 1;
        }
        else if (a.top === b.top) {
            if (a.left > b.left) {
                return 1;
            }
            else if (a.left === b.left) {
                return 0;
            }
            else {
                return -1;
            }
        }
        else {
            return -1;
        }
    });
    const linkHintLetters = getLinkHintLetters(letters, sortedLinkHints.length);
    sortedLinkHints.forEach((linkHint, i) => {
        linkHint.letter = linkHintLetters[i];
    });
    return sortedLinkHints;
}
function checkIsPreviewElOnScreen(parent, el) {
    el = el.closest('[data-view-type="table"], table') || el;
    return el.offsetTop < parent.scrollTop || el.offsetTop > parent.scrollTop + parent.offsetHeight;
}
function displayPreviewPopovers(linkHints) {
    const linkHintHtmlElements = [];
    for (const linkHint of linkHints) {
        linkHint.linkElement.classList.add('jl-anchor');
        const popoverElement = linkHint.linkElement.createSpan({
            text: linkHint.letter,
            cls: ['jl', 'jl-' + linkHint.type, 'jl-preview', 'popover'],
        });
        linkHintHtmlElements.push(popoverElement);
    }
    return linkHintHtmlElements;
}

class PreviewLinkProcessor {
    constructor(view, alphabet) {
        this.view = view;
        this.alphabet = alphabet;
    }
    init() {
        const { view, alphabet } = this;
        const links = getPreviewLinkHints(view, alphabet);
        displayPreviewPopovers(links);
        return links;
    }
}

class LivePreviewLinkProcessor {
    constructor(view, editor, alphabet) {
        this.getSourceLinkHints = () => {
            const { alphabet } = this;
            const { index, content } = this.getVisibleLines();
            return getMDHintLinks(content, index, alphabet);
        };
        this.view = view;
        this.cmEditor = editor;
        this.alphabet = alphabet;
    }
    init() {
        const { view, alphabet } = this;
        const links = getPreviewLinkHints(view, alphabet);
        const sourceLinks = this.getSourceLinkHints();
        const linkHintLetters = getLinkHintLetters(alphabet, links.length + sourceLinks.length);
        const linksRemapped = links.map((link, idx) => ({ ...link, letter: linkHintLetters[idx] })).filter(link => link.letter);
        const sourceLinksRemapped = sourceLinks.map((link, idx) => ({ ...link, letter: linkHintLetters[idx + links.length] })).filter(link => link.letter);
        const linkHintHtmlElements = displayPreviewPopovers(linksRemapped);
        return [linksRemapped, sourceLinksRemapped, linkHintHtmlElements];
    }
    getVisibleLines() {
        return getVisibleRange(this.cmEditor);
    }
}

var VIEW_MODE;
(function (VIEW_MODE) {
    VIEW_MODE[VIEW_MODE["SOURCE"] = 0] = "SOURCE";
    VIEW_MODE[VIEW_MODE["PREVIEW"] = 1] = "PREVIEW";
    VIEW_MODE[VIEW_MODE["LEGACY"] = 2] = "LEGACY";
    VIEW_MODE[VIEW_MODE["LIVE_PREVIEW"] = 3] = "LIVE_PREVIEW";
})(VIEW_MODE || (VIEW_MODE = {}));
class JumpToLink extends obsidian.Plugin {
    constructor() {
        super(...arguments);
        this.isLinkHintActive = false;
        this.prefixInfo = undefined;
        this.currentCursor = {};
        this.cursorBeforeJump = {};
        this.handleJumpToLink = () => {
            const { settings: { letters } } = this;
            const { mode, currentView } = this;
            switch (mode) {
                case VIEW_MODE.LEGACY: {
                    const cmEditor = this.cmEditor;
                    const sourceLinkHints = new LegacySourceLinkProcessor(cmEditor, letters).init();
                    this.handleActions(sourceLinkHints);
                    break;
                }
                case VIEW_MODE.LIVE_PREVIEW: {
                    const cm6Editor = this.cmEditor;
                    const previewViewEl = currentView.currentMode.editor.containerEl;
                    const [previewLinkHints, sourceLinkHints, linkHintHtmlElements] = new LivePreviewLinkProcessor(previewViewEl, cm6Editor, letters).init();
                    cm6Editor.plugin(this.markViewPlugin).setLinks(sourceLinkHints);
                    this.app.workspace.updateOptions();
                    this.handleActions([...previewLinkHints, ...sourceLinkHints], linkHintHtmlElements);
                    break;
                }
                case VIEW_MODE.PREVIEW: {
                    const previewViewEl = currentView.previewMode.containerEl.querySelector('div.markdown-preview-view');
                    const previewLinkHints = new PreviewLinkProcessor(previewViewEl, letters).init();
                    this.handleActions(previewLinkHints);
                    break;
                }
                case VIEW_MODE.SOURCE: {
                    const cm6Editor = this.cmEditor;
                    const livePreviewLinks = new CM6LinkProcessor(cm6Editor, letters).init();
                    cm6Editor.plugin(this.markViewPlugin).setLinks(livePreviewLinks);
                    this.app.workspace.updateOptions();
                    this.handleActions(livePreviewLinks);
                    break;
                }
            }
        };
        /*
        *  caseSensitive is only for lightspeed and shall not affect jumpToAnywhere, so it is true
        *  by default
        */
        this.handleJumpToRegex = (stringToSearch, caseSensitive = true) => {
            const { settings: { letters, jumpToAnywhereRegex } } = this;
            const whatToLookAt = stringToSearch || jumpToAnywhereRegex;
            const { mode } = this;
            switch (mode) {
                case VIEW_MODE.SOURCE:
                    this.handleMarkdownRegex(letters, whatToLookAt, caseSensitive);
                    break;
                case VIEW_MODE.LIVE_PREVIEW:
                    this.handleMarkdownRegex(letters, whatToLookAt, caseSensitive);
                    break;
                case VIEW_MODE.PREVIEW:
                    break;
                case VIEW_MODE.LEGACY: {
                    const cmEditor = this.cmEditor;
                    const links = new LegacyRegexpProcessor(cmEditor, whatToLookAt, letters, caseSensitive).init();
                    this.handleActions(links);
                    break;
                }
            }
        };
        this.handleMarkdownRegex = (letters, whatToLookAt, caseSensitive) => {
            const cm6Editor = this.cmEditor;
            const livePreviewLinks = new CM6RegexProcessor(cm6Editor, letters, whatToLookAt, caseSensitive).init();
            cm6Editor.plugin(this.markViewPlugin).setLinks(livePreviewLinks);
            this.app.workspace.updateOptions();
            this.handleActions(livePreviewLinks);
        };
    }
    async onload() {
        this.settings = await this.loadData() || new Settings();
        this.addSettingTab(new SettingTab(this.app, this));
        const markViewPlugin = this.markViewPlugin = view.ViewPlugin.fromClass(MarkPlugin, {
            decorations: (v) => v.decorations
        });
        this.registerEditorExtension([markViewPlugin]);
        this.watchForSelectionChange();
        this.addCommand({
            id: 'activate-jump-to-link',
            name: 'Show link hints',
            callback: () => this.action('link'),
        });
        this.addCommand({
            id: "activate-jump-to-anywhere",
            name: "Jump to anywhere regex",
            callback: () => this.action('regexp'),
        });
        this.addCommand({
            id: "activate-lightspeed-jump",
            name: "Lightspeed jump",
            callback: () => this.action('lightspeed'),
        });
    }
    action(type) {
        if (this.isLinkHintActive) {
            return;
        }
        const activeViewOfType = this.app.workspace.getActiveViewOfType(obsidian.MarkdownView);
        if (!activeViewOfType) {
            return;
        }
        const currentView = this.currentView = activeViewOfType.leaf.view;
        const mode = this.mode = this.getMode(this.currentView);
        this.contentElement = activeViewOfType.contentEl;
        this.cursorBeforeJump = this.currentCursor;
        switch (mode) {
            case VIEW_MODE.LEGACY:
                this.cmEditor = currentView.sourceMode.cmEditor;
                break;
            case VIEW_MODE.LIVE_PREVIEW:
            case VIEW_MODE.SOURCE:
                this.cmEditor = currentView.editor.cm;
                break;
        }
        switch (type) {
            case "link":
                this.handleJumpToLink();
                return;
            case "regexp":
                this.handleJumpToRegex();
                return;
            case "lightspeed":
                this.handleLightspeedJump();
                return;
        }
    }
    getMode(currentView) {
        var _a;
        const isLegacy = this.app.vault.getConfig("legacyEditor");
        if (currentView.getState().mode === 'preview') {
            return VIEW_MODE.PREVIEW;
        }
        else if (isLegacy) {
            return VIEW_MODE.LEGACY;
        }
        else if (currentView.getState().mode === 'source') {
            try {
                const isLivePreview = (_a = currentView.editor.cm.state) === null || _a === void 0 ? void 0 : _a.field(obsidian.editorLivePreviewField);
                if (isLivePreview)
                    return VIEW_MODE.LIVE_PREVIEW;
            }
            catch (e) {
                console.error(e);
            }
            return VIEW_MODE.SOURCE;
        }
    }
    // adapted from: https://github.com/mrjackphil/obsidian-jump-to-link/issues/35#issuecomment-1085905668
    handleLightspeedJump() {
        const activeView = this.app.workspace.getActiveViewOfType(obsidian.MarkdownView);
        if (!activeView) {
            return;
        }
        const { contentEl } = activeView;
        if (!contentEl) {
            return;
        }
        // this element doesn't exist in cm5/has a different class, so lightspeed will not work in cm5
        const contentContainer = contentEl.getElementsByClassName("cm-contentContainer")[0];
        if (!contentContainer) {
            return;
        }
        // dim all the text while waiting for the characters to jump to
        contentContainer.classList.add('jl-lightspeed');
        const restoreTextColor = () => contentContainer.classList.remove('jl-lightspeed');
        const keyArray = [];
        const grabKey = (event) => {
            event.preventDefault();
            // handle Escape to reject the mode
            if (event.key === 'Escape') {
                contentEl.removeEventListener("keydown", grabKey, { capture: true });
                restoreTextColor();
            }
            // test if keypress is capitalized
            if (/^[\w\S\W]$/i.test(event.key)) {
                const isCapital = event.shiftKey;
                if (isCapital) {
                    // capture uppercase
                    keyArray.push((event.key).toUpperCase());
                }
                else {
                    // capture lowercase
                    keyArray.push(event.key);
                }
            }
            // stop when length of array is equal to lightspeedCharacterCount
            if (keyArray.length === this.settings.lightspeedCharacterCount) {
                const stringToSearch = this.settings.lightspeedJumpToStartOfWord ? "\\b" + keyArray.join("") : keyArray.join("");
                this.handleJumpToRegex(stringToSearch, this.settings.lightspeedCaseSensitive);
                // removing eventListener after proceeded
                contentEl.removeEventListener("keydown", grabKey, { capture: true });
                restoreTextColor();
            }
        };
        contentEl.addEventListener('keydown', grabKey, { capture: true });
    }
    handleHotkey(heldShiftKey, link) {
        if (link.linkElement) {
            const event = new MouseEvent("click", {
                bubbles: true,
                cancelable: true,
                view: window,
                metaKey: heldShiftKey,
            });
            link.linkElement.dispatchEvent(event);
        }
        else if (link.type === 'internal') {
            const file = this.app.workspace.getActiveFile();
            if (file) {
                // the second argument is for the link resolution
                void this.app.workspace.openLinkText(decodeURI(link.linkText), file.path, heldShiftKey, { active: true });
            }
        }
        else if (link.type === 'external') {
            window.open(link.linkText);
        }
        else {
            const editor = this.cmEditor;
            if (editor instanceof view.EditorView) {
                const index = link.index;
                const { vimMode, anchor } = this.cursorBeforeJump;
                const useSelection = heldShiftKey || (vimMode === 'visual' || vimMode === 'visual block');
                if (useSelection && anchor !== undefined) {
                    editor.dispatch({ selection: state.EditorSelection.range(anchor, index) });
                }
                else {
                    editor.dispatch({ selection: state.EditorSelection.cursor(index) });
                }
            }
            else {
                editor.setCursor(editor.posFromIndex(link.index));
            }
        }
    }
    removePopovers(linkHintHtmlElements = []) {
        const currentView = this.contentElement;
        currentView.removeEventListener('click', () => this.removePopovers(linkHintHtmlElements));
        linkHintHtmlElements === null || linkHintHtmlElements === void 0 ? void 0 : linkHintHtmlElements.forEach(e => e.remove());
        currentView.querySelectorAll('.jl.popover').forEach(e => e.remove());
        currentView.querySelectorAll('.jl-anchor').forEach(e => e.classList.remove('jl-anchor'));
        this.prefixInfo = undefined;
        if (this.mode == VIEW_MODE.SOURCE || this.mode == VIEW_MODE.LIVE_PREVIEW) {
            this.cmEditor.plugin(this.markViewPlugin).clean();
        }
        this.app.workspace.updateOptions();
        this.isLinkHintActive = false;
    }
    removePopoversWithoutPrefixEventKey(eventKey, linkHintHtmlElements = []) {
        const currentView = this.contentElement;
        linkHintHtmlElements === null || linkHintHtmlElements === void 0 ? void 0 : linkHintHtmlElements.forEach(e => {
            if (e.textContent.length == 2 && e.textContent[0] == eventKey) {
                e.classList.add("matched");
                return;
            }
            e.remove();
        });
        currentView.querySelectorAll('.jl.popover').forEach(e => {
            if (e.textContent.length == 2 && e.textContent[0] == eventKey) {
                e.classList.add("matched");
                return;
            }
            e.remove();
        });
        if (this.mode == VIEW_MODE.SOURCE || this.mode == VIEW_MODE.LIVE_PREVIEW) {
            this.cmEditor.plugin(this.markViewPlugin).filterWithEventKey(eventKey);
        }
        this.app.workspace.updateOptions();
    }
    handleActions(linkHints, linkHintHtmlElements) {
        var _a;
        const contentElement = this.contentElement;
        if (!linkHints.length) {
            return;
        }
        const linkHintMap = {};
        linkHints.forEach(x => linkHintMap[x.letter] = x);
        const handleKeyDown = (event) => {
            var _a;
            if (['Shift', 'Control', 'CapsLock', 'ScrollLock', 'GroupNext', 'Meta'].includes(event.key)) {
                return;
            }
            const eventKey = event.key.toUpperCase();
            const prefixes = new Set(Object.keys(linkHintMap).filter(x => x.length > 1).map(x => x[0]));
            let linkHint;
            if (this.prefixInfo) {
                linkHint = linkHintMap[this.prefixInfo.prefix + eventKey];
            }
            else {
                linkHint = linkHintMap[eventKey];
                if (!linkHint && prefixes && prefixes.has(eventKey)) {
                    this.prefixInfo = { prefix: eventKey, shiftKey: event.shiftKey };
                    event.preventDefault();
                    event.stopPropagation();
                    event.stopImmediatePropagation();
                    this.removePopoversWithoutPrefixEventKey(eventKey, linkHintHtmlElements);
                    return;
                }
            }
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
            const heldShiftKey = ((_a = this.prefixInfo) === null || _a === void 0 ? void 0 : _a.shiftKey) || event.shiftKey;
            if (linkHint) {
                this.handleHotkey(heldShiftKey, linkHint);
            }
            this.removePopovers(linkHintHtmlElements);
            contentElement.removeEventListener('keydown', handleKeyDown, { capture: true });
        };
        if (linkHints.length === 1 && this.settings.jumpToLinkIfOneLinkOnly) {
            const heldShiftKey = (_a = this.prefixInfo) === null || _a === void 0 ? void 0 : _a.shiftKey;
            this.handleHotkey(heldShiftKey, linkHints[0]);
            this.removePopovers(linkHintHtmlElements);
            return;
        }
        contentElement.addEventListener('click', () => this.removePopovers(linkHintHtmlElements));
        contentElement.addEventListener('keydown', handleKeyDown, { capture: true });
        this.isLinkHintActive = true;
    }
    /**
     * CodeMirror's vim automatically exits visual mode when executing a command.
     * This keeps track of selection changes so we can restore the selection.
     *
     * This is the same approach taken by the obsidian-vimrc-plugin
     */
    watchForSelectionChange() {
        const updateSelection = (editor) => this.updateSelection(editor);
        const watchForChanges = () => {
            var _a, _b;
            const editor = (_a = this.app.workspace.getActiveViewOfType(obsidian.MarkdownView)) === null || _a === void 0 ? void 0 : _a.editor;
            const cm = (_b = editor === null || editor === void 0 ? void 0 : editor.cm) === null || _b === void 0 ? void 0 : _b.cm;
            if (cm && !cm._handlers.cursorActivity.includes(updateSelection)) {
                cm.on("cursorActivity", updateSelection);
                this.register(() => cm.off("cursorActivity", updateSelection));
            }
        };
        this.registerEvent(this.app.workspace.on("active-leaf-change", watchForChanges));
        this.registerEvent(this.app.workspace.on("file-open", watchForChanges));
        watchForChanges();
    }
    updateSelection(editor) {
        var _a, _b;
        const anchor = (_a = editor.listSelections()[0]) === null || _a === void 0 ? void 0 : _a.anchor;
        this.currentCursor = {
            anchor: anchor ? editor.indexFromPos(anchor) : undefined,
            vimMode: (_b = editor.state.vim) === null || _b === void 0 ? void 0 : _b.mode
        };
    }
}
class SettingTab extends obsidian.PluginSettingTab {
    constructor(app, plugin) {
        super(app, plugin);
        this.plugin = plugin;
    }
    /**
     * The declarative settings API (Obsidian 1.13+). Obsidian renders the tab
     * from these definitions and indexes every setting for the settings search,
     * which is why there is no `display()`.
     */
    getSettingDefinitions() {
        const defaults = new Settings();
        return [
            {
                name: 'Characters used for link hints',
                desc: 'The characters placed next to each link after enter link-hint mode.',
                control: { type: 'text', key: 'letters', defaultValue: defaults.letters }
            },
            {
                name: 'Jump to anywhere',
                desc: 'Regex based navigating in editor mode',
                control: {
                    type: 'text',
                    key: 'jumpToAnywhereRegex',
                    placeholder: 'Custom regex',
                    defaultValue: defaults.jumpToAnywhereRegex
                }
            },
            {
                name: 'Lightspeed regex case sensitivity',
                desc: 'If enabled, the regex for matching will be case sensitive.',
                control: {
                    type: 'toggle',
                    key: 'lightspeedCaseSensitive',
                    defaultValue: defaults.lightspeedCaseSensitive
                }
            },
            {
                name: 'Jump to link if only one link in page',
                desc: 'If enabled, auto jump to link if there is only one link in page',
                control: {
                    type: 'toggle',
                    key: 'jumpToLinkIfOneLinkOnly',
                    defaultValue: defaults.jumpToLinkIfOneLinkOnly
                }
            },
            {
                name: 'Lightspeed only jumps to start of words',
                desc: 'If enabled, lightspeed jumps will only target characters occuring at the start of words.',
                control: {
                    type: 'toggle',
                    key: 'lightspeedJumpToStartOfWord',
                    defaultValue: defaults.lightspeedJumpToStartOfWord
                }
            },
            {
                name: 'Number of characters for lightspeed jump',
                desc: 'Determines how many characters you need to type to perform a lightspeed jump.',
                control: {
                    type: 'number',
                    key: 'lightspeedCharacterCount',
                    min: 1,
                    max: 5,
                    defaultValue: defaults.lightspeedCharacterCount
                }
            }
        ];
    }
    getControlValue(key) {
        return this.plugin.settings[key];
    }
    /** Persists through `saveData`, the store the settings are loaded from. */
    setControlValue(key, value) {
        this.plugin.settings[key] = value;
        return this.plugin.saveData(this.plugin.settings);
    }
}

module.exports = JumpToLink;
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibWFpbi5qcyIsInNvdXJjZXMiOlsidHlwZXMudHMiLCJzcmMvY202LXdpZGdldC9NYXJrV2lkZ2V0LnRzIiwic3JjL2NtNi13aWRnZXQvTWFya1BsdWdpbi50cyIsInNyYy91dGlscy9jb21tb24udHMiLCJzcmMvdXRpbHMvY202LWludGVybmFscy50cyIsInNyYy9wcm9jZXNzb3JzL0NNNkxpbmtQcm9jZXNzb3IudHMiLCJzcmMvdXRpbHMvcmVnZXhwLnRzIiwic3JjL3Byb2Nlc3NvcnMvQ002UmVnZXhQcm9jZXNzb3IudHMiLCJzcmMvcHJvY2Vzc29ycy9MZWdhY3lSZWdleHBQcm9jZXNzb3IudHMiLCJzcmMvcHJvY2Vzc29ycy9MZWdhY3lTb3VyY2VMaW5rUHJvY2Vzc29yLnRzIiwic3JjL3V0aWxzL3ByZXZpZXcudHMiLCJzcmMvcHJvY2Vzc29ycy9QcmV2aWV3TGlua1Byb2Nlc3Nvci50cyIsInNyYy9wcm9jZXNzb3JzL0xpdmVQcmV2aWV3TGlua1Byb2Nlc3Nvci50cyIsInNyYy9tYWluLnRzIl0sInNvdXJjZXNDb250ZW50IjpbImV4cG9ydCB0eXBlIExpbmtIaW50VHlwZSA9ICdpbnRlcm5hbCcgfCAnZXh0ZXJuYWwnIHwgJ3JlZ2V4JztcblxuZXhwb3J0IGludGVyZmFjZSBMaW5rSGludEJhc2Uge1xuXHRsZXR0ZXI6IHN0cmluZztcblx0dHlwZTogTGlua0hpbnRUeXBlO1xuXHRsaW5rVGV4dDogc3RyaW5nO1xuXHRsaW5rRWxlbWVudD86IEhUTUxFbGVtZW50O1xufVxuXG5leHBvcnQgaW50ZXJmYWNlIFByZXZpZXdMaW5rSGludCBleHRlbmRzIExpbmtIaW50QmFzZSB7XG5cdGxpbmtFbGVtZW50OiBIVE1MRWxlbWVudFxuXHRsZWZ0OiBudW1iZXI7XG5cdHRvcDogbnVtYmVyO1xufVxuXG5leHBvcnQgaW50ZXJmYWNlIFNvdXJjZUxpbmtIaW50IGV4dGVuZHMgTGlua0hpbnRCYXNlIHtcblx0aW5kZXg6IG51bWJlclxufVxuXG5leHBvcnQgY2xhc3MgU2V0dGluZ3Mge1xuXHQvLyBEZWZhdWx0cyBhcyBpbiBWaW1pdW0gZXh0ZW5zaW9uIGZvciBicm93c2Vyc1xuXHRsZXR0ZXJzOiBzdHJpbmcgPSAnc2FkZmprbGV3Y21wZ2gnO1xuXHRqdW1wVG9Bbnl3aGVyZVJlZ2V4OiBzdHJpbmcgPSAnXFxcXGJcXFxcd3szLH1cXFxcYic7XG5cdGxpZ2h0c3BlZWRDYXNlU2Vuc2l0aXZlOiBib29sZWFuID0gZmFsc2U7XG5cdGp1bXBUb0xpbmtJZk9uZUxpbmtPbmx5OiBib29sZWFuID0gdHJ1ZTtcblx0bGlnaHRzcGVlZEp1bXBUb1N0YXJ0T2ZXb3JkOiBib29sZWFuID0gdHJ1ZTtcblx0bGlnaHRzcGVlZENoYXJhY3RlckNvdW50OiBudW1iZXIgPSAyO1xufVxuXG5leHBvcnQgY2xhc3MgUHJvY2Vzc29yIHtcblx0bGV0dGVyczogc3RyaW5nO1xuXG5cdHB1YmxpYyBpbml0OiAoKSA9PiBMaW5rSGludEJhc2VbXTtcbn1cbiIsImltcG9ydCB7V2lkZ2V0VHlwZX0gZnJvbSBcIkBjb2RlbWlycm9yL3ZpZXdcIjtcblxuZXhwb3J0IGNsYXNzIE1hcmtXaWRnZXQgZXh0ZW5kcyBXaWRnZXRUeXBlIHtcbiAgICBjb25zdHJ1Y3RvcihyZWFkb25seSBtYXJrOiBzdHJpbmcsIHJlYWRvbmx5IHR5cGU6IHN0cmluZywgcmVhZG9ubHkgbWF0Y2hlZEV2ZW50S2V5OiBzdHJpbmcpIHtcbiAgICAgICAgc3VwZXIoKTtcbiAgICB9XG5cbiAgICBlcShvdGhlcjogTWFya1dpZGdldCkge1xuICAgICAgICByZXR1cm4gb3RoZXIubWFyayA9PT0gdGhpcy5tYXJrICYmIG90aGVyLm1hdGNoZWRFdmVudEtleSA9PSB0aGlzLm1hdGNoZWRFdmVudEtleTtcbiAgICB9XG5cbiAgICB0b0RPTSgpIHtcbiAgICAgICAgLy8gVGhlIHdpbmRvdyBsZXZlbCBgY3JlYXRlRGl2YCByZXR1cm5zIGEgREVUQUNIRUQgZWxlbWVudCwgd2hpbGUgdGhlIE5vZGVcbiAgICAgICAgLy8gbGV2ZWwgaGVscGVycyBhcHBlbmQgdG8gdGhlIG5vZGUgdGhleSBhcmUgY2FsbGVkIG9uIC0gc29cbiAgICAgICAgLy8gYGFjdGl2ZURvY3VtZW50LmNyZWF0ZURpdigpYCB3b3VsZCB0aHJvdy4gQSB3aWRnZXQgaGFzIHRvIGJlIGRldGFjaGVkLlxuICAgICAgICBjb25zdCBjbHMgPSBbXCJqbFwiLCBcImpsLVwiICsgdGhpcy50eXBlLCBcImpsLWlubGluZVwiLCBcInBvcG92ZXJcIl07XG4gICAgICAgIGlmICh0aGlzLm1hdGNoZWRFdmVudEtleSAmJiB0aGlzLm1hcmsudG9VcHBlckNhc2UoKS5zdGFydHNXaXRoKHRoaXMubWF0Y2hlZEV2ZW50S2V5LnRvVXBwZXJDYXNlKCkpKSB7XG4gICAgICAgICAgICBjbHMucHVzaCgnbWF0Y2hlZCcpO1xuICAgICAgICB9XG4gICAgICAgIGNvbnN0IHdyYXBwZXIgPSBjcmVhdGVEaXYoe2Nsc30pO1xuICAgICAgICB3cmFwcGVyLmNyZWF0ZVNwYW4oe3RleHQ6IHRoaXMubWFya30pO1xuXG4gICAgICAgIHJldHVybiB3cmFwcGVyO1xuICAgIH1cblxuICAgIGlnbm9yZUV2ZW50KCkge1xuICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgfVxufVxuIiwiaW1wb3J0IHtcbiAgICBEZWNvcmF0aW9uLFxuICAgIERlY29yYXRpb25TZXQsXG4gICAgRWRpdG9yVmlldyxcbiAgICBWaWV3VXBkYXRlLFxufSBmcm9tIFwiQGNvZGVtaXJyb3Ivdmlld1wiO1xuaW1wb3J0IHsgTWFya1dpZGdldCB9IGZyb20gXCIuL01hcmtXaWRnZXRcIjtcbmltcG9ydCB7U291cmNlTGlua0hpbnR9IGZyb20gXCIuLi8uLi90eXBlc1wiO1xuXG5leHBvcnQgY2xhc3MgTWFya1BsdWdpbiB7XG4gICAgZGVjb3JhdGlvbnM6IERlY29yYXRpb25TZXQ7XG4gICAgbGlua3M6IFNvdXJjZUxpbmtIaW50W10gPSBbXTtcbiAgICBtYXRjaGVkRXZlbnRLZXk6IHN0cmluZyB8IHVuZGVmaW5lZCA9IHVuZGVmaW5lZDtcblxuICAgIGNvbnN0cnVjdG9yKF92aWV3OiBFZGl0b3JWaWV3KSB7XG4gICAgICAgIHRoaXMubGlua3MgPSBbXTtcbiAgICAgICAgdGhpcy5tYXRjaGVkRXZlbnRLZXkgPSB1bmRlZmluZWQ7XG4gICAgICAgIHRoaXMuZGVjb3JhdGlvbnMgPSBEZWNvcmF0aW9uLm5vbmVcbiAgICB9XG5cbiAgICBzZXRMaW5rcyhsaW5rczogU291cmNlTGlua0hpbnRbXSkge1xuICAgICAgICB0aGlzLmxpbmtzID0gbGlua3M7XG4gICAgICAgIHRoaXMubWF0Y2hlZEV2ZW50S2V5ID0gdW5kZWZpbmVkO1xuICAgIH1cblxuICAgIGNsZWFuKCkge1xuICAgICAgICB0aGlzLmxpbmtzID0gW107XG4gICAgICAgIHRoaXMubWF0Y2hlZEV2ZW50S2V5ID0gdW5kZWZpbmVkO1xuICAgIH1cblxuICAgIGZpbHRlcldpdGhFdmVudEtleShldmVudEtleTogc3RyaW5nKSB7XG4gICAgICAgIGlmIChldmVudEtleS5sZW5ndGggIT0gMSkgcmV0dXJuO1xuXG4gICAgICAgIHRoaXMubGlua3MgPSB0aGlzLmxpbmtzLmZpbHRlcih2ID0+IHtcbiAgICAgICAgICAgIHJldHVybiB2LmxldHRlci5sZW5ndGggPT0gMiAmJiB2LmxldHRlclswXS50b1VwcGVyQ2FzZSgpID09IGV2ZW50S2V5LnRvVXBwZXJDYXNlKClcbiAgICAgICAgfSk7XG5cbiAgICAgICAgdGhpcy5tYXRjaGVkRXZlbnRLZXkgPSBldmVudEtleTtcbiAgICB9XG5cbiAgICBnZXQgdmlzaWJsZSgpIHtcbiAgICAgICAgcmV0dXJuIHRoaXMubGlua3MubGVuZ3RoID4gMDtcbiAgICB9XG5cbiAgICB1cGRhdGUoX3VwZGF0ZTogVmlld1VwZGF0ZSkge1xuICAgICAgICBjb25zdCB3aWRnZXRzID0gdGhpcy5saW5rcy5tYXAoKHgpID0+XG4gICAgICAgICAgICBEZWNvcmF0aW9uLndpZGdldCh7XG4gICAgICAgICAgICAgICAgd2lkZ2V0OiBuZXcgTWFya1dpZGdldCh4LmxldHRlciwgeC50eXBlLCB0aGlzLm1hdGNoZWRFdmVudEtleSksXG4gICAgICAgICAgICAgICAgc2lkZTogMSxcbiAgICAgICAgICAgIH0pLnJhbmdlKHguaW5kZXgpXG4gICAgICAgICk7XG5cbiAgICAgICAgdGhpcy5kZWNvcmF0aW9ucyA9IERlY29yYXRpb24uc2V0KHdpZGdldHMpXG4gICAgfVxufVxuXG4iLCJpbXBvcnQge0VkaXRvcn0gZnJvbSBcImNvZGVtaXJyb3JcIjtcbmltcG9ydCB7U291cmNlTGlua0hpbnR9IGZyb20gXCIuLi8uLi90eXBlc1wiO1xuXG4vKipcbiAqIEdldCBvbmx5IHZpc2libGUgY29udGVudFxuICogQHBhcmFtIGNtRWRpdG9yXG4gKiBAcmV0dXJucyBMZXR0ZXIgb2Zmc2V0IGFuZCB2aXNpYmxlIGNvbnRlbnQgYXMgYSBzdHJpbmdcbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGdldFZpc2libGVMaW5lVGV4dChjbUVkaXRvcjogRWRpdG9yKTogeyBpbmRPZmZzZXQ6IG51bWJlciwgc3Ryczogc3RyaW5nIH0ge1xuICAgIGNvbnN0IHNjcm9sbEluZm8gPSBjbUVkaXRvci5nZXRTY3JvbGxJbmZvKCk7XG4gICAgY29uc3QgeyBsaW5lOiBmcm9tIH0gPSBjbUVkaXRvci5jb29yZHNDaGFyKHsgbGVmdDogMCwgdG9wOiAwIH0sICdwYWdlJyk7XG4gICAgY29uc3QgeyBsaW5lOiB0byB9ID0gY21FZGl0b3IuY29vcmRzQ2hhcih7IGxlZnQ6IHNjcm9sbEluZm8ubGVmdCwgdG9wOiBzY3JvbGxJbmZvLnRvcCArIHNjcm9sbEluZm8uaGVpZ2h0fSlcbiAgICBjb25zdCBpbmRPZmZzZXQgPSBjbUVkaXRvci5pbmRleEZyb21Qb3Moe2NoOjAsIGxpbmU6IGZyb219KVxuICAgIGNvbnN0IHN0cnMgPSBjbUVkaXRvci5nZXRSYW5nZSh7Y2g6IDAsIGxpbmU6IGZyb219LCB7Y2g6IDAsIGxpbmU6IHRvICsgMX0pXG5cbiAgICByZXR1cm4geyBpbmRPZmZzZXQsIHN0cnMgfTtcbn1cblxuLyoqXG4gKlxuICogQHBhcmFtIGFscGhhYmV0IC0gTGV0dGVycyB3aGljaCB1c2VkIHRvIHByb2R1Y2UgaGludHNcbiAqIEBwYXJhbSBudW1MaW5rSGludHMgLSBDb3VudCBvZiBuZWVkZWQgbGlua3NcbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGdldExpbmtIaW50TGV0dGVycyhhbHBoYWJldDogc3RyaW5nLCBudW1MaW5rSGludHM6IG51bWJlcik6IHN0cmluZ1tdIHtcbiAgICBjb25zdCBhbHBoYWJldFVwcGVyY2FzZSA9IGFscGhhYmV0LnRvVXBwZXJDYXNlKClcblxuICAgIGxldCBwcmVmaXhDb3VudCA9IE1hdGguY2VpbCgobnVtTGlua0hpbnRzIC0gYWxwaGFiZXRVcHBlcmNhc2UubGVuZ3RoKSAvIChhbHBoYWJldFVwcGVyY2FzZS5sZW5ndGggLSAxKSlcblxuICAgIC8vIGVuc3VyZSAwIDw9IHByZWZpeENvdW50IDw9IGFscGhhYmV0Lmxlbmd0aFxuICAgIHByZWZpeENvdW50ID0gTWF0aC5tYXgocHJlZml4Q291bnQsIDApO1xuICAgIHByZWZpeENvdW50ID0gTWF0aC5taW4ocHJlZml4Q291bnQsIGFscGhhYmV0VXBwZXJjYXNlLmxlbmd0aCk7XG5cbiAgICBjb25zdCBwcmVmaXhlcyA9IFsnJywgLi4uQXJyYXkuZnJvbShhbHBoYWJldFVwcGVyY2FzZS5zbGljZSgwLCBwcmVmaXhDb3VudCkpXTtcblxuICAgIGNvbnN0IGxpbmtIaW50TGV0dGVycyA9IFtdXG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBwcmVmaXhlcy5sZW5ndGg7IGkrKykge1xuICAgICAgICBjb25zdCBwcmVmaXggPSBwcmVmaXhlc1tpXVxuICAgICAgICBmb3IgKGxldCBqID0gMDsgaiA8IGFscGhhYmV0VXBwZXJjYXNlLmxlbmd0aDsgaisrKSB7XG4gICAgICAgICAgICBpZiAobGlua0hpbnRMZXR0ZXJzLmxlbmd0aCA8IG51bUxpbmtIaW50cykge1xuICAgICAgICAgICAgICAgIGNvbnN0IGxldHRlciA9IGFscGhhYmV0VXBwZXJjYXNlW2pdO1xuICAgICAgICAgICAgICAgIGlmIChwcmVmaXggPT09ICcnKSB7XG4gICAgICAgICAgICAgICAgICAgIGlmICghcHJlZml4ZXMuY29udGFpbnMobGV0dGVyKSkge1xuICAgICAgICAgICAgICAgICAgICAgICAgbGlua0hpbnRMZXR0ZXJzLnB1c2gobGV0dGVyKTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgICAgIGxpbmtIaW50TGV0dGVycy5wdXNoKHByZWZpeCArIGxldHRlcilcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgIGJyZWFrO1xuICAgICAgICAgICAgfVxuICAgICAgICB9XG4gICAgfVxuXG4gICAgcmV0dXJuIGxpbmtIaW50TGV0dGVycztcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldE1ESGludExpbmtzKGNvbnRlbnQ6IHN0cmluZywgb2Zmc2V0OiBudW1iZXIsIGxldHRlcnM6IHN0cmluZyk6IFNvdXJjZUxpbmtIaW50W10ge1xuICAgIC8vIGV4cGVjdGluZyBlaXRoZXIgW1tMaW5rXV0gb3IgW1tMaW5rfFRpdGxlXV1cbiAgICBjb25zdCByZWdFeEludGVybmFsID0gL1xcW1xcWyguKz8pKFxcfC4rPyk/XV0vZztcbiAgICAvLyBleHBlY3RpbmcgW1RpdGxlXSguLi9leGFtcGxlLm1kKVxuICAgIGNvbnN0IHJlZ0V4TWRJbnRlcm5hbCA9IC9cXFtbXltcXF1dKz9cXF1cXCgoKFxcLlxcLnxcXHd8XFxkKS4rPylcXCkvZztcbiAgICAvLyBleHBlY3RpbmcgW1RpdGxlXShmaWxlOi8vbGluayksIFtUaXRsZV0oaHR0cHM6Ly9saW5rKSBvciBhbnkgb3RoZXIgW0ppcmEtMTIzXShqaXJhOi8vYmxhLWJsYSkgbGlua1xuICAgIGNvbnN0IHJlZ0V4RXh0ZXJuYWwgPSAvXFxbW15bXFxdXSs/XFxdXFwoKC4rPzpcXC9cXC8uKz8pXFwpL2c7XG4gICAgLy8gZXhwZWN0aW5nIGh0dHA6Ly9ob2dlaG9nZSBvciBodHRwczovL2hvZ2Vob2dlXG4gICAgY29uc3QgcmVnRXhVcmwgPSAvKCB8XFxufF4pKGh0dHBzPzpcXC9cXC9bXiBcXG5dKykvZztcblxuICAgIHR5cGUgSW5kZXhlZExpbmsgPSB7IGluZGV4OiBudW1iZXIsIHR5cGU6ICdpbnRlcm5hbCcgfCAnZXh0ZXJuYWwnLCBsaW5rVGV4dDogc3RyaW5nIH1cbiAgICBjb25zdCBpbmRleGVzID0gbmV3IFNldDxudW1iZXI+KClcbiAgICBjb25zdCBsaW5rc1dpdGhJbmRleDogSW5kZXhlZExpbmtbXSA9IFtdO1xuICAgIGxldCByZWdFeFJlc3VsdDtcblxuICAgIGNvbnN0IGFkZExpbmtUb0FycmF5ID0gKGxpbms6IEluZGV4ZWRMaW5rKSA9PiB7XG4gICAgICAgIGlmKGluZGV4ZXMuaGFzKGxpbmsuaW5kZXgpKSByZXR1cm5cbiAgICAgICAgaW5kZXhlcy5hZGQobGluay5pbmRleClcbiAgICAgICAgbGlua3NXaXRoSW5kZXgucHVzaChsaW5rKVxuICAgIH1cblxuICAgIHdoaWxlKChyZWdFeFJlc3VsdCA9IHJlZ0V4SW50ZXJuYWwuZXhlYyhjb250ZW50KSkgIT09IG51bGwpIHtcbiAgICAgICAgY29uc3QgbGlua1RleHQgPSByZWdFeFJlc3VsdFsxXT8udHJpbSgpO1xuICAgICAgICBhZGRMaW5rVG9BcnJheSh7IGluZGV4OiByZWdFeFJlc3VsdC5pbmRleCArIG9mZnNldCwgdHlwZTogJ2ludGVybmFsJywgbGlua1RleHQgfSk7XG4gICAgfVxuXG4gICAgLy8gRXh0ZXJuYWwgTGluayBhYm92ZSBpbnRlcm5hbCwgdG8gcHJlZmVyIHR5cGUgZXh0ZXJuYWwgb3ZlciBpbnRlcmFsIGluIGNhc2Ugb2YgYSBkdXBlXG4gICAgd2hpbGUoKHJlZ0V4UmVzdWx0ID0gcmVnRXhFeHRlcm5hbC5leGVjKGNvbnRlbnQpKSAhPT0gbnVsbCkge1xuICAgICAgICBjb25zdCBsaW5rVGV4dCA9IHJlZ0V4UmVzdWx0WzFdO1xuICAgICAgICBhZGRMaW5rVG9BcnJheSh7IGluZGV4OiByZWdFeFJlc3VsdC5pbmRleCArIG9mZnNldCwgdHlwZTogJ2V4dGVybmFsJywgbGlua1RleHQgfSlcbiAgICB9XG5cbiAgICB3aGlsZSgocmVnRXhSZXN1bHQgPSByZWdFeE1kSW50ZXJuYWwuZXhlYyhjb250ZW50KSkgIT09IG51bGwpIHtcbiAgICAgICAgY29uc3QgbGlua1RleHQgPSByZWdFeFJlc3VsdFsxXTtcbiAgICAgICAgYWRkTGlua1RvQXJyYXkoeyBpbmRleDogcmVnRXhSZXN1bHQuaW5kZXggKyBvZmZzZXQsIHR5cGU6ICdpbnRlcm5hbCcsIGxpbmtUZXh0IH0pO1xuICAgIH1cblxuICAgIHdoaWxlKChyZWdFeFJlc3VsdCA9IHJlZ0V4VXJsLmV4ZWMoY29udGVudCkpICE9PSBudWxsKSB7XG4gICAgICAgIGNvbnN0IGxpbmtUZXh0ID0gcmVnRXhSZXN1bHRbMl07XG4gICAgICAgIGFkZExpbmtUb0FycmF5KHsgaW5kZXg6IHJlZ0V4UmVzdWx0LmluZGV4ICsgb2Zmc2V0ICsgMSwgdHlwZTogJ2V4dGVybmFsJywgbGlua1RleHQgfSlcbiAgICB9XG5cbiAgICBjb25zdCBsaW5rSGludExldHRlcnMgPSBnZXRMaW5rSGludExldHRlcnMobGV0dGVycywgbGlua3NXaXRoSW5kZXgubGVuZ3RoKTtcblxuICAgIGNvbnN0IGxpbmtzV2l0aExldHRlcjogU291cmNlTGlua0hpbnRbXSA9IFtdO1xuICAgIGxpbmtzV2l0aEluZGV4XG4gICAgICAgIC5zb3J0KCh4LHkpID0+IHguaW5kZXggLSB5LmluZGV4KVxuICAgICAgICAuZm9yRWFjaCgobGlua0hpbnQsIGkpID0+IHtcbiAgICAgICAgICAgIGxpbmtzV2l0aExldHRlci5wdXNoKHsgbGV0dGVyOiBsaW5rSGludExldHRlcnNbaV0sIC4uLmxpbmtIaW50fSk7XG4gICAgICAgIH0pO1xuXG4gICAgcmV0dXJuIGxpbmtzV2l0aExldHRlci5maWx0ZXIobGluayA9PiBsaW5rLmxldHRlcik7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVXaWRnZXRFbGVtZW50KGNvbnRlbnQ6IHN0cmluZywgdHlwZTogc3RyaW5nKSB7XG4gICAgcmV0dXJuIGNyZWF0ZURpdih7Y2xzOiBbJ2psJywgJ2psLScgKyB0eXBlLCAncG9wb3ZlciddLCB0ZXh0OiBjb250ZW50fSk7XG59XG5cbi8qKlxuICogQ29kZU1pcnJvciA1IGV4cG9zZXMgYGFkZFdpZGdldGAgYnV0IGl0IGlzIG1pc3NpbmcgZnJvbSB0aGUgcHVibGlzaGVkIHR5cGluZ3MsXG4gKiBhbmQgaXRzIGZvdXJ0aCBwYXJhbWV0ZXIgKHBsYWNlbWVudCkgaXMgdW5kb2N1bWVudGVkLlxuICovXG5pbnRlcmZhY2UgTGVnYWN5V2lkZ2V0RWRpdG9yIHtcbiAgICBhZGRXaWRnZXQocG9zOiB7IGxpbmU6IG51bWJlciwgY2g6IG51bWJlciB9LCBub2RlOiBIVE1MRWxlbWVudCwgc2Nyb2xsSW50b1ZpZXc6IGJvb2xlYW4sIHZlcnQ/OiBzdHJpbmcpOiB2b2lkO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZGlzcGxheVNvdXJjZVBvcG92ZXJzKGNtRWRpdG9yOiBFZGl0b3IsIGxpbmtLZXlNYXA6IFNvdXJjZUxpbmtIaW50W10pOiB2b2lkIHtcbiAgICBjb25zdCBkcmF3V2lkZ2V0ID0gKGNtRWRpdG9yOiBFZGl0b3IsIGxpbmtIaW50OiBTb3VyY2VMaW5rSGludCkgPT4ge1xuICAgICAgICBjb25zdCBwb3MgPSBjbUVkaXRvci5wb3NGcm9tSW5kZXgobGlua0hpbnQuaW5kZXgpO1xuICAgICAgICAoY21FZGl0b3IgYXMgdW5rbm93biBhcyBMZWdhY3lXaWRnZXRFZGl0b3IpLmFkZFdpZGdldChwb3MsIGNyZWF0ZVdpZGdldEVsZW1lbnQobGlua0hpbnQubGV0dGVyLCBsaW5rSGludC50eXBlKSwgZmFsc2UsICdvdmVyJyk7XG4gICAgfVxuXG4gICAgbGlua0tleU1hcC5mb3JFYWNoKHggPT4gZHJhd1dpZGdldChjbUVkaXRvciwgeCkpO1xufVxuIiwiaW1wb3J0IHtFZGl0b3JWaWV3fSBmcm9tIFwiQGNvZGVtaXJyb3Ivdmlld1wiO1xuXG4vKipcbiAqIGB2aWV3U3RhdGVgIGlzIGludGVybmFsIHRvIENvZGVNaXJyb3IgNiBhbmQgaXMgbm90IHBhcnQgb2YgaXRzIHB1YmxpYyB0eXBpbmdzLFxuICogYnV0IGl0IGlzIHRoZSBvbmx5IHdheSB0byBsZWFybiB3aGljaCBsaW5lcyBhcmUgYWN0dWFsbHkgb24gc2NyZWVuLlxuICovXG5pbnRlcmZhY2UgVmlld1N0YXRlSW50ZXJuYWxzIHtcbiAgICBwaXhlbFZpZXdwb3J0PzogeyB0b3A6IG51bWJlciB9O1xuICAgIHZpZXdwb3J0TGluZXM/OiB7IHRvcDogbnVtYmVyLCBmcm9tOiBudW1iZXIgfVtdO1xufVxuXG4vKipcbiAqIFJldHVybnMgdGhlIHJhbmdlIG9mIHRoZSBkb2N1bWVudCB3aGljaCBpcyB2aXNpYmxlIG9uIHNjcmVlbiwgdHJpbW1pbmcgdGhlXG4gKiBsaW5lcyBDb2RlTWlycm9yIHJlbmRlcnMgYWJvdmUgdGhlIHZpZXdwb3J0IHdoZW4gdGhhdCBpbmZvcm1hdGlvbiBpcyBhdmFpbGFibGUuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBnZXRWaXNpYmxlUmFuZ2UoY21FZGl0b3I6IEVkaXRvclZpZXcpOiB7IGluZGV4OiBudW1iZXIsIGNvbnRlbnQ6IHN0cmluZyB9IHtcbiAgICBsZXQgeyBmcm9tIH0gPSBjbUVkaXRvci52aWV3cG9ydDtcbiAgICBjb25zdCB7IHRvIH0gPSBjbUVkaXRvci52aWV3cG9ydDtcblxuICAgIGNvbnN0IHZpZXdTdGF0ZSA9IChjbUVkaXRvciBhcyB1bmtub3duIGFzIHsgdmlld1N0YXRlPzogVmlld1N0YXRlSW50ZXJuYWxzIH0pLnZpZXdTdGF0ZTtcbiAgICBjb25zdCBwaXhlbE9mZnNldFRvcCA9IHZpZXdTdGF0ZT8ucGl4ZWxWaWV3cG9ydD8udG9wO1xuXG4gICAgaWYgKHBpeGVsT2Zmc2V0VG9wKSB7XG4gICAgICAgIGNvbnN0IGxpbmVzID0gdmlld1N0YXRlLnZpZXdwb3J0TGluZXMgPz8gW107XG4gICAgICAgIGZyb20gPSBsaW5lcy5maWx0ZXIobGluZSA9PiBsaW5lLnRvcCA+IHBpeGVsT2Zmc2V0VG9wKVswXT8uZnJvbTtcbiAgICB9XG5cbiAgICByZXR1cm4geyBpbmRleDogZnJvbSwgY29udGVudDogY21FZGl0b3Iuc3RhdGUuc2xpY2VEb2MoZnJvbSwgdG8pIH07XG59XG4iLCJpbXBvcnQge1Byb2Nlc3NvciwgU291cmNlTGlua0hpbnR9IGZyb20gXCIuLi8uLi90eXBlc1wiO1xuaW1wb3J0IHtFZGl0b3JWaWV3fSBmcm9tIFwiQGNvZGVtaXJyb3Ivdmlld1wiO1xuaW1wb3J0IHtnZXRNREhpbnRMaW5rc30gZnJvbSBcIi4uL3V0aWxzL2NvbW1vblwiO1xuaW1wb3J0IHtnZXRWaXNpYmxlUmFuZ2V9IGZyb20gXCIuLi91dGlscy9jbTYtaW50ZXJuYWxzXCI7XG5cbmV4cG9ydCBkZWZhdWx0IGNsYXNzIENNNkxpbmtQcm9jZXNzb3IgaW1wbGVtZW50cyBQcm9jZXNzb3Ige1xuICAgIGNtRWRpdG9yOiBFZGl0b3JWaWV3O1xuICAgIGxldHRlcnM6IHN0cmluZztcblxuICAgIGNvbnN0cnVjdG9yKGVkaXRvcjogRWRpdG9yVmlldywgYWxwaGFiZXQ6IHN0cmluZykge1xuICAgICAgICB0aGlzLmNtRWRpdG9yID0gZWRpdG9yO1xuICAgICAgICB0aGlzLmxldHRlcnMgPSBhbHBoYWJldDtcbiAgICB9XG5cbiAgICBwdWJsaWMgaW5pdCgpOiBTb3VyY2VMaW5rSGludFtdIHtcbiAgICAgICAgcmV0dXJuIHRoaXMuZ2V0U291cmNlTGlua0hpbnRzKCk7XG4gICAgfVxuXG4gICAgcHVibGljIGdldFZpc2libGVMaW5lcygpIHtcbiAgICAgICAgcmV0dXJuIGdldFZpc2libGVSYW5nZSh0aGlzLmNtRWRpdG9yKTtcbiAgICB9XG5cbiAgICBwcml2YXRlIGdldFNvdXJjZUxpbmtIaW50cyA9ICgpOiBTb3VyY2VMaW5rSGludFtdID0+IHtcbiAgICAgICAgY29uc3QgeyBsZXR0ZXJzIH0gPSB0aGlzO1xuICAgICAgICBjb25zdCB7IGluZGV4LCBjb250ZW50IH0gPSB0aGlzLmdldFZpc2libGVMaW5lcygpO1xuXG4gICAgICAgIHJldHVybiBnZXRNREhpbnRMaW5rcyhjb250ZW50LCBpbmRleCwgbGV0dGVycyk7XG4gICAgfVxufSIsImltcG9ydCB7Z2V0TGlua0hpbnRMZXR0ZXJzfSBmcm9tIFwiLi9jb21tb25cIjtcbmltcG9ydCB7U291cmNlTGlua0hpbnR9IGZyb20gXCIuLi8uLi90eXBlc1wiO1xuXG5leHBvcnQgZnVuY3Rpb24gZXh0cmFjdFJlZ2V4cEJsb2Nrcyhjb250ZW50OiBzdHJpbmcsIG9mZnNldDogbnVtYmVyLCByZWdleHA6IHN0cmluZywgbGV0dGVyczogc3RyaW5nLCBjYXNlU2Vuc2l0aXZlOiBib29sZWFuKSB7XG4gICAgY29uc3QgcmVnRXhVcmwgPSBjYXNlU2Vuc2l0aXZlID8gbmV3IFJlZ0V4cChyZWdleHAsICdnJykgOiBuZXcgUmVnRXhwKHJlZ2V4cCwgJ2lnJyk7XG5cbiAgICBsZXQgbGlua3NXaXRoSW5kZXg6IHtcbiAgICAgICAgaW5kZXg6IG51bWJlcjtcbiAgICAgICAgdHlwZTogXCJyZWdleFwiO1xuICAgICAgICBsaW5rVGV4dDogc3RyaW5nO1xuICAgIH1bXSA9IFtdO1xuXG4gICAgbGV0IHJlZ0V4UmVzdWx0O1xuXG4gICAgd2hpbGUgKChyZWdFeFJlc3VsdCA9IHJlZ0V4VXJsLmV4ZWMoY29udGVudCkpKSB7XG4gICAgICAgIGNvbnN0IGxpbmtUZXh0ID0gcmVnRXhSZXN1bHRbMV07XG4gICAgICAgIGxpbmtzV2l0aEluZGV4LnB1c2goe1xuICAgICAgICAgICAgaW5kZXg6IHJlZ0V4UmVzdWx0LmluZGV4ICsgb2Zmc2V0LFxuICAgICAgICAgICAgdHlwZTogXCJyZWdleFwiLFxuICAgICAgICAgICAgbGlua1RleHQsXG4gICAgICAgIH0pO1xuICAgIH1cblxuICAgIGNvbnN0IGxpbmtIaW50TGV0dGVycyA9IGdldExpbmtIaW50TGV0dGVycyhsZXR0ZXJzLCBsaW5rc1dpdGhJbmRleC5sZW5ndGgpO1xuXG4gICAgY29uc3QgbGlua3NXaXRoTGV0dGVyOiBTb3VyY2VMaW5rSGludFtdID0gW107XG4gICAgbGlua3NXaXRoSW5kZXhcbiAgICAgICAgLnNvcnQoKHgsIHkpID0+IHguaW5kZXggLSB5LmluZGV4KVxuICAgICAgICAuZm9yRWFjaCgobGlua0hpbnQsIGkpID0+IHtcbiAgICAgICAgICAgIGxpbmtzV2l0aExldHRlci5wdXNoKHtcbiAgICAgICAgICAgICAgICBsZXR0ZXI6IGxpbmtIaW50TGV0dGVyc1tpXSxcbiAgICAgICAgICAgICAgICAuLi5saW5rSGludCxcbiAgICAgICAgICAgIH0pO1xuICAgICAgICB9KTtcblxuICAgIHJldHVybiBsaW5rc1dpdGhMZXR0ZXIuZmlsdGVyKGxpbmsgPT4gbGluay5sZXR0ZXIpO1xufVxuIiwiaW1wb3J0IENNNkxpbmtQcm9jZXNzb3IgZnJvbSBcIi4vQ002TGlua1Byb2Nlc3NvclwiO1xuaW1wb3J0IHtQcm9jZXNzb3J9IGZyb20gXCIuLi8uLi90eXBlc1wiO1xuaW1wb3J0IHtFZGl0b3JWaWV3fSBmcm9tIFwiQGNvZGVtaXJyb3Ivdmlld1wiO1xuaW1wb3J0IHtleHRyYWN0UmVnZXhwQmxvY2tzfSBmcm9tIFwiLi4vdXRpbHMvcmVnZXhwXCI7XG5cbmV4cG9ydCBkZWZhdWx0IGNsYXNzIENNNlJlZ2V4UHJvY2Vzc29yIGV4dGVuZHMgQ002TGlua1Byb2Nlc3NvciBpbXBsZW1lbnRzIFByb2Nlc3NvciB7XG4gICAgcmVnZXhwOiBzdHJpbmc7XG4gICAgY2FzZVNlbnNpdGl2ZTogYm9vbGVhbjtcbiAgICBjb25zdHJ1Y3RvcihlZGl0b3I6IEVkaXRvclZpZXcsIGFscGhhYmV0OiBzdHJpbmcsIHJlZ2V4cDogc3RyaW5nLCBjYXNlU2Vuc2l0aXZlOiBib29sZWFuKSB7XG4gICAgICAgIHN1cGVyKGVkaXRvciwgYWxwaGFiZXQpO1xuICAgICAgICB0aGlzLnJlZ2V4cCA9IHJlZ2V4cDtcbiAgICAgICAgdGhpcy5jYXNlU2Vuc2l0aXZlID0gY2FzZVNlbnNpdGl2ZTtcbiAgICB9XG5cbiAgICBpbml0KCkge1xuICAgICAgICBjb25zdCB7IGxldHRlcnMsIHJlZ2V4cCB9ID0gdGhpcztcbiAgICAgICAgY29uc3QgeyBpbmRleCwgY29udGVudCB9ID0gdGhpcy5nZXRWaXNpYmxlTGluZXMoKTtcbiAgICAgICAgcmV0dXJuIGV4dHJhY3RSZWdleHBCbG9ja3MoY29udGVudCwgaW5kZXgsIHJlZ2V4cCwgbGV0dGVycywgdGhpcy5jYXNlU2Vuc2l0aXZlKTtcbiAgICB9XG59XG4iLCJpbXBvcnQge0VkaXRvcn0gZnJvbSBcImNvZGVtaXJyb3JcIjtcbmltcG9ydCB7UHJvY2Vzc29yLCBTb3VyY2VMaW5rSGludH0gZnJvbSBcIi4uLy4uL3R5cGVzXCI7XG5pbXBvcnQge2Rpc3BsYXlTb3VyY2VQb3BvdmVycywgZ2V0VmlzaWJsZUxpbmVUZXh0fSBmcm9tIFwiLi4vdXRpbHMvY29tbW9uXCI7XG5pbXBvcnQge2V4dHJhY3RSZWdleHBCbG9ja3N9IGZyb20gXCIuLi91dGlscy9yZWdleHBcIjtcblxuZXhwb3J0IGRlZmF1bHQgY2xhc3MgTGVnYWN5UmVnZXhwUHJvY2Vzc29yIGltcGxlbWVudHMgUHJvY2Vzc29yIHtcbiAgICBjbUVkaXRvcjogRWRpdG9yO1xuICAgIHJlZ2V4cDogc3RyaW5nO1xuICAgIGxldHRlcnM6IHN0cmluZztcbiAgICBjYXNlU2Vuc2l0aXZlOiBib29sZWFuO1xuXG4gICAgY29uc3RydWN0b3IoY21FZGl0b3I6IEVkaXRvciwgcmVnZXhwOiBzdHJpbmcsIGFscGhhYmV0OiBzdHJpbmcsIGNhc2VTZW5zaXRpdmU6IGJvb2xlYW4pIHtcbiAgICAgICAgdGhpcy5jbUVkaXRvciA9IGNtRWRpdG9yO1xuICAgICAgICB0aGlzLnJlZ2V4cCA9IHJlZ2V4cDtcbiAgICAgICAgdGhpcy5sZXR0ZXJzID0gYWxwaGFiZXQ7XG4gICAgICAgIHRoaXMuY2FzZVNlbnNpdGl2ZSA9IGNhc2VTZW5zaXRpdmU7XG4gICAgfVxuXG4gICAgcHVibGljIGluaXQoKTogU291cmNlTGlua0hpbnRbXSB7XG4gICAgICAgIGNvbnN0IFtjb250ZW50LCBvZmZzZXRdID0gdGhpcy5nZXRWaXNpYmxlQ29udGVudCgpO1xuICAgICAgICBjb25zdCBsaW5rcyA9IHRoaXMuZ2V0TGlua3MoY29udGVudCwgb2Zmc2V0KTtcblxuICAgICAgICB0aGlzLmRpc3BsYXkobGlua3MpO1xuICAgICAgICByZXR1cm4gbGlua3M7XG4gICAgfVxuXG4gICAgcHJpdmF0ZSBnZXRWaXNpYmxlQ29udGVudCgpOiBbc3RyaW5nLCBudW1iZXJdIHtcbiAgICAgICAgY29uc3QgeyBjbUVkaXRvciB9ID0gdGhpcztcbiAgICAgICAgY29uc3QgeyBpbmRPZmZzZXQsIHN0cnMgfSA9IGdldFZpc2libGVMaW5lVGV4dChjbUVkaXRvcik7XG5cbiAgICAgICAgcmV0dXJuIFtzdHJzLCBpbmRPZmZzZXRdO1xuICAgIH1cblxuICAgIHByaXZhdGUgZ2V0TGlua3MoY29udGVudDogc3RyaW5nLCBvZmZzZXQ6IG51bWJlcik6IFNvdXJjZUxpbmtIaW50W10ge1xuICAgICAgICBjb25zdCB7IHJlZ2V4cCwgbGV0dGVycyB9ID0gdGhpc1xuICAgICAgICByZXR1cm4gZXh0cmFjdFJlZ2V4cEJsb2Nrcyhjb250ZW50LCBvZmZzZXQsIHJlZ2V4cCwgbGV0dGVycywgdGhpcy5jYXNlU2Vuc2l0aXZlKTtcbiAgICB9XG5cbiAgICBwcml2YXRlIGRpc3BsYXkobGlua3M6IFNvdXJjZUxpbmtIaW50W10pOiB2b2lkIHtcbiAgICAgICAgY29uc3QgeyBjbUVkaXRvciB9ID0gdGhpc1xuICAgICAgICBkaXNwbGF5U291cmNlUG9wb3ZlcnMoY21FZGl0b3IsIGxpbmtzKTtcbiAgICB9XG59XG4iLCJpbXBvcnQge1Byb2Nlc3NvciwgU291cmNlTGlua0hpbnR9IGZyb20gXCIuLi8uLi90eXBlc1wiO1xuaW1wb3J0IHtFZGl0b3J9IGZyb20gXCJjb2RlbWlycm9yXCI7XG5pbXBvcnQge2Rpc3BsYXlTb3VyY2VQb3BvdmVycywgZ2V0TURIaW50TGlua3MsIGdldFZpc2libGVMaW5lVGV4dH0gZnJvbSBcIi4uL3V0aWxzL2NvbW1vblwiO1xuXG5leHBvcnQgZGVmYXVsdCBjbGFzcyBMZWdhY3lTb3VyY2VMaW5rUHJvY2Vzc29yIGltcGxlbWVudHMgUHJvY2Vzc29yIHtcbiAgICBjbUVkaXRvcjogRWRpdG9yO1xuICAgIGxldHRlcnM6IHN0cmluZztcblxuICAgIGNvbnN0cnVjdG9yKGVkaXRvcjogRWRpdG9yLCBhbHBoYWJldDogc3RyaW5nKSB7XG4gICAgICAgIHRoaXMuY21FZGl0b3IgPSBlZGl0b3I7XG4gICAgICAgIHRoaXMubGV0dGVycyA9IGFscGhhYmV0O1xuICAgIH1cblxuICAgIHB1YmxpYyBpbml0KCkge1xuICAgICAgICBjb25zdCB7IGNtRWRpdG9yIH0gPSB0aGlzO1xuXG4gICAgICAgIGNvbnN0IGxpbmtIaW50cyA9IHRoaXMuZ2V0U291cmNlTGlua0hpbnRzKGNtRWRpdG9yKTtcbiAgICAgICAgZGlzcGxheVNvdXJjZVBvcG92ZXJzKGNtRWRpdG9yLCBsaW5rSGludHMpO1xuXG4gICAgICAgIHJldHVybiBsaW5rSGludHM7XG4gICAgfVxuXG4gICAgcHJpdmF0ZSBnZXRTb3VyY2VMaW5rSGludHMgPSAoY21FZGl0b3I6IEVkaXRvcik6IFNvdXJjZUxpbmtIaW50W10gPT4ge1xuICAgICAgICBjb25zdCB7IGxldHRlcnMgfSA9IHRoaXM7XG4gICAgICAgIGNvbnN0IHsgaW5kT2Zmc2V0LCBzdHJzIH0gPSBnZXRWaXNpYmxlTGluZVRleHQoY21FZGl0b3IpO1xuXG4gICAgICAgIHJldHVybiBnZXRNREhpbnRMaW5rcyhzdHJzLCBpbmRPZmZzZXQsIGxldHRlcnMpO1xuICAgIH1cbn0iLCJpbXBvcnQge0xpbmtIaW50VHlwZSwgUHJldmlld0xpbmtIaW50fSBmcm9tIFwiLi4vLi4vdHlwZXNcIjtcbmltcG9ydCB7Z2V0TGlua0hpbnRMZXR0ZXJzfSBmcm9tIFwiLi9jb21tb25cIjtcblxuZXhwb3J0IGZ1bmN0aW9uIGdldFByZXZpZXdMaW5rSGludHMocHJldmlld1ZpZXdFbDogSFRNTEVsZW1lbnQsIGxldHRlcnM6IHN0cmluZyApOiBQcmV2aWV3TGlua0hpbnRbXSB7XG4gICAgY29uc3QgYW5jaG9yRWxzID0gcHJldmlld1ZpZXdFbC5xdWVyeVNlbGVjdG9yQWxsPEhUTUxBbmNob3JFbGVtZW50PignYSwgLm1ldGFkYXRhLWxpbmstaW5uZXInKTtcbiAgICBjb25zdCBlbWJlZEVscyA9IHByZXZpZXdWaWV3RWwucXVlcnlTZWxlY3RvckFsbCgnLmludGVybmFsLWVtYmVkJyk7XG5cbiAgICBjb25zdCBsaW5rSGludHM6IFByZXZpZXdMaW5rSGludFtdID0gW107XG4gICAgYW5jaG9yRWxzLmZvckVhY2goKGFuY2hvckVsLCBfaSkgPT4ge1xuICAgICAgICBpZiAoY2hlY2tJc1ByZXZpZXdFbE9uU2NyZWVuKHByZXZpZXdWaWV3RWwsIGFuY2hvckVsKSkge1xuICAgICAgICAgICAgcmV0dXJuXG4gICAgICAgIH1cblxuICAgICAgICBjb25zdCBsaW5rVHlwZTogTGlua0hpbnRUeXBlID0gYW5jaG9yRWwuY2xhc3NMaXN0LmNvbnRhaW5zKCdpbnRlcm5hbC1saW5rJylcbiAgICAgICAgICAgID8gJ2ludGVybmFsJ1xuICAgICAgICAgICAgOiAnZXh0ZXJuYWwnO1xuXG4gICAgICAgIGNvbnN0IGxpbmtUZXh0ID0gbGlua1R5cGUgPT09ICdpbnRlcm5hbCdcbiAgICAgICAgICAgID8gYW5jaG9yRWwuZGF0YXNldFsnaHJlZiddID8/IGFuY2hvckVsLmhyZWZcbiAgICAgICAgICAgIDogYW5jaG9yRWwuaHJlZjtcblxuICAgICAgICBsZXQgb2Zmc2V0UGFyZW50ID0gYW5jaG9yRWwub2Zmc2V0UGFyZW50IGFzIEhUTUxFbGVtZW50O1xuICAgICAgICBsZXQgdG9wID0gYW5jaG9yRWwub2Zmc2V0VG9wO1xuICAgICAgICBsZXQgbGVmdCA9IGFuY2hvckVsLm9mZnNldExlZnQ7XG5cbiAgICAgICAgd2hpbGUgKG9mZnNldFBhcmVudCkge1xuICAgICAgICAgICAgaWYgKG9mZnNldFBhcmVudCA9PSBwcmV2aWV3Vmlld0VsKSB7XG4gICAgICAgICAgICAgICAgb2Zmc2V0UGFyZW50ID0gdW5kZWZpbmVkO1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICB0b3AgKz0gb2Zmc2V0UGFyZW50Lm9mZnNldFRvcDtcbiAgICAgICAgICAgICAgICBsZWZ0ICs9IG9mZnNldFBhcmVudC5vZmZzZXRMZWZ0O1xuICAgICAgICAgICAgICAgIG9mZnNldFBhcmVudCA9IG9mZnNldFBhcmVudC5vZmZzZXRQYXJlbnQgYXMgSFRNTEVsZW1lbnQ7XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgbGlua0hpbnRzLnB1c2goe1xuICAgICAgICAgICAgbGlua0VsZW1lbnQ6IGFuY2hvckVsLFxuICAgICAgICAgICAgbGV0dGVyOiAnJyxcbiAgICAgICAgICAgIGxpbmtUZXh0OiBsaW5rVGV4dCxcbiAgICAgICAgICAgIHR5cGU6IGxpbmtUeXBlLFxuICAgICAgICAgICAgdG9wOiB0b3AsXG4gICAgICAgICAgICBsZWZ0OiBsZWZ0LFxuICAgICAgICB9KTtcbiAgICB9KTtcblxuICAgIGVtYmVkRWxzLmZvckVhY2goKGVtYmVkRWwsIF9pKSA9PiB7XG4gICAgICAgIGNvbnN0IGxpbmtUZXh0ID0gZW1iZWRFbC5nZXRBdHRyaWJ1dGUoJ3NyYycpO1xuICAgICAgICBjb25zdCBsaW5rRWwgPSBlbWJlZEVsLnF1ZXJ5U2VsZWN0b3I8SFRNTEVsZW1lbnQ+KCcubWFya2Rvd24tZW1iZWQtbGluaycpO1xuXG4gICAgICAgIGlmIChsaW5rVGV4dCAmJiBsaW5rRWwpIHtcbiAgICAgICAgICAgIGlmIChjaGVja0lzUHJldmlld0VsT25TY3JlZW4ocHJldmlld1ZpZXdFbCwgbGlua0VsKSkge1xuICAgICAgICAgICAgICAgIHJldHVyblxuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICBsZXQgb2Zmc2V0UGFyZW50ID0gbGlua0VsLm9mZnNldFBhcmVudCBhcyBIVE1MRWxlbWVudDtcbiAgICAgICAgICAgIGxldCB0b3AgPSBsaW5rRWwub2Zmc2V0VG9wO1xuICAgICAgICAgICAgbGV0IGxlZnQgPSBsaW5rRWwub2Zmc2V0TGVmdDtcblxuICAgICAgICAgICAgd2hpbGUgKG9mZnNldFBhcmVudCkge1xuICAgICAgICAgICAgICAgIGlmIChvZmZzZXRQYXJlbnQgPT0gcHJldmlld1ZpZXdFbCkge1xuICAgICAgICAgICAgICAgICAgICBvZmZzZXRQYXJlbnQgPSB1bmRlZmluZWQ7XG4gICAgICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICAgICAgdG9wICs9IG9mZnNldFBhcmVudC5vZmZzZXRUb3A7XG4gICAgICAgICAgICAgICAgICAgIGxlZnQgKz0gb2Zmc2V0UGFyZW50Lm9mZnNldExlZnQ7XG4gICAgICAgICAgICAgICAgICAgIG9mZnNldFBhcmVudCA9IG9mZnNldFBhcmVudC5vZmZzZXRQYXJlbnQgYXMgSFRNTEVsZW1lbnQ7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICBsaW5rSGludHMucHVzaCh7XG4gICAgICAgICAgICAgICAgbGlua0VsZW1lbnQ6IGxpbmtFbCxcbiAgICAgICAgICAgICAgICBsZXR0ZXI6ICcnLFxuICAgICAgICAgICAgICAgIGxpbmtUZXh0OiBsaW5rVGV4dCxcbiAgICAgICAgICAgICAgICB0eXBlOiAnaW50ZXJuYWwnLFxuICAgICAgICAgICAgICAgIHRvcDogdG9wLFxuICAgICAgICAgICAgICAgIGxlZnQ6IGxlZnQsXG4gICAgICAgICAgICB9KTtcbiAgICAgICAgfVxuICAgIH0pO1xuXG4gICAgY29uc3Qgc29ydGVkTGlua0hpbnRzID0gbGlua0hpbnRzLnNvcnQoKGEsIGIpID0+IHtcbiAgICAgICAgaWYgKGEudG9wID4gYi50b3ApIHtcbiAgICAgICAgICAgIHJldHVybiAxO1xuICAgICAgICB9IGVsc2UgaWYgKGEudG9wID09PSBiLnRvcCkge1xuICAgICAgICAgICAgaWYgKGEubGVmdCA+IGIubGVmdCkge1xuICAgICAgICAgICAgICAgIHJldHVybiAxO1xuICAgICAgICAgICAgfSBlbHNlIGlmIChhLmxlZnQgPT09IGIubGVmdCkge1xuICAgICAgICAgICAgICAgIHJldHVybiAwO1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICByZXR1cm4gLTE7XG4gICAgICAgICAgICB9XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByZXR1cm4gLTE7XG4gICAgICAgIH1cbiAgICB9KTtcblxuICAgIGNvbnN0IGxpbmtIaW50TGV0dGVycyA9IGdldExpbmtIaW50TGV0dGVycyhsZXR0ZXJzLCBzb3J0ZWRMaW5rSGludHMubGVuZ3RoKTtcblxuICAgIHNvcnRlZExpbmtIaW50cy5mb3JFYWNoKChsaW5rSGludCwgaSkgPT4ge1xuICAgICAgICBsaW5rSGludC5sZXR0ZXIgPSBsaW5rSGludExldHRlcnNbaV07XG4gICAgfSk7XG5cbiAgICByZXR1cm4gc29ydGVkTGlua0hpbnRzO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY2hlY2tJc1ByZXZpZXdFbE9uU2NyZWVuKHBhcmVudDogSFRNTEVsZW1lbnQsIGVsOiBIVE1MRWxlbWVudCkge1xuICAgIGVsID0gZWwuY2xvc2VzdCgnW2RhdGEtdmlldy10eXBlPVwidGFibGVcIl0sIHRhYmxlJykgfHwgZWw7XG4gICAgcmV0dXJuIGVsLm9mZnNldFRvcCA8IHBhcmVudC5zY3JvbGxUb3AgfHwgZWwub2Zmc2V0VG9wID4gcGFyZW50LnNjcm9sbFRvcCArIHBhcmVudC5vZmZzZXRIZWlnaHRcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGRpc3BsYXlQcmV2aWV3UG9wb3ZlcnMobGlua0hpbnRzOiBQcmV2aWV3TGlua0hpbnRbXSk6IEhUTUxFbGVtZW50W10ge1xuICAgIGNvbnN0IGxpbmtIaW50SHRtbEVsZW1lbnRzOiBIVE1MRWxlbWVudFtdID0gW11cbiAgICBmb3IgKGNvbnN0IGxpbmtIaW50IG9mIGxpbmtIaW50cykge1xuICAgICAgICBsaW5rSGludC5saW5rRWxlbWVudC5jbGFzc0xpc3QuYWRkKCdqbC1hbmNob3InKTtcbiAgICAgICAgY29uc3QgcG9wb3ZlckVsZW1lbnQgPSBsaW5rSGludC5saW5rRWxlbWVudC5jcmVhdGVTcGFuKHtcbiAgICAgICAgICAgIHRleHQ6IGxpbmtIaW50LmxldHRlcixcbiAgICAgICAgICAgIGNsczogWydqbCcsICdqbC0nICsgbGlua0hpbnQudHlwZSwgJ2psLXByZXZpZXcnLCAncG9wb3ZlciddLFxuICAgICAgICB9KTtcbiAgICAgICAgbGlua0hpbnRIdG1sRWxlbWVudHMucHVzaChwb3BvdmVyRWxlbWVudClcbiAgICB9XG4gICAgcmV0dXJuIGxpbmtIaW50SHRtbEVsZW1lbnRzXG59XG4iLCJpbXBvcnQge1ByZXZpZXdMaW5rSGludH0gZnJvbSBcIi4uLy4uL3R5cGVzXCI7XG5pbXBvcnQge2Rpc3BsYXlQcmV2aWV3UG9wb3ZlcnMsIGdldFByZXZpZXdMaW5rSGludHN9IGZyb20gXCIuLi91dGlscy9wcmV2aWV3XCI7XG5cbmV4cG9ydCBkZWZhdWx0IGNsYXNzIFByZXZpZXdMaW5rUHJvY2Vzc29yIHtcbiAgICB2aWV3OiBIVE1MRWxlbWVudDtcbiAgICBhbHBoYWJldDogc3RyaW5nO1xuXG4gICAgY29uc3RydWN0b3IodmlldzogSFRNTEVsZW1lbnQsIGFscGhhYmV0OiBzdHJpbmcpIHtcbiAgICAgICAgdGhpcy52aWV3ID0gdmlldztcbiAgICAgICAgdGhpcy5hbHBoYWJldCA9IGFscGhhYmV0O1xuICAgIH1cblxuICAgIHB1YmxpYyBpbml0KCk6IFByZXZpZXdMaW5rSGludFtdIHtcbiAgICAgICAgY29uc3QgeyB2aWV3LCBhbHBoYWJldCB9ID0gdGhpc1xuICAgICAgICBjb25zdCBsaW5rcyA9IGdldFByZXZpZXdMaW5rSGludHModmlldywgYWxwaGFiZXQpO1xuICAgICAgICBkaXNwbGF5UHJldmlld1BvcG92ZXJzKGxpbmtzKTtcbiAgICAgICAgcmV0dXJuIGxpbmtzO1xuICAgIH1cbn0iLCJpbXBvcnQge1ByZXZpZXdMaW5rSGludCwgU291cmNlTGlua0hpbnR9IGZyb20gXCIuLi8uLi90eXBlc1wiO1xuaW1wb3J0IHtFZGl0b3JWaWV3fSBmcm9tIFwiQGNvZGVtaXJyb3Ivdmlld1wiO1xuaW1wb3J0IHtkaXNwbGF5UHJldmlld1BvcG92ZXJzLCBnZXRQcmV2aWV3TGlua0hpbnRzfSBmcm9tIFwiLi4vdXRpbHMvcHJldmlld1wiO1xuXG5pbXBvcnQge2dldExpbmtIaW50TGV0dGVycywgZ2V0TURIaW50TGlua3N9IGZyb20gXCIuLi91dGlscy9jb21tb25cIjtcbmltcG9ydCB7Z2V0VmlzaWJsZVJhbmdlfSBmcm9tIFwiLi4vdXRpbHMvY202LWludGVybmFsc1wiO1xuXG5leHBvcnQgZGVmYXVsdCBjbGFzcyBMaXZlUHJldmlld0xpbmtQcm9jZXNzb3Ige1xuICAgIHZpZXc6IEhUTUxFbGVtZW50O1xuICAgIGNtRWRpdG9yOiBFZGl0b3JWaWV3O1xuICAgIGFscGhhYmV0OiBzdHJpbmc7XG5cbiAgICBjb25zdHJ1Y3Rvcih2aWV3OiBIVE1MRWxlbWVudCwgZWRpdG9yOiBFZGl0b3JWaWV3LCBhbHBoYWJldDogc3RyaW5nKSB7XG4gICAgICAgIHRoaXMudmlldyA9IHZpZXc7XG4gICAgICAgIHRoaXMuY21FZGl0b3IgPSBlZGl0b3JcbiAgICAgICAgdGhpcy5hbHBoYWJldCA9IGFscGhhYmV0O1xuICAgIH1cblxuICAgIHB1YmxpYyBpbml0KCk6IFtQcmV2aWV3TGlua0hpbnRbXSxTb3VyY2VMaW5rSGludFtdLEhUTUxFbGVtZW50W11dIHtcbiAgICAgICAgY29uc3QgeyB2aWV3LCBhbHBoYWJldCB9ID0gdGhpc1xuICAgICAgICBjb25zdCBsaW5rcyA9IGdldFByZXZpZXdMaW5rSGludHModmlldywgYWxwaGFiZXQpO1xuICAgICAgICBjb25zdCBzb3VyY2VMaW5rcyA9IHRoaXMuZ2V0U291cmNlTGlua0hpbnRzKCk7XG4gICAgICAgIGNvbnN0IGxpbmtIaW50TGV0dGVycyA9IGdldExpbmtIaW50TGV0dGVycyhhbHBoYWJldCwgbGlua3MubGVuZ3RoICsgc291cmNlTGlua3MubGVuZ3RoKTtcbiAgICAgICAgY29uc3QgbGlua3NSZW1hcHBlZCA9IGxpbmtzLm1hcCgobGluaywgaWR4KSA9PiAoey4uLmxpbmssIGxldHRlcjogbGlua0hpbnRMZXR0ZXJzW2lkeF19KSkuZmlsdGVyKGxpbmsgPT4gbGluay5sZXR0ZXIpXG4gICAgICAgIGNvbnN0IHNvdXJjZUxpbmtzUmVtYXBwZWQgPSBzb3VyY2VMaW5rcy5tYXAoKGxpbmssIGlkeCkgPT4gKHsuLi5saW5rLCBsZXR0ZXI6IGxpbmtIaW50TGV0dGVyc1tpZHggKyBsaW5rcy5sZW5ndGhdfSkpLmZpbHRlcihsaW5rID0+IGxpbmsubGV0dGVyKVxuICAgICAgICBjb25zdCBsaW5rSGludEh0bWxFbGVtZW50cyA9IGRpc3BsYXlQcmV2aWV3UG9wb3ZlcnMobGlua3NSZW1hcHBlZCk7XG4gICAgICAgIHJldHVybiBbbGlua3NSZW1hcHBlZCwgc291cmNlTGlua3NSZW1hcHBlZCwgbGlua0hpbnRIdG1sRWxlbWVudHNdO1xuICAgIH1cblxuICAgIHB1YmxpYyBnZXRWaXNpYmxlTGluZXMoKSB7XG4gICAgICAgIHJldHVybiBnZXRWaXNpYmxlUmFuZ2UodGhpcy5jbUVkaXRvcik7XG4gICAgfVxuXG4gICAgcHJpdmF0ZSBnZXRTb3VyY2VMaW5rSGludHMgPSAoKTogU291cmNlTGlua0hpbnRbXSA9PiB7XG4gICAgICAgIGNvbnN0IHsgYWxwaGFiZXQgfSA9IHRoaXM7XG4gICAgICAgIGNvbnN0IHsgaW5kZXgsIGNvbnRlbnQgfSA9IHRoaXMuZ2V0VmlzaWJsZUxpbmVzKCk7XG5cbiAgICAgICAgcmV0dXJuIGdldE1ESGludExpbmtzKGNvbnRlbnQsIGluZGV4LCBhbHBoYWJldCk7XG4gICAgfVxufSIsImltcG9ydCB7XG4gICAgQXBwLFxuICAgIE1hcmtkb3duVmlldyxcbiAgICBQbHVnaW4sXG4gICAgUGx1Z2luU2V0dGluZ1RhYixcbiAgICBTZXR0aW5nRGVmaW5pdGlvbkl0ZW0sXG4gICAgVmlldyxcbiAgICBlZGl0b3JMaXZlUHJldmlld0ZpZWxkXG59IGZyb20gJ29ic2lkaWFuJztcbmltcG9ydCB7RWRpdG9yIGFzIENNNUVkaXRvcn0gZnJvbSBcImNvZGVtaXJyb3JcIjtcbmltcG9ydCB7RWRpdG9yU2VsZWN0aW9ufSBmcm9tIFwiQGNvZGVtaXJyb3Ivc3RhdGVcIjtcbmltcG9ydCB7RWRpdG9yVmlldywgVmlld1BsdWdpbn0gZnJvbSBcIkBjb2RlbWlycm9yL3ZpZXdcIjtcbmltcG9ydCB7TGlua0hpbnRCYXNlLCBTZXR0aW5ncywgU291cmNlTGlua0hpbnR9IGZyb20gJ3R5cGVzJztcbmltcG9ydCB7TWFya1BsdWdpbn0gZnJvbSBcIi4vY202LXdpZGdldC9NYXJrUGx1Z2luXCI7XG5cbmltcG9ydCBDTTZMaW5rUHJvY2Vzc29yIGZyb20gXCIuL3Byb2Nlc3NvcnMvQ002TGlua1Byb2Nlc3NvclwiO1xuaW1wb3J0IENNNlJlZ2V4UHJvY2Vzc29yIGZyb20gXCIuL3Byb2Nlc3NvcnMvQ002UmVnZXhQcm9jZXNzb3JcIjtcbmltcG9ydCBMZWdhY3lSZWdleHBQcm9jZXNzb3IgZnJvbSBcIi4vcHJvY2Vzc29ycy9MZWdhY3lSZWdleHBQcm9jZXNzb3JcIjtcbmltcG9ydCBMZWdhY3lTb3VyY2VMaW5rUHJvY2Vzc29yIGZyb20gXCIuL3Byb2Nlc3NvcnMvTGVnYWN5U291cmNlTGlua1Byb2Nlc3NvclwiO1xuaW1wb3J0IFByZXZpZXdMaW5rUHJvY2Vzc29yIGZyb20gXCIuL3Byb2Nlc3NvcnMvUHJldmlld0xpbmtQcm9jZXNzb3JcIjtcbmltcG9ydCBMaXZlUHJldmlld0xpbmtQcm9jZXNzb3IgZnJvbSAnLi9wcm9jZXNzb3JzL0xpdmVQcmV2aWV3TGlua1Byb2Nlc3Nvcic7XG5cbmVudW0gVklFV19NT0RFIHtcbiAgICBTT1VSQ0UsXG4gICAgUFJFVklFVyxcbiAgICBMRUdBQ1ksXG4gICAgTElWRV9QUkVWSUVXXG59XG5pbnRlcmZhY2UgQ3Vyc29yU3RhdGUge1xuICAgIHZpbU1vZGU/OiBzdHJpbmc7XG4gICAgYW5jaG9yPzogbnVtYmVyO1xufVxuXG4vKiogUGFydHMgb2YgdGhlIHZpZXcvZWRpdG9yIGludGVybmFscyB3aGljaCBhcmUgbm90IGNvdmVyZWQgYnkgdGhlIHB1YmxpYyB0eXBpbmdzLiAqL1xudHlwZSBWaWV3V2l0aENNNiA9IHsgZWRpdG9yPzogeyBjbTogRWRpdG9yVmlldyB9IH07XG50eXBlIExlZ2FjeVNvdXJjZVZpZXcgPSB7IHNvdXJjZU1vZGU6IHsgY21FZGl0b3I6IENNNUVkaXRvciB9IH07XG50eXBlIExpdmVQcmV2aWV3VmlldyA9IHsgY3VycmVudE1vZGU6IHsgZWRpdG9yOiB7IGNvbnRhaW5lckVsOiBIVE1MRWxlbWVudCB9IH0gfTtcbnR5cGUgUHJldmlld1ZpZXcgPSB7IHByZXZpZXdNb2RlOiB7IGNvbnRhaW5lckVsOiBIVE1MRWxlbWVudCB9IH07XG5cbi8qKlxuICogVGhlIENvZGVNaXJyb3IgNSBpbnN0YW5jZSB0aGUgdmltIGV4dGVuc2lvbiBrZWVwcyBhcm91bmQuIEl0IGlzIHJlYWNoYWJsZVxuICogdGhyb3VnaCBgZWRpdG9yLmNtLmNtYCBvbmx5IHdoaWxlIHZpbSBtb2RlIGlzIGVuYWJsZWQuXG4gKi9cbnR5cGUgQ3Vyc29yQWN0aXZpdHlIYW5kbGVyID0gKGVkaXRvcjogQ001RWRpdG9yKSA9PiB2b2lkO1xuaW50ZXJmYWNlIFZpbUVkaXRvciB7XG4gICAgX2hhbmRsZXJzOiB7IGN1cnNvckFjdGl2aXR5OiBDdXJzb3JBY3Rpdml0eUhhbmRsZXJbXSB9O1xuICAgIG9uKGV2ZW50OiAnY3Vyc29yQWN0aXZpdHknLCBoYW5kbGVyOiBDdXJzb3JBY3Rpdml0eUhhbmRsZXIpOiB2b2lkO1xuICAgIG9mZihldmVudDogJ2N1cnNvckFjdGl2aXR5JywgaGFuZGxlcjogQ3Vyc29yQWN0aXZpdHlIYW5kbGVyKTogdm9pZDtcbn1cbnR5cGUgRWRpdG9yV2l0aFZpbSA9IHsgY20/OiB7IGNtPzogVmltRWRpdG9yIH0gfTtcbi8qKiBgc3RhdGUudmltYCBpcyBhZGRlZCBieSB0aGUgdmltIGV4dGVuc2lvbiBhbmQgaXMgbm90IHBhcnQgb2YgdGhlIENvZGVNaXJyb3IgdHlwaW5ncy4gKi9cbnR5cGUgVmltU3RhdGUgPSB7IHZpbT86IHsgbW9kZT86IHN0cmluZyB9IH07XG5cbmV4cG9ydCBkZWZhdWx0IGNsYXNzIEp1bXBUb0xpbmsgZXh0ZW5kcyBQbHVnaW4ge1xuICAgIGlzTGlua0hpbnRBY3RpdmU6IGJvb2xlYW4gPSBmYWxzZTtcbiAgICBzZXR0aW5nczogU2V0dGluZ3M7XG4gICAgcHJlZml4SW5mbzogeyBwcmVmaXg6IHN0cmluZywgc2hpZnRLZXk6IGJvb2xlYW4gfSB8IHVuZGVmaW5lZCA9IHVuZGVmaW5lZDtcbiAgICBtYXJrVmlld1BsdWdpbjogVmlld1BsdWdpbjxNYXJrUGx1Z2luPlxuICAgIGNtRWRpdG9yOiBDTTVFZGl0b3IgfCBFZGl0b3JWaWV3XG4gICAgY3VycmVudFZpZXc6IFZpZXdcbiAgICBjb250ZW50RWxlbWVudDogSFRNTEVsZW1lbnRcbiAgICBtb2RlOiBWSUVXX01PREVcbiAgICBjdXJyZW50Q3Vyc29yOiBDdXJzb3JTdGF0ZSA9IHt9O1xuICAgIGN1cnNvckJlZm9yZUp1bXA6IEN1cnNvclN0YXRlID0ge307XG5cbiAgICBhc3luYyBvbmxvYWQoKSB7XG4gICAgICAgIHRoaXMuc2V0dGluZ3MgPSBhd2FpdCB0aGlzLmxvYWREYXRhKCkgYXMgU2V0dGluZ3MgfHwgbmV3IFNldHRpbmdzKCk7XG5cbiAgICAgICAgdGhpcy5hZGRTZXR0aW5nVGFiKG5ldyBTZXR0aW5nVGFiKHRoaXMuYXBwLCB0aGlzKSk7XG5cbiAgICAgICAgY29uc3QgbWFya1ZpZXdQbHVnaW4gPSB0aGlzLm1hcmtWaWV3UGx1Z2luID0gVmlld1BsdWdpbi5mcm9tQ2xhc3MoTWFya1BsdWdpbiwge1xuICAgICAgICAgICAgZGVjb3JhdGlvbnM6ICh2OiBNYXJrUGx1Z2luKSA9PiB2LmRlY29yYXRpb25zXG4gICAgICAgIH0pO1xuICAgICAgICB0aGlzLnJlZ2lzdGVyRWRpdG9yRXh0ZW5zaW9uKFttYXJrVmlld1BsdWdpbl0pXG5cbiAgICAgICAgdGhpcy53YXRjaEZvclNlbGVjdGlvbkNoYW5nZSgpO1xuXG4gICAgICAgIHRoaXMuYWRkQ29tbWFuZCh7XG4gICAgICAgICAgICBpZDogJ2FjdGl2YXRlLWp1bXAtdG8tbGluaycsXG4gICAgICAgICAgICBuYW1lOiAnU2hvdyBsaW5rIGhpbnRzJyxcbiAgICAgICAgICAgIGNhbGxiYWNrOiAoKSA9PiB0aGlzLmFjdGlvbignbGluaycpLFxuICAgICAgICB9KTtcblxuICAgICAgICB0aGlzLmFkZENvbW1hbmQoe1xuICAgICAgICAgICAgaWQ6IFwiYWN0aXZhdGUtanVtcC10by1hbnl3aGVyZVwiLFxuICAgICAgICAgICAgbmFtZTogXCJKdW1wIHRvIGFueXdoZXJlIHJlZ2V4XCIsXG4gICAgICAgICAgICBjYWxsYmFjazogKCkgPT4gdGhpcy5hY3Rpb24oJ3JlZ2V4cCcpLFxuICAgICAgICB9KTtcblxuICAgICAgICB0aGlzLmFkZENvbW1hbmQoe1xuICAgICAgICAgICAgaWQ6IFwiYWN0aXZhdGUtbGlnaHRzcGVlZC1qdW1wXCIsXG4gICAgICAgICAgICBuYW1lOiBcIkxpZ2h0c3BlZWQganVtcFwiLFxuICAgICAgICAgICAgY2FsbGJhY2s6ICgpID0+IHRoaXMuYWN0aW9uKCdsaWdodHNwZWVkJyksXG4gICAgICAgIH0pO1xuICAgIH1cblxuICAgIGFjdGlvbih0eXBlOiAnbGluaycgfCAncmVnZXhwJyB8ICdsaWdodHNwZWVkJykge1xuICAgICAgICBpZiAodGhpcy5pc0xpbmtIaW50QWN0aXZlKSB7XG4gICAgICAgICAgICByZXR1cm47XG4gICAgICAgIH1cblxuICAgICAgICBjb25zdCBhY3RpdmVWaWV3T2ZUeXBlID0gdGhpcy5hcHAud29ya3NwYWNlLmdldEFjdGl2ZVZpZXdPZlR5cGUoTWFya2Rvd25WaWV3KVxuICAgICAgICBpZiAoIWFjdGl2ZVZpZXdPZlR5cGUpIHsgcmV0dXJuOyB9XG4gICAgICAgIGNvbnN0IGN1cnJlbnRWaWV3ID0gdGhpcy5jdXJyZW50VmlldyA9IGFjdGl2ZVZpZXdPZlR5cGUubGVhZi52aWV3O1xuICAgICAgICBjb25zdCBtb2RlID0gdGhpcy5tb2RlID0gdGhpcy5nZXRNb2RlKHRoaXMuY3VycmVudFZpZXcpO1xuICAgICAgICB0aGlzLmNvbnRlbnRFbGVtZW50ID0gYWN0aXZlVmlld09mVHlwZS5jb250ZW50RWxcbiAgICAgICAgdGhpcy5jdXJzb3JCZWZvcmVKdW1wID0gdGhpcy5jdXJyZW50Q3Vyc29yO1xuXG4gICAgICAgIHN3aXRjaCAobW9kZSkge1xuICAgICAgICAgICAgY2FzZSBWSUVXX01PREUuTEVHQUNZOlxuICAgICAgICAgICAgICAgIHRoaXMuY21FZGl0b3IgPSAoY3VycmVudFZpZXcgYXMgdW5rbm93biBhcyBMZWdhY3lTb3VyY2VWaWV3KS5zb3VyY2VNb2RlLmNtRWRpdG9yO1xuICAgICAgICAgICAgICAgIGJyZWFrO1xuICAgICAgICAgICAgY2FzZSBWSUVXX01PREUuTElWRV9QUkVWSUVXOlxuICAgICAgICAgICAgY2FzZSBWSUVXX01PREUuU09VUkNFOlxuICAgICAgICAgICAgICAgIHRoaXMuY21FZGl0b3IgPSAoPFZpZXdXaXRoQ002PmN1cnJlbnRWaWV3KS5lZGl0b3IuY207XG4gICAgICAgICAgICAgICAgYnJlYWs7XG4gICAgICAgIH1cblxuICAgICAgICBzd2l0Y2ggKHR5cGUpIHtcbiAgICAgICAgICAgIGNhc2UgXCJsaW5rXCI6XG4gICAgICAgICAgICAgICAgdGhpcy5oYW5kbGVKdW1wVG9MaW5rKCk7XG4gICAgICAgICAgICAgICAgcmV0dXJuXG4gICAgICAgICAgICBjYXNlIFwicmVnZXhwXCI6XG4gICAgICAgICAgICAgICAgdGhpcy5oYW5kbGVKdW1wVG9SZWdleCgpO1xuICAgICAgICAgICAgICAgIHJldHVyblxuICAgICAgICAgICAgY2FzZSBcImxpZ2h0c3BlZWRcIjpcbiAgICAgICAgICAgICAgICB0aGlzLmhhbmRsZUxpZ2h0c3BlZWRKdW1wKCk7XG4gICAgICAgICAgICAgICAgcmV0dXJuXG4gICAgICAgIH1cbiAgICB9XG5cbiAgICBnZXRNb2RlKGN1cnJlbnRWaWV3OiBWaWV3KTogVklFV19NT0RFIHtcbiAgICAgICAgY29uc3QgaXNMZWdhY3kgPSAodGhpcy5hcHAudmF1bHQgYXMgdW5rbm93biBhcyB7IGdldENvbmZpZyhrZXk6IHN0cmluZyk6IHVua25vd24gfSkuZ2V0Q29uZmlnKFwibGVnYWN5RWRpdG9yXCIpXG5cbiAgICAgICAgaWYgKGN1cnJlbnRWaWV3LmdldFN0YXRlKCkubW9kZSA9PT0gJ3ByZXZpZXcnKSB7XG4gICAgICAgICAgICByZXR1cm4gVklFV19NT0RFLlBSRVZJRVc7XG4gICAgICAgIH0gZWxzZSBpZiAoaXNMZWdhY3kpIHtcbiAgICAgICAgICAgIHJldHVybiBWSUVXX01PREUuTEVHQUNZO1xuICAgICAgICB9IGVsc2UgaWYgKGN1cnJlbnRWaWV3LmdldFN0YXRlKCkubW9kZSA9PT0gJ3NvdXJjZScpIHtcbiAgICAgICAgICAgIHRyeSB7XG4gICAgICAgICAgICAgICAgY29uc3QgaXNMaXZlUHJldmlldyA9ICg8Vmlld1dpdGhDTTY+Y3VycmVudFZpZXcpLmVkaXRvci5jbS5zdGF0ZT8uZmllbGQoZWRpdG9yTGl2ZVByZXZpZXdGaWVsZClcbiAgICAgICAgICAgICAgICBpZiAoaXNMaXZlUHJldmlldykgcmV0dXJuIFZJRVdfTU9ERS5MSVZFX1BSRVZJRVc7XG4gICAgICAgICAgICB9IGNhdGNoIChlKSB7XG4gICAgICAgICAgICAgICAgY29uc29sZS5lcnJvcihlKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIHJldHVybiBWSUVXX01PREUuU09VUkNFO1xuICAgICAgICB9XG5cbiAgICB9XG5cbiAgICBoYW5kbGVKdW1wVG9MaW5rID0gKCkgPT4ge1xuICAgICAgICBjb25zdCB7c2V0dGluZ3M6IHtsZXR0ZXJzfSB9ID0gdGhpc1xuXG4gICAgICAgIGNvbnN0IHsgbW9kZSwgY3VycmVudFZpZXcgfSA9IHRoaXM7XG5cbiAgICAgICAgc3dpdGNoIChtb2RlKSB7XG4gICAgICAgICAgICBjYXNlIFZJRVdfTU9ERS5MRUdBQ1k6IHtcbiAgICAgICAgICAgICAgICBjb25zdCBjbUVkaXRvciA9IHRoaXMuY21FZGl0b3IgYXMgQ001RWRpdG9yO1xuICAgICAgICAgICAgICAgIGNvbnN0IHNvdXJjZUxpbmtIaW50cyA9IG5ldyBMZWdhY3lTb3VyY2VMaW5rUHJvY2Vzc29yKGNtRWRpdG9yLCBsZXR0ZXJzKS5pbml0KCk7XG4gICAgICAgICAgICAgICAgdGhpcy5oYW5kbGVBY3Rpb25zKHNvdXJjZUxpbmtIaW50cyk7XG4gICAgICAgICAgICAgICAgYnJlYWs7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBjYXNlIFZJRVdfTU9ERS5MSVZFX1BSRVZJRVc6IHtcbiAgICAgICAgICAgICAgICBjb25zdCBjbTZFZGl0b3IgPSB0aGlzLmNtRWRpdG9yIGFzIEVkaXRvclZpZXc7XG4gICAgICAgICAgICAgICAgY29uc3QgcHJldmlld1ZpZXdFbDogSFRNTEVsZW1lbnQgPSAoY3VycmVudFZpZXcgYXMgdW5rbm93biBhcyBMaXZlUHJldmlld1ZpZXcpLmN1cnJlbnRNb2RlLmVkaXRvci5jb250YWluZXJFbDtcbiAgICAgICAgICAgICAgICBjb25zdCBbcHJldmlld0xpbmtIaW50cywgc291cmNlTGlua0hpbnRzLCBsaW5rSGludEh0bWxFbGVtZW50c10gPSBuZXcgTGl2ZVByZXZpZXdMaW5rUHJvY2Vzc29yKHByZXZpZXdWaWV3RWwsIGNtNkVkaXRvciwgbGV0dGVycykuaW5pdCgpO1xuICAgICAgICAgICAgICAgIGNtNkVkaXRvci5wbHVnaW4odGhpcy5tYXJrVmlld1BsdWdpbikuc2V0TGlua3Moc291cmNlTGlua0hpbnRzKTtcbiAgICAgICAgICAgICAgICB0aGlzLmFwcC53b3Jrc3BhY2UudXBkYXRlT3B0aW9ucygpO1xuICAgICAgICAgICAgICAgIHRoaXMuaGFuZGxlQWN0aW9ucyhbLi4ucHJldmlld0xpbmtIaW50cywgLi4uc291cmNlTGlua0hpbnRzXSwgbGlua0hpbnRIdG1sRWxlbWVudHMpO1xuICAgICAgICAgICAgICAgIGJyZWFrO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgY2FzZSBWSUVXX01PREUuUFJFVklFVzoge1xuICAgICAgICAgICAgICAgIGNvbnN0IHByZXZpZXdWaWV3RWw6IEhUTUxFbGVtZW50ID0gKGN1cnJlbnRWaWV3IGFzIHVua25vd24gYXMgUHJldmlld1ZpZXcpLnByZXZpZXdNb2RlLmNvbnRhaW5lckVsLnF1ZXJ5U2VsZWN0b3IoJ2Rpdi5tYXJrZG93bi1wcmV2aWV3LXZpZXcnKTtcbiAgICAgICAgICAgICAgICBjb25zdCBwcmV2aWV3TGlua0hpbnRzID0gbmV3IFByZXZpZXdMaW5rUHJvY2Vzc29yKHByZXZpZXdWaWV3RWwsIGxldHRlcnMpLmluaXQoKTtcbiAgICAgICAgICAgICAgICB0aGlzLmhhbmRsZUFjdGlvbnMocHJldmlld0xpbmtIaW50cyk7XG4gICAgICAgICAgICAgICAgYnJlYWs7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBjYXNlIFZJRVdfTU9ERS5TT1VSQ0U6IHtcbiAgICAgICAgICAgICAgICBjb25zdCBjbTZFZGl0b3IgPSB0aGlzLmNtRWRpdG9yIGFzIEVkaXRvclZpZXc7XG4gICAgICAgICAgICAgICAgY29uc3QgbGl2ZVByZXZpZXdMaW5rcyA9IG5ldyBDTTZMaW5rUHJvY2Vzc29yKGNtNkVkaXRvciwgbGV0dGVycykuaW5pdCgpO1xuICAgICAgICAgICAgICAgIGNtNkVkaXRvci5wbHVnaW4odGhpcy5tYXJrVmlld1BsdWdpbikuc2V0TGlua3MobGl2ZVByZXZpZXdMaW5rcyk7XG4gICAgICAgICAgICAgICAgdGhpcy5hcHAud29ya3NwYWNlLnVwZGF0ZU9wdGlvbnMoKTtcbiAgICAgICAgICAgICAgICB0aGlzLmhhbmRsZUFjdGlvbnMobGl2ZVByZXZpZXdMaW5rcyk7XG4gICAgICAgICAgICAgICAgYnJlYWs7XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICAvKlxuICAgICogIGNhc2VTZW5zaXRpdmUgaXMgb25seSBmb3IgbGlnaHRzcGVlZCBhbmQgc2hhbGwgbm90IGFmZmVjdCBqdW1wVG9Bbnl3aGVyZSwgc28gaXQgaXMgdHJ1ZVxuICAgICogIGJ5IGRlZmF1bHRcbiAgICAqL1xuICAgIGhhbmRsZUp1bXBUb1JlZ2V4ID0gKHN0cmluZ1RvU2VhcmNoPzogc3RyaW5nLCBjYXNlU2Vuc2l0aXZlOiBib29sZWFuID0gdHJ1ZSkgPT4ge1xuICAgICAgICBjb25zdCB7c2V0dGluZ3M6IHtsZXR0ZXJzLCBqdW1wVG9Bbnl3aGVyZVJlZ2V4fX0gPSB0aGlzXG4gICAgICAgIGNvbnN0IHdoYXRUb0xvb2tBdCA9IHN0cmluZ1RvU2VhcmNoIHx8IGp1bXBUb0FueXdoZXJlUmVnZXg7XG5cbiAgICAgICAgY29uc3QgeyBtb2RlIH0gPSB0aGlzXG5cbiAgICAgICAgc3dpdGNoIChtb2RlKSB7XG4gICAgICAgICAgICBjYXNlIFZJRVdfTU9ERS5TT1VSQ0U6XG4gICAgICAgICAgICAgICAgdGhpcy5oYW5kbGVNYXJrZG93blJlZ2V4KGxldHRlcnMsIHdoYXRUb0xvb2tBdCwgY2FzZVNlbnNpdGl2ZSk7XG4gICAgICAgICAgICAgICAgYnJlYWs7XG4gICAgICAgICAgICBjYXNlIFZJRVdfTU9ERS5MSVZFX1BSRVZJRVc6XG4gICAgICAgICAgICAgICAgdGhpcy5oYW5kbGVNYXJrZG93blJlZ2V4KGxldHRlcnMsIHdoYXRUb0xvb2tBdCwgY2FzZVNlbnNpdGl2ZSk7XG4gICAgICAgICAgICAgICAgYnJlYWtcbiAgICAgICAgICAgIGNhc2UgVklFV19NT0RFLlBSRVZJRVc6XG4gICAgICAgICAgICAgICAgYnJlYWs7XG4gICAgICAgICAgICBjYXNlIFZJRVdfTU9ERS5MRUdBQ1k6IHtcbiAgICAgICAgICAgICAgICBjb25zdCBjbUVkaXRvciA9IHRoaXMuY21FZGl0b3IgYXMgQ001RWRpdG9yXG4gICAgICAgICAgICAgICAgY29uc3QgbGlua3MgPSBuZXcgTGVnYWN5UmVnZXhwUHJvY2Vzc29yKGNtRWRpdG9yLCB3aGF0VG9Mb29rQXQsIGxldHRlcnMsIGNhc2VTZW5zaXRpdmUpLmluaXQoKTtcbiAgICAgICAgICAgICAgICB0aGlzLmhhbmRsZUFjdGlvbnMobGlua3MpO1xuICAgICAgICAgICAgICAgIGJyZWFrO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgZGVmYXVsdDpcbiAgICAgICAgICAgICAgICBicmVhaztcbiAgICAgICAgfVxuXG4gICAgfVxuXG4gICAgaGFuZGxlTWFya2Rvd25SZWdleCA9IChsZXR0ZXJzOiBzdHJpbmcsIHdoYXRUb0xvb2tBdDogc3RyaW5nLCBjYXNlU2Vuc2l0aXZlOiBib29sZWFuKSA9PiB7XG4gICAgICAgIGNvbnN0IGNtNkVkaXRvciA9IHRoaXMuY21FZGl0b3IgYXMgRWRpdG9yVmlld1xuICAgICAgICBjb25zdCBsaXZlUHJldmlld0xpbmtzID0gbmV3IENNNlJlZ2V4UHJvY2Vzc29yKGNtNkVkaXRvciwgbGV0dGVycywgd2hhdFRvTG9va0F0LCBjYXNlU2Vuc2l0aXZlKS5pbml0KCk7XG4gICAgICAgIGNtNkVkaXRvci5wbHVnaW4odGhpcy5tYXJrVmlld1BsdWdpbikuc2V0TGlua3MobGl2ZVByZXZpZXdMaW5rcyk7XG4gICAgICAgIHRoaXMuYXBwLndvcmtzcGFjZS51cGRhdGVPcHRpb25zKCk7XG4gICAgICAgIHRoaXMuaGFuZGxlQWN0aW9ucyhsaXZlUHJldmlld0xpbmtzKTtcbiAgICB9XG5cbiAgICAvLyBhZGFwdGVkIGZyb206IGh0dHBzOi8vZ2l0aHViLmNvbS9tcmphY2twaGlsL29ic2lkaWFuLWp1bXAtdG8tbGluay9pc3N1ZXMvMzUjaXNzdWVjb21tZW50LTEwODU5MDU2NjhcbiAgICBoYW5kbGVMaWdodHNwZWVkSnVtcCgpIHtcbiAgICAgICAgY29uc3QgYWN0aXZlVmlldyA9IHRoaXMuYXBwLndvcmtzcGFjZS5nZXRBY3RpdmVWaWV3T2ZUeXBlKE1hcmtkb3duVmlldyk7XG4gICAgICAgIGlmICghYWN0aXZlVmlldykgeyByZXR1cm4gfVxuXG4gICAgICAgIGNvbnN0IHsgY29udGVudEVsIH0gPSBhY3RpdmVWaWV3O1xuICAgICAgICBpZiAoIWNvbnRlbnRFbCkgeyByZXR1cm4gfVxuXG4gICAgICAgIC8vIHRoaXMgZWxlbWVudCBkb2Vzbid0IGV4aXN0IGluIGNtNS9oYXMgYSBkaWZmZXJlbnQgY2xhc3MsIHNvIGxpZ2h0c3BlZWQgd2lsbCBub3Qgd29yayBpbiBjbTVcbiAgICAgICAgY29uc3QgY29udGVudENvbnRhaW5lciA9IGNvbnRlbnRFbC5nZXRFbGVtZW50c0J5Q2xhc3NOYW1lKFwiY20tY29udGVudENvbnRhaW5lclwiKVswXTtcbiAgICAgICAgaWYgKCFjb250ZW50Q29udGFpbmVyKSB7IHJldHVybiB9XG5cbiAgICAgICAgLy8gZGltIGFsbCB0aGUgdGV4dCB3aGlsZSB3YWl0aW5nIGZvciB0aGUgY2hhcmFjdGVycyB0byBqdW1wIHRvXG4gICAgICAgIGNvbnRlbnRDb250YWluZXIuY2xhc3NMaXN0LmFkZCgnamwtbGlnaHRzcGVlZCcpO1xuICAgICAgICBjb25zdCByZXN0b3JlVGV4dENvbG9yID0gKCkgPT4gY29udGVudENvbnRhaW5lci5jbGFzc0xpc3QucmVtb3ZlKCdqbC1saWdodHNwZWVkJyk7XG5cbiAgICAgICAgY29uc3Qga2V5QXJyYXk6IHN0cmluZ1tdID0gW107XG4gICAgICAgIGNvbnN0IGdyYWJLZXkgPSAoZXZlbnQ6IEtleWJvYXJkRXZlbnQpID0+IHtcbiAgICAgICAgICAgIGV2ZW50LnByZXZlbnREZWZhdWx0KCk7XG5cbiAgICAgICAgICAgIC8vIGhhbmRsZSBFc2NhcGUgdG8gcmVqZWN0IHRoZSBtb2RlXG4gICAgICAgICAgICBpZiAoZXZlbnQua2V5ID09PSAnRXNjYXBlJykge1xuICAgICAgICAgICAgICAgIGNvbnRlbnRFbC5yZW1vdmVFdmVudExpc3RlbmVyKFwia2V5ZG93blwiLCBncmFiS2V5LCB7IGNhcHR1cmU6IHRydWUgfSk7XG4gICAgICAgICAgICAgICAgcmVzdG9yZVRleHRDb2xvcigpO1xuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICAvLyB0ZXN0IGlmIGtleXByZXNzIGlzIGNhcGl0YWxpemVkXG4gICAgICAgICAgICBpZiAoL15bXFx3XFxTXFxXXSQvaS50ZXN0KGV2ZW50LmtleSkpIHtcbiAgICAgICAgICAgICAgICBjb25zdCBpc0NhcGl0YWwgPSBldmVudC5zaGlmdEtleTtcbiAgICAgICAgICAgICAgICBpZiAoaXNDYXBpdGFsKSB7XG4gICAgICAgICAgICAgICAgICAgIC8vIGNhcHR1cmUgdXBwZXJjYXNlXG4gICAgICAgICAgICAgICAgICAgIGtleUFycmF5LnB1c2goKGV2ZW50LmtleSkudG9VcHBlckNhc2UoKSk7XG4gICAgICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICAgICAgLy8gY2FwdHVyZSBsb3dlcmNhc2VcbiAgICAgICAgICAgICAgICAgICAga2V5QXJyYXkucHVzaChldmVudC5rZXkpO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgLy8gc3RvcCB3aGVuIGxlbmd0aCBvZiBhcnJheSBpcyBlcXVhbCB0byBsaWdodHNwZWVkQ2hhcmFjdGVyQ291bnRcbiAgICAgICAgICAgIGlmIChrZXlBcnJheS5sZW5ndGggPT09IHRoaXMuc2V0dGluZ3MubGlnaHRzcGVlZENoYXJhY3RlckNvdW50KSB7XG4gICAgICAgICAgICAgICAgY29uc3Qgc3RyaW5nVG9TZWFyY2ggPSB0aGlzLnNldHRpbmdzLmxpZ2h0c3BlZWRKdW1wVG9TdGFydE9mV29yZCA/IFwiXFxcXGJcIiArIGtleUFycmF5LmpvaW4oXCJcIikgOiBrZXlBcnJheS5qb2luKFwiXCIpO1xuXG4gICAgICAgICAgICAgICAgdGhpcy5oYW5kbGVKdW1wVG9SZWdleChzdHJpbmdUb1NlYXJjaCwgdGhpcy5zZXR0aW5ncy5saWdodHNwZWVkQ2FzZVNlbnNpdGl2ZSk7XG5cbiAgICAgICAgICAgICAgICAvLyByZW1vdmluZyBldmVudExpc3RlbmVyIGFmdGVyIHByb2NlZWRlZFxuICAgICAgICAgICAgICAgIGNvbnRlbnRFbC5yZW1vdmVFdmVudExpc3RlbmVyKFwia2V5ZG93blwiLCBncmFiS2V5LCB7IGNhcHR1cmU6IHRydWUgfSk7XG4gICAgICAgICAgICAgICAgcmVzdG9yZVRleHRDb2xvcigpO1xuICAgICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIGNvbnRlbnRFbC5hZGRFdmVudExpc3RlbmVyKCdrZXlkb3duJywgZ3JhYktleSwgeyBjYXB0dXJlOiB0cnVlIH0pO1xuICAgIH1cblxuICAgIGhhbmRsZUhvdGtleShoZWxkU2hpZnRLZXk6IGJvb2xlYW4sIGxpbms6IFNvdXJjZUxpbmtIaW50IHwgTGlua0hpbnRCYXNlKSB7XG4gICAgICAgIGlmIChsaW5rLmxpbmtFbGVtZW50KSB7XG4gICAgICAgICAgICBjb25zdCBldmVudCA9IG5ldyBNb3VzZUV2ZW50KFwiY2xpY2tcIiwge1xuICAgICAgICAgICAgICAgIGJ1YmJsZXM6IHRydWUsXG4gICAgICAgICAgICAgICAgY2FuY2VsYWJsZTogdHJ1ZSxcbiAgICAgICAgICAgICAgICB2aWV3OiB3aW5kb3csXG4gICAgICAgICAgICAgICAgbWV0YUtleTogaGVsZFNoaWZ0S2V5LFxuICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICBsaW5rLmxpbmtFbGVtZW50LmRpc3BhdGNoRXZlbnQoZXZlbnQpO1xuICAgICAgICB9IGVsc2UgaWYgKGxpbmsudHlwZSA9PT0gJ2ludGVybmFsJykge1xuICAgICAgICAgICAgY29uc3QgZmlsZSA9IHRoaXMuYXBwLndvcmtzcGFjZS5nZXRBY3RpdmVGaWxlKClcbiAgICAgICAgICAgIGlmIChmaWxlKSB7XG4gICAgICAgICAgICAgICAgLy8gdGhlIHNlY29uZCBhcmd1bWVudCBpcyBmb3IgdGhlIGxpbmsgcmVzb2x1dGlvblxuICAgICAgICAgICAgICAgIHZvaWQgdGhpcy5hcHAud29ya3NwYWNlLm9wZW5MaW5rVGV4dChkZWNvZGVVUkkobGluay5saW5rVGV4dCksIGZpbGUucGF0aCwgaGVsZFNoaWZ0S2V5LCB7YWN0aXZlOiB0cnVlfSk7XG4gICAgICAgICAgICB9XG4gICAgICAgIH0gZWxzZSBpZiAobGluay50eXBlID09PSAnZXh0ZXJuYWwnKSB7XG4gICAgICAgICAgICB3aW5kb3cub3BlbihsaW5rLmxpbmtUZXh0KTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIGNvbnN0IGVkaXRvciA9IHRoaXMuY21FZGl0b3I7XG4gICAgICAgICAgICBpZiAoZWRpdG9yIGluc3RhbmNlb2YgRWRpdG9yVmlldykge1xuICAgICAgICAgICAgICAgIGNvbnN0IGluZGV4ID0gKGxpbmsgYXMgU291cmNlTGlua0hpbnQpLmluZGV4O1xuICAgICAgICAgICAgICAgIGNvbnN0IHt2aW1Nb2RlLCBhbmNob3J9ID0gdGhpcy5jdXJzb3JCZWZvcmVKdW1wO1xuICAgICAgICAgICAgICAgIGNvbnN0IHVzZVNlbGVjdGlvbiA9IGhlbGRTaGlmdEtleSB8fCAodmltTW9kZSA9PT0gJ3Zpc3VhbCcgfHwgdmltTW9kZSA9PT0gJ3Zpc3VhbCBibG9jaycpXG5cbiAgICAgICAgICAgICAgICBpZiAodXNlU2VsZWN0aW9uICYmIGFuY2hvciAhPT0gdW5kZWZpbmVkKSB7XG4gICAgICAgICAgICAgICAgICAgIGVkaXRvci5kaXNwYXRjaCh7c2VsZWN0aW9uOiBFZGl0b3JTZWxlY3Rpb24ucmFuZ2UoYW5jaG9yLCBpbmRleCl9KVxuICAgICAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgICAgIGVkaXRvci5kaXNwYXRjaCh7IHNlbGVjdGlvbjogRWRpdG9yU2VsZWN0aW9uLmN1cnNvcihpbmRleCkgfSlcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgIGVkaXRvci5zZXRDdXJzb3IoZWRpdG9yLnBvc0Zyb21JbmRleCgoPFNvdXJjZUxpbmtIaW50PmxpbmspLmluZGV4KSk7XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICByZW1vdmVQb3BvdmVycyhsaW5rSGludEh0bWxFbGVtZW50czogSFRNTEVsZW1lbnRbXSB8IHVuZGVmaW5lZCA9IFtdKSB7XG4gICAgICAgIGNvbnN0IGN1cnJlbnRWaWV3ID0gdGhpcy5jb250ZW50RWxlbWVudDtcblxuICAgICAgICBjdXJyZW50Vmlldy5yZW1vdmVFdmVudExpc3RlbmVyKCdjbGljaycsICgpID0+IHRoaXMucmVtb3ZlUG9wb3ZlcnMobGlua0hpbnRIdG1sRWxlbWVudHMpKVxuICAgICAgICBsaW5rSGludEh0bWxFbGVtZW50cz8uZm9yRWFjaChlID0+IGUucmVtb3ZlKCkpO1xuICAgICAgICBjdXJyZW50Vmlldy5xdWVyeVNlbGVjdG9yQWxsKCcuamwucG9wb3ZlcicpLmZvckVhY2goZSA9PiBlLnJlbW92ZSgpKTtcbiAgICAgICAgY3VycmVudFZpZXcucXVlcnlTZWxlY3RvckFsbCgnLmpsLWFuY2hvcicpLmZvckVhY2goZSA9PiBlLmNsYXNzTGlzdC5yZW1vdmUoJ2psLWFuY2hvcicpKTtcblxuICAgICAgICB0aGlzLnByZWZpeEluZm8gPSB1bmRlZmluZWQ7XG4gICAgICAgIGlmICh0aGlzLm1vZGUgPT0gVklFV19NT0RFLlNPVVJDRSB8fCB0aGlzLm1vZGUgPT0gVklFV19NT0RFLkxJVkVfUFJFVklFVykge1xuICAgICAgICAgICAgKHRoaXMuY21FZGl0b3IgYXMgRWRpdG9yVmlldykucGx1Z2luKHRoaXMubWFya1ZpZXdQbHVnaW4pLmNsZWFuKCk7XG4gICAgICAgIH1cbiAgICAgICAgdGhpcy5hcHAud29ya3NwYWNlLnVwZGF0ZU9wdGlvbnMoKTtcbiAgICAgICAgdGhpcy5pc0xpbmtIaW50QWN0aXZlID0gZmFsc2U7XG4gICAgfVxuXG4gICAgcmVtb3ZlUG9wb3ZlcnNXaXRob3V0UHJlZml4RXZlbnRLZXkoZXZlbnRLZXk6IHN0cmluZywgbGlua0hpbnRIdG1sRWxlbWVudHM6IEhUTUxFbGVtZW50W10gfCB1bmRlZmluZWQgPSBbXSkge1xuICAgICAgICBjb25zdCBjdXJyZW50VmlldyA9IHRoaXMuY29udGVudEVsZW1lbnQ7XG5cbiAgICAgICAgbGlua0hpbnRIdG1sRWxlbWVudHM/LmZvckVhY2goZSA9PiB7XG4gICAgICAgICAgICBpZiAoZS50ZXh0Q29udGVudC5sZW5ndGggPT0gMiAmJiBlLnRleHRDb250ZW50WzBdID09IGV2ZW50S2V5KSB7XG4gICAgICAgICAgICAgICAgZS5jbGFzc0xpc3QuYWRkKFwibWF0Y2hlZFwiKTtcbiAgICAgICAgICAgICAgICByZXR1cm47XG4gICAgICAgICAgICB9XG5cbiAgICAgICAgICAgIGUucmVtb3ZlKCk7XG4gICAgICAgIH0pO1xuXG4gICAgICAgIGN1cnJlbnRWaWV3LnF1ZXJ5U2VsZWN0b3JBbGwoJy5qbC5wb3BvdmVyJykuZm9yRWFjaChlID0+IHtcbiAgICAgICAgICAgIGlmIChlLnRleHRDb250ZW50Lmxlbmd0aCA9PSAyICYmIGUudGV4dENvbnRlbnRbMF0gPT0gZXZlbnRLZXkpIHtcbiAgICAgICAgICAgICAgICBlLmNsYXNzTGlzdC5hZGQoXCJtYXRjaGVkXCIpO1xuICAgICAgICAgICAgICAgIHJldHVybjtcbiAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgZS5yZW1vdmUoKTtcbiAgICAgICAgfSk7XG5cbiAgICAgICAgaWYgKHRoaXMubW9kZSA9PSBWSUVXX01PREUuU09VUkNFIHx8IHRoaXMubW9kZSA9PSBWSUVXX01PREUuTElWRV9QUkVWSUVXKSB7XG4gICAgICAgICAgICAodGhpcy5jbUVkaXRvciBhcyBFZGl0b3JWaWV3KS5wbHVnaW4odGhpcy5tYXJrVmlld1BsdWdpbikuZmlsdGVyV2l0aEV2ZW50S2V5KGV2ZW50S2V5KTtcbiAgICAgICAgfVxuICAgICAgICB0aGlzLmFwcC53b3Jrc3BhY2UudXBkYXRlT3B0aW9ucygpO1xuICAgIH1cblxuICAgIGhhbmRsZUFjdGlvbnMobGlua0hpbnRzOiBMaW5rSGludEJhc2VbXSwgbGlua0hpbnRIdG1sRWxlbWVudHM/OiBIVE1MRWxlbWVudFtdKTogdm9pZCB7XG4gICAgICAgIGNvbnN0IGNvbnRlbnRFbGVtZW50ID0gdGhpcy5jb250ZW50RWxlbWVudFxuICAgICAgICBpZiAoIWxpbmtIaW50cy5sZW5ndGgpIHtcbiAgICAgICAgICAgIHJldHVybjtcbiAgICAgICAgfVxuXG4gICAgICAgIGNvbnN0IGxpbmtIaW50TWFwOiB7IFtsZXR0ZXI6IHN0cmluZ106IExpbmtIaW50QmFzZSB9ID0ge307XG4gICAgICAgIGxpbmtIaW50cy5mb3JFYWNoKHggPT4gbGlua0hpbnRNYXBbeC5sZXR0ZXJdID0geCk7XG5cbiAgICAgICAgY29uc3QgaGFuZGxlS2V5RG93biA9IChldmVudDogS2V5Ym9hcmRFdmVudCk6IHZvaWQgPT4ge1xuICAgICAgICAgICAgaWYgKFsnU2hpZnQnLCAnQ29udHJvbCcsICdDYXBzTG9jaycsICdTY3JvbGxMb2NrJywgJ0dyb3VwTmV4dCcsICdNZXRhJ10uaW5jbHVkZXMoZXZlbnQua2V5KSkge1xuICAgICAgICAgICAgICAgIHJldHVybjtcbiAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgY29uc3QgZXZlbnRLZXkgPSBldmVudC5rZXkudG9VcHBlckNhc2UoKTtcbiAgICAgICAgICAgIGNvbnN0IHByZWZpeGVzID0gbmV3IFNldChPYmplY3Qua2V5cyhsaW5rSGludE1hcCkuZmlsdGVyKHggPT4geC5sZW5ndGggPiAxKS5tYXAoeCA9PiB4WzBdKSk7XG5cbiAgICAgICAgICAgIGxldCBsaW5rSGludDogTGlua0hpbnRCYXNlO1xuICAgICAgICAgICAgaWYgKHRoaXMucHJlZml4SW5mbykge1xuICAgICAgICAgICAgICAgIGxpbmtIaW50ID0gbGlua0hpbnRNYXBbdGhpcy5wcmVmaXhJbmZvLnByZWZpeCArIGV2ZW50S2V5XTtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgbGlua0hpbnQgPSBsaW5rSGludE1hcFtldmVudEtleV07XG4gICAgICAgICAgICAgICAgaWYgKCFsaW5rSGludCAmJiBwcmVmaXhlcyAmJiBwcmVmaXhlcy5oYXMoZXZlbnRLZXkpKSB7XG4gICAgICAgICAgICAgICAgICAgIHRoaXMucHJlZml4SW5mbyA9IHtwcmVmaXg6IGV2ZW50S2V5LCBzaGlmdEtleTogZXZlbnQuc2hpZnRLZXl9O1xuXG4gICAgICAgICAgICAgICAgICAgIGV2ZW50LnByZXZlbnREZWZhdWx0KCk7XG4gICAgICAgICAgICAgICAgICAgIGV2ZW50LnN0b3BQcm9wYWdhdGlvbigpO1xuICAgICAgICAgICAgICAgICAgICBldmVudC5zdG9wSW1tZWRpYXRlUHJvcGFnYXRpb24oKTtcblxuICAgICAgICAgICAgICAgICAgICB0aGlzLnJlbW92ZVBvcG92ZXJzV2l0aG91dFByZWZpeEV2ZW50S2V5KGV2ZW50S2V5LCBsaW5rSGludEh0bWxFbGVtZW50cyk7XG5cbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgZXZlbnQucHJldmVudERlZmF1bHQoKTtcbiAgICAgICAgICAgIGV2ZW50LnN0b3BQcm9wYWdhdGlvbigpO1xuICAgICAgICAgICAgZXZlbnQuc3RvcEltbWVkaWF0ZVByb3BhZ2F0aW9uKCk7XG5cbiAgICAgICAgICAgIGNvbnN0IGhlbGRTaGlmdEtleSA9IHRoaXMucHJlZml4SW5mbz8uc2hpZnRLZXkgfHwgZXZlbnQuc2hpZnRLZXk7XG5cbiAgICAgICAgICAgIGlmIChsaW5rSGludCkge1xuICAgICAgICAgICAgICAgIHRoaXMuaGFuZGxlSG90a2V5KGhlbGRTaGlmdEtleSwgbGlua0hpbnQpO1xuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICB0aGlzLnJlbW92ZVBvcG92ZXJzKGxpbmtIaW50SHRtbEVsZW1lbnRzKTtcbiAgICAgICAgICAgIGNvbnRlbnRFbGVtZW50LnJlbW92ZUV2ZW50TGlzdGVuZXIoJ2tleWRvd24nLCBoYW5kbGVLZXlEb3duLCB7IGNhcHR1cmU6IHRydWUgfSk7XG4gICAgICAgIH07XG5cbiAgICAgICAgaWYgKGxpbmtIaW50cy5sZW5ndGggPT09IDEgJiYgdGhpcy5zZXR0aW5ncy5qdW1wVG9MaW5rSWZPbmVMaW5rT25seSkge1xuICAgICAgICAgICAgY29uc3QgaGVsZFNoaWZ0S2V5ID0gdGhpcy5wcmVmaXhJbmZvPy5zaGlmdEtleTtcbiAgICAgICAgICAgIHRoaXMuaGFuZGxlSG90a2V5KGhlbGRTaGlmdEtleSwgbGlua0hpbnRzWzBdKTtcbiAgICAgICAgICAgIHRoaXMucmVtb3ZlUG9wb3ZlcnMobGlua0hpbnRIdG1sRWxlbWVudHMpO1xuICAgICAgICAgICAgcmV0dXJuXG4gICAgICAgIH1cblxuICAgICAgICBjb250ZW50RWxlbWVudC5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsICgpID0+IHRoaXMucmVtb3ZlUG9wb3ZlcnMobGlua0hpbnRIdG1sRWxlbWVudHMpKVxuICAgICAgICBjb250ZW50RWxlbWVudC5hZGRFdmVudExpc3RlbmVyKCdrZXlkb3duJywgaGFuZGxlS2V5RG93biwgeyBjYXB0dXJlOiB0cnVlIH0pO1xuICAgICAgICB0aGlzLmlzTGlua0hpbnRBY3RpdmUgPSB0cnVlO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIENvZGVNaXJyb3IncyB2aW0gYXV0b21hdGljYWxseSBleGl0cyB2aXN1YWwgbW9kZSB3aGVuIGV4ZWN1dGluZyBhIGNvbW1hbmQuXG4gICAgICogVGhpcyBrZWVwcyB0cmFjayBvZiBzZWxlY3Rpb24gY2hhbmdlcyBzbyB3ZSBjYW4gcmVzdG9yZSB0aGUgc2VsZWN0aW9uLlxuICAgICAqXG4gICAgICogVGhpcyBpcyB0aGUgc2FtZSBhcHByb2FjaCB0YWtlbiBieSB0aGUgb2JzaWRpYW4tdmltcmMtcGx1Z2luXG4gICAgICovXG4gICAgd2F0Y2hGb3JTZWxlY3Rpb25DaGFuZ2UoKSB7XG4gICAgICAgIGNvbnN0IHVwZGF0ZVNlbGVjdGlvbjogQ3Vyc29yQWN0aXZpdHlIYW5kbGVyID0gKGVkaXRvcikgPT4gdGhpcy51cGRhdGVTZWxlY3Rpb24oZWRpdG9yKTtcbiAgICAgICAgY29uc3Qgd2F0Y2hGb3JDaGFuZ2VzID0gKCkgPT4ge1xuICAgICAgICAgICAgY29uc3QgZWRpdG9yID0gdGhpcy5hcHAud29ya3NwYWNlLmdldEFjdGl2ZVZpZXdPZlR5cGUoTWFya2Rvd25WaWV3KT8uZWRpdG9yO1xuICAgICAgICAgICAgY29uc3QgY206IFZpbUVkaXRvciB8IHVuZGVmaW5lZCA9IChlZGl0b3IgYXMgdW5rbm93biBhcyBFZGl0b3JXaXRoVmltKT8uY20/LmNtO1xuXG4gICAgICAgICAgICBpZiAoY20gJiYgIWNtLl9oYW5kbGVycy5jdXJzb3JBY3Rpdml0eS5pbmNsdWRlcyh1cGRhdGVTZWxlY3Rpb24pKSB7XG4gICAgICAgICAgICAgICAgY20ub24oXCJjdXJzb3JBY3Rpdml0eVwiLCB1cGRhdGVTZWxlY3Rpb24pO1xuICAgICAgICAgICAgICAgIHRoaXMucmVnaXN0ZXIoKCkgPT4gY20ub2ZmKFwiY3Vyc29yQWN0aXZpdHlcIiwgdXBkYXRlU2VsZWN0aW9uKSk7XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgdGhpcy5yZWdpc3RlckV2ZW50KHRoaXMuYXBwLndvcmtzcGFjZS5vbihcImFjdGl2ZS1sZWFmLWNoYW5nZVwiLCB3YXRjaEZvckNoYW5nZXMpKTtcbiAgICAgICAgdGhpcy5yZWdpc3RlckV2ZW50KHRoaXMuYXBwLndvcmtzcGFjZS5vbihcImZpbGUtb3BlblwiLCB3YXRjaEZvckNoYW5nZXMpKTtcbiAgICAgICAgd2F0Y2hGb3JDaGFuZ2VzKCk7XG4gICAgfVxuXG4gICAgdXBkYXRlU2VsZWN0aW9uKGVkaXRvcjogQ001RWRpdG9yKSB7XG4gICAgICAgIGNvbnN0IGFuY2hvciA9IGVkaXRvci5saXN0U2VsZWN0aW9ucygpWzBdPy5hbmNob3JcbiAgICAgICAgdGhpcy5jdXJyZW50Q3Vyc29yID0ge1xuICAgICAgICAgICAgYW5jaG9yOiBhbmNob3IgPyBlZGl0b3IuaW5kZXhGcm9tUG9zKGFuY2hvcikgOiB1bmRlZmluZWQsXG4gICAgICAgICAgICB2aW1Nb2RlOiAoZWRpdG9yLnN0YXRlIGFzIFZpbVN0YXRlKS52aW0/Lm1vZGVcbiAgICAgICAgfVxuICAgIH1cbn1cblxuY2xhc3MgU2V0dGluZ1RhYiBleHRlbmRzIFBsdWdpblNldHRpbmdUYWIge1xuICAgIHBsdWdpbjogSnVtcFRvTGlua1xuXG4gICAgY29uc3RydWN0b3IoYXBwOiBBcHAsIHBsdWdpbjogSnVtcFRvTGluaykge1xuICAgICAgICBzdXBlcihhcHAsIHBsdWdpbilcblxuICAgICAgICB0aGlzLnBsdWdpbiA9IHBsdWdpblxuICAgIH1cblxuICAgIC8qKlxuICAgICAqIFRoZSBkZWNsYXJhdGl2ZSBzZXR0aW5ncyBBUEkgKE9ic2lkaWFuIDEuMTMrKS4gT2JzaWRpYW4gcmVuZGVycyB0aGUgdGFiXG4gICAgICogZnJvbSB0aGVzZSBkZWZpbml0aW9ucyBhbmQgaW5kZXhlcyBldmVyeSBzZXR0aW5nIGZvciB0aGUgc2V0dGluZ3Mgc2VhcmNoLFxuICAgICAqIHdoaWNoIGlzIHdoeSB0aGVyZSBpcyBubyBgZGlzcGxheSgpYC5cbiAgICAgKi9cbiAgICBnZXRTZXR0aW5nRGVmaW5pdGlvbnMoKTogU2V0dGluZ0RlZmluaXRpb25JdGVtPGtleW9mIFNldHRpbmdzPltdIHtcbiAgICAgICAgY29uc3QgZGVmYXVsdHMgPSBuZXcgU2V0dGluZ3MoKTtcblxuICAgICAgICByZXR1cm4gW1xuICAgICAgICAgICAge1xuICAgICAgICAgICAgICAgIG5hbWU6ICdDaGFyYWN0ZXJzIHVzZWQgZm9yIGxpbmsgaGludHMnLFxuICAgICAgICAgICAgICAgIGRlc2M6ICdUaGUgY2hhcmFjdGVycyBwbGFjZWQgbmV4dCB0byBlYWNoIGxpbmsgYWZ0ZXIgZW50ZXIgbGluay1oaW50IG1vZGUuJyxcbiAgICAgICAgICAgICAgICBjb250cm9sOiB7dHlwZTogJ3RleHQnLCBrZXk6ICdsZXR0ZXJzJywgZGVmYXVsdFZhbHVlOiBkZWZhdWx0cy5sZXR0ZXJzfVxuICAgICAgICAgICAgfSxcbiAgICAgICAgICAgIHtcbiAgICAgICAgICAgICAgICBuYW1lOiAnSnVtcCB0byBhbnl3aGVyZScsXG4gICAgICAgICAgICAgICAgZGVzYzogJ1JlZ2V4IGJhc2VkIG5hdmlnYXRpbmcgaW4gZWRpdG9yIG1vZGUnLFxuICAgICAgICAgICAgICAgIGNvbnRyb2w6IHtcbiAgICAgICAgICAgICAgICAgICAgdHlwZTogJ3RleHQnLFxuICAgICAgICAgICAgICAgICAgICBrZXk6ICdqdW1wVG9Bbnl3aGVyZVJlZ2V4JyxcbiAgICAgICAgICAgICAgICAgICAgcGxhY2Vob2xkZXI6ICdDdXN0b20gcmVnZXgnLFxuICAgICAgICAgICAgICAgICAgICBkZWZhdWx0VmFsdWU6IGRlZmF1bHRzLmp1bXBUb0FueXdoZXJlUmVnZXhcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9LFxuICAgICAgICAgICAge1xuICAgICAgICAgICAgICAgIG5hbWU6ICdMaWdodHNwZWVkIHJlZ2V4IGNhc2Ugc2Vuc2l0aXZpdHknLFxuICAgICAgICAgICAgICAgIGRlc2M6ICdJZiBlbmFibGVkLCB0aGUgcmVnZXggZm9yIG1hdGNoaW5nIHdpbGwgYmUgY2FzZSBzZW5zaXRpdmUuJyxcbiAgICAgICAgICAgICAgICBjb250cm9sOiB7XG4gICAgICAgICAgICAgICAgICAgIHR5cGU6ICd0b2dnbGUnLFxuICAgICAgICAgICAgICAgICAgICBrZXk6ICdsaWdodHNwZWVkQ2FzZVNlbnNpdGl2ZScsXG4gICAgICAgICAgICAgICAgICAgIGRlZmF1bHRWYWx1ZTogZGVmYXVsdHMubGlnaHRzcGVlZENhc2VTZW5zaXRpdmVcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9LFxuICAgICAgICAgICAge1xuICAgICAgICAgICAgICAgIG5hbWU6ICdKdW1wIHRvIGxpbmsgaWYgb25seSBvbmUgbGluayBpbiBwYWdlJyxcbiAgICAgICAgICAgICAgICBkZXNjOiAnSWYgZW5hYmxlZCwgYXV0byBqdW1wIHRvIGxpbmsgaWYgdGhlcmUgaXMgb25seSBvbmUgbGluayBpbiBwYWdlJyxcbiAgICAgICAgICAgICAgICBjb250cm9sOiB7XG4gICAgICAgICAgICAgICAgICAgIHR5cGU6ICd0b2dnbGUnLFxuICAgICAgICAgICAgICAgICAgICBrZXk6ICdqdW1wVG9MaW5rSWZPbmVMaW5rT25seScsXG4gICAgICAgICAgICAgICAgICAgIGRlZmF1bHRWYWx1ZTogZGVmYXVsdHMuanVtcFRvTGlua0lmT25lTGlua09ubHlcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9LFxuICAgICAgICAgICAge1xuICAgICAgICAgICAgICAgIG5hbWU6ICdMaWdodHNwZWVkIG9ubHkganVtcHMgdG8gc3RhcnQgb2Ygd29yZHMnLFxuICAgICAgICAgICAgICAgIGRlc2M6ICdJZiBlbmFibGVkLCBsaWdodHNwZWVkIGp1bXBzIHdpbGwgb25seSB0YXJnZXQgY2hhcmFjdGVycyBvY2N1cmluZyBhdCB0aGUgc3RhcnQgb2Ygd29yZHMuJyxcbiAgICAgICAgICAgICAgICBjb250cm9sOiB7XG4gICAgICAgICAgICAgICAgICAgIHR5cGU6ICd0b2dnbGUnLFxuICAgICAgICAgICAgICAgICAgICBrZXk6ICdsaWdodHNwZWVkSnVtcFRvU3RhcnRPZldvcmQnLFxuICAgICAgICAgICAgICAgICAgICBkZWZhdWx0VmFsdWU6IGRlZmF1bHRzLmxpZ2h0c3BlZWRKdW1wVG9TdGFydE9mV29yZFxuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH0sXG4gICAgICAgICAgICB7XG4gICAgICAgICAgICAgICAgbmFtZTogJ051bWJlciBvZiBjaGFyYWN0ZXJzIGZvciBsaWdodHNwZWVkIGp1bXAnLFxuICAgICAgICAgICAgICAgIGRlc2M6ICdEZXRlcm1pbmVzIGhvdyBtYW55IGNoYXJhY3RlcnMgeW91IG5lZWQgdG8gdHlwZSB0byBwZXJmb3JtIGEgbGlnaHRzcGVlZCBqdW1wLicsXG4gICAgICAgICAgICAgICAgY29udHJvbDoge1xuICAgICAgICAgICAgICAgICAgICB0eXBlOiAnbnVtYmVyJyxcbiAgICAgICAgICAgICAgICAgICAga2V5OiAnbGlnaHRzcGVlZENoYXJhY3RlckNvdW50JyxcbiAgICAgICAgICAgICAgICAgICAgbWluOiAxLFxuICAgICAgICAgICAgICAgICAgICBtYXg6IDUsXG4gICAgICAgICAgICAgICAgICAgIGRlZmF1bHRWYWx1ZTogZGVmYXVsdHMubGlnaHRzcGVlZENoYXJhY3RlckNvdW50XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICBdO1xuICAgIH1cblxuICAgIGdldENvbnRyb2xWYWx1ZShrZXk6IHN0cmluZyk6IHVua25vd24ge1xuICAgICAgICByZXR1cm4gKHRoaXMucGx1Z2luLnNldHRpbmdzIGFzIHVua25vd24gYXMgUmVjb3JkPHN0cmluZywgdW5rbm93bj4pW2tleV07XG4gICAgfVxuXG4gICAgLyoqIFBlcnNpc3RzIHRocm91Z2ggYHNhdmVEYXRhYCwgdGhlIHN0b3JlIHRoZSBzZXR0aW5ncyBhcmUgbG9hZGVkIGZyb20uICovXG4gICAgc2V0Q29udHJvbFZhbHVlKGtleTogc3RyaW5nLCB2YWx1ZTogdW5rbm93bik6IFByb21pc2U8dm9pZD4ge1xuICAgICAgICAodGhpcy5wbHVnaW4uc2V0dGluZ3MgYXMgdW5rbm93biBhcyBSZWNvcmQ8c3RyaW5nLCB1bmtub3duPilba2V5XSA9IHZhbHVlO1xuXG4gICAgICAgIHJldHVybiB0aGlzLnBsdWdpbi5zYXZlRGF0YSh0aGlzLnBsdWdpbi5zZXR0aW5ncyk7XG4gICAgfVxufVxuIl0sIm5hbWVzIjpbIldpZGdldFR5cGUiLCJEZWNvcmF0aW9uIiwiUGx1Z2luIiwiVmlld1BsdWdpbiIsIk1hcmtkb3duVmlldyIsImVkaXRvckxpdmVQcmV2aWV3RmllbGQiLCJFZGl0b3JWaWV3IiwiRWRpdG9yU2VsZWN0aW9uIiwiUGx1Z2luU2V0dGluZ1RhYiJdLCJtYXBwaW5ncyI6Ijs7Ozs7O01BbUJhLFFBQVEsQ0FBQTtBQUFyQixJQUFBLFdBQUEsR0FBQTs7UUFFQyxJQUFPLENBQUEsT0FBQSxHQUFXLGdCQUFnQixDQUFDO1FBQ25DLElBQW1CLENBQUEsbUJBQUEsR0FBVyxlQUFlLENBQUM7UUFDOUMsSUFBdUIsQ0FBQSx1QkFBQSxHQUFZLEtBQUssQ0FBQztRQUN6QyxJQUF1QixDQUFBLHVCQUFBLEdBQVksSUFBSSxDQUFDO1FBQ3hDLElBQTJCLENBQUEsMkJBQUEsR0FBWSxJQUFJLENBQUM7UUFDNUMsSUFBd0IsQ0FBQSx3QkFBQSxHQUFXLENBQUMsQ0FBQztLQUNyQztBQUFBOztBQ3pCSyxNQUFPLFVBQVcsU0FBUUEsZUFBVSxDQUFBO0FBQ3RDLElBQUEsV0FBQSxDQUFxQixJQUFZLEVBQVcsSUFBWSxFQUFXLGVBQXVCLEVBQUE7QUFDdEYsUUFBQSxLQUFLLEVBQUUsQ0FBQztRQURTLElBQUksQ0FBQSxJQUFBLEdBQUosSUFBSSxDQUFRO1FBQVcsSUFBSSxDQUFBLElBQUEsR0FBSixJQUFJLENBQVE7UUFBVyxJQUFlLENBQUEsZUFBQSxHQUFmLGVBQWUsQ0FBUTtLQUV6RjtBQUVELElBQUEsRUFBRSxDQUFDLEtBQWlCLEVBQUE7QUFDaEIsUUFBQSxPQUFPLEtBQUssQ0FBQyxJQUFJLEtBQUssSUFBSSxDQUFDLElBQUksSUFBSSxLQUFLLENBQUMsZUFBZSxJQUFJLElBQUksQ0FBQyxlQUFlLENBQUM7S0FDcEY7SUFFRCxLQUFLLEdBQUE7Ozs7QUFJRCxR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
