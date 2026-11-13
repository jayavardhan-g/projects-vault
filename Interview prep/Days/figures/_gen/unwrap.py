import re, sys
p = sys.argv[1]; apply = len(sys.argv) > 2
L = open(p, encoding='utf-8').read().split('\n')
marker = re.compile(r'^\s*(?:>\s*)*(?:[-*+] |\d+\. |#|\||```|\[!|!\[\[|\$\$|---|<)')
def prefix(s):  # blockquote / indent prefix
    return re.match(r'^(\s*(?:>\s?)*)', s).group(1)
out = []; fence = False; joins = 0; i = 0
while i < len(L):
    cur = L[i]
    if cur.lstrip('> ').startswith('```'): fence = not fence
    while (not fence and i + 1 < len(L) and len(cur) >= 95 and cur.strip() and not cur.lstrip().startswith(('|', '#', '```'))
           and L[i+1].strip() and L[i+1].strip('> ').strip() and not marker.match(L[i+1])
           and prefix(L[i+1]).replace(' ', '').count('>') == prefix(cur).replace(' ', '').count('>')
           and not cur.endswith('  ') and '[!' not in cur[:12]
           and not re.search(r'[.?!:][*_`)"]*$', cur.rstrip())):
        nxt = L[i+1][len(prefix(L[i+1])):].lstrip()
        long_next = len(L[i+1]) >= 95
        cur = cur.rstrip() + ' ' + nxt; joins += 1; i += 1
        if not long_next: break
    out.append(cur); i += 1
print(p[-45:], 'joins:', joins, 'lines', len(L), '->', len(out))
if apply: open(p, 'w', encoding='utf-8').write('\n'.join(out))
else:
    import difflib
    d = [x for x in difflib.unified_diff(L, out, lineterm='', n=0) if x[:1] in '+-' and x[:3] not in ('+++', '---')]
    print('\n'.join(x[:230] for x in d[:36]))
