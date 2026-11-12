"""Split every frame-per-step animation in ../ into still pictures, one SVG per step: ../steps/<figure>-stepNN.svg
(run from this folder: python3 make_steps.py). Only figures built with mxfigs.save() are frame-per-step."""
import os, re, glob, xml.dom.minidom as dom
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.dirname(HERE); OUT = os.path.join(FIG, 'steps')
os.makedirs(OUT, exist_ok=True)
OPEN = r"<g opacity='[01]'><animate attributeName='opacity'[^>]*calcMode='discrete'[^>]*/>\n"
for f in sorted(glob.glob(os.path.join(FIG, '*.svg'))):
    s = open(f, encoding='utf-8').read()
    parts = re.split(OPEN, s)
    if len(parts) < 3 or '<animate' in ''.join(parts): continue      # not a pure frame-per-step figure
    name = os.path.basename(f)[:-4]; n = len(parts) - 1
    for k in range(1, n + 1):
        body = parts[k]
        svg = parts[0] + '<g>\n' + body + ('' if k == n else '</svg>')
        dom.parseString(svg)
        open(os.path.join(OUT, f'{name}-step{k:02d}.svg'), 'w', encoding='utf-8').write(svg)
    print(name, n)
