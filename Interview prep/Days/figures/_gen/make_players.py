"""Build a step-through player page for every animated SVG in ../ (run from this folder: python3 make_players.py).
Each page inlines the figure and drives its animation clock: Prev / Next step, Play / Pause, a scrub slider, speed.
Output: ../players/<figure>.html  (open in any browser; in Obsidian, clicking the link opens it in the default browser)."""
import os, re, glob, json, html
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.dirname(HERE); OUT = os.path.join(FIG, 'players')
os.makedirs(OUT, exist_ok=True)
PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title><style>
:root{--bg:#f6f8fa;--ink:#1f2328;--mute:#57606a;--line:#d0d7de;--acc:#0969da;--card:#fff}
@media (prefers-color-scheme: dark){:root{--bg:#0d1117;--ink:#e6edf3;--mute:#9198a1;--line:#30363d;--acc:#4493f8;--card:#161b22}}
body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,Helvetica,Arial,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:16px}
.fig{background:#fff;border:1px solid var(--line);border-radius:12px;overflow:hidden}
.fig svg{display:block;width:100%;height:auto}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:12px 0;padding:10px 12px;background:var(--card);border:1px solid var(--line);border-radius:10px}
button,select{font:inherit;font-weight:600;padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
button:hover{border-color:var(--acc)} button.main{background:var(--acc);color:#fff;border-color:var(--acc)}
input[type=range]{flex:1;min-width:160px;accent-color:var(--acc)}
#lab{font-weight:700;min-width:110px;font-variant-numeric:tabular-nums} .hint{color:var(--mute);font-size:13px}
</style></head><body><div class="wrap">
<div class="bar"><button id="prev" title="Left arrow">◀ Previous</button><button id="next" class="main" title="Right arrow">Next ▶</button>
<span id="lab"></span><input id="sl" type="range" min="0" max="1000" value="0" aria-label="position">
<button id="play" title="Space">Play</button>
<select id="sp" aria-label="speed"><option value="0.25">¼×</option><option value="0.5">½×</option><option value="1" selected>1×</option><option value="2">2×</option></select></div>
<div class="fig">__SVG__</div>
<p class="hint">Keys: ← previous step, → next step, space play / pause. Drag the slider to move freely. The picture stays still until you move it.</p>
</div><script>
const D=__DUR__, steps=__STEPS__, svg=document.querySelector('.fig svg'), EPS=0.02;
const lab=document.getElementById('lab'), sl=document.getElementById('sl'), playB=document.getElementById('play'), sp=document.getElementById('sp');
let t=EPS, playing=false, last=0;
svg.pauseAnimations();
function cur(){let i=0; for(let k=0;k<steps.length;k++) if(steps[k]<=t+1e-6) i=k; return i;}
function show(){svg.setCurrentTime(t); sl.value=Math.round(t/D*1000); lab.textContent='step '+(cur()+1)+' / '+steps.length;}
function go(i){playing=false; playB.textContent='Play'; i=(i+steps.length)%steps.length; t=steps[i]+EPS; show();}
document.getElementById('next').onclick=()=>go(cur()+1);
document.getElementById('prev').onclick=()=>go(cur()-1);
sl.oninput=()=>{playing=false; playB.textContent='Play'; t=Math.min(D-EPS,Math.max(EPS,sl.value/1000*D)); show();};
function tick(now){ if(!playing) return; t+=(now-last)/1000*parseFloat(sp.value); last=now; if(t>=D) t=EPS; show(); requestAnimationFrame(tick); }
playB.onclick=()=>{playing=!playing; playB.textContent=playing?'Pause':'Play'; if(playing){last=performance.now(); requestAnimationFrame(tick);} };
document.addEventListener('keydown',e=>{ if(e.key==='ArrowRight'){go(cur()+1);e.preventDefault();} else if(e.key==='ArrowLeft'){go(cur()-1);e.preventDefault();} else if(e.key===' '){playB.click();e.preventDefault();} });
show();
</script></body></html>"""
n = 0
for f in sorted(glob.glob(os.path.join(FIG, '*.svg'))):
    s = open(f, encoding='utf-8').read()
    if '<animate' not in s: continue
    durs = [float(x) for x in re.findall(r"dur='([0-9.]+)s'", s)]
    D = max(durs)
    pts = {0.0}
    for m in re.finditer(r"<animate\b[^>]*>", s):
        tag = m.group(0)
        d = re.search(r"dur='([0-9.]+)s'", tag); k = re.search(r"keyTimes='([^']*)'", tag)
        if d and k and "calcMode='discrete'" in tag:
            for x in k.group(1).split(';'):
                v = round(float(x) * float(d.group(1)), 3)
                if v < D - 1e-6: pts.add(v)
    steps = sorted(pts)
    # merge boundaries closer than 0.15 s; fall back to even steps when there are none or far too many
    merged = []
    for v in steps:
        if not merged or v - merged[-1] >= 0.15: merged.append(v)
    if len(merged) < 3 or len(merged) > 40: merged = [round(i * D / 16, 3) for i in range(16)]
    name = os.path.basename(f)[:-4]
    title = re.search(r'<title>(.*?)</title>', s)
    svg = re.sub(r"(<svg\b[^>]*?)\swidth='[^']*'\sheight='[^']*'", r'\1', s, count=1)
    page = (PAGE.replace('__TITLE__', html.escape(html.unescape(title.group(1))) if title else name)
            .replace('__DUR__', repr(D)).replace('__STEPS__', json.dumps(merged)).replace('__SVG__', svg))
    open(os.path.join(OUT, name + '.html'), 'w', encoding='utf-8').write(page); n += 1
print(n, 'players written to', OUT)
