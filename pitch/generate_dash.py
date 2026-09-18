"""Render a faithful Stripify 'Overview' dashboard PNG (1872x917) to replace the old
Hebbia screenshot on slide 3. Mirrors the real comptroller app: sidebar, KPI cards,
spend-by-category bars, risk panel, and a flagged-transactions table."""
from PIL import Image, ImageDraw, ImageFont

W, H = 1872, 917
BG, PANEL, PANEL2, LINE, LINE2 = "#fbfbfc", "#ffffff", "#f4f5f7", "#e6e8ec", "#eef0f3"
INK, MUTED = "#1b1e25", "#6b7280"
STRIPE, STRIPEBG, GREEN, AMBER, RED = "#635bff", "#efeeff", "#15a34a", "#b7791f", "#dc2626"

def font(path_opts, size):
    for p in path_opts:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()

R = ["C:/Windows/Fonts/segoeui.ttf", "DejaVuSans.ttf"]
B = ["C:/Windows/Fonts/segoeuib.ttf", "DejaVuSans-Bold.ttf"]
SB = ["C:/Windows/Fonts/seguisb.ttf", "C:/Windows/Fonts/segoeuib.ttf", "DejaVuSans-Bold.ttf"]

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

def rrect(xy, radius, fill=None, outline=None, width=1):
    try:
        d.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
    except Exception:
        d.rectangle(xy, fill=fill, outline=outline, width=width)

def text(pos, s, fnt, fill=INK, anchor=None):
    d.text(pos, s, font=fnt, fill=fill, anchor=anchor)

# ---------- sidebar ----------
SW = 300
d.rectangle([0, 0, SW, H], fill=PANEL2)
d.line([SW, 0, SW, H], fill=LINE, width=1)
text((28, 30), "Stripe", font(B, 30), fill=STRIPE)
bw = d.textlength("Stripe", font=font(B, 30))
text((28 + bw, 30), "ify", font(B, 30), fill=INK)

nav = [("COMPANY", None), ("Overview", "active"), ("Transactions", None), ("Cards", None),
       ("People", None), ("CONTROLS", None), ("Fraud & Risk", None), ("Bill Pay / AP", None),
       ("Policy", None), ("AUTOMATION", None), ("Scheduled Exports", None)]
y = 92
for label, state in nav:
    if label.isupper():
        text((30, y + 6), label, font(SB, 13), fill="#9ca3af")
        y += 34
        continue
    if state == "active":
        rrect([22, y - 4, SW - 18, y + 30], 9, fill=STRIPE)
        text((34, y + 2), label, font(SB, 18), fill="#ffffff")
    else:
        text((34, y + 2), label, font(R, 18), fill=INK)
    y += 42

# ---------- topbar ----------
PX = SW + 32
d.line([SW, 70, W, 70], fill=LINE, width=1)
text((PX, 22), "Overview", font(B, 26), fill=INK)
rrect([W - 210, 20, W - 32, 54], 9, outline=LINE, width=1, fill=PANEL)
text((W - 194, 28), "Finance Admin", font(R, 16), fill=MUTED)

# ---------- lede ----------
text((PX, 92), "Vertex Capital Inc  ·  42 employees  ·  5,862 card transactions  ·  calibrated demo data",
     font(R, 17), fill=MUTED)

# ---------- KPI cards ----------
kpis = [("VALUE IDENTIFIED", "$1.24M", "", True), ("CASH", "$4.82M", "healthy", False),
        ("RUNWAY", "13 mo", "", False), ("MONTHLY SPEND", "$1.61M", "", False),
        ("COMPLIANCE", "92.4%", "", False), ("FRAUD ALERTS", "15", "3 rings", False)]
n = len(kpis)
gap, kx0, ky0 = 16, PX, 128
kw = (W - PX - 32 - gap * (n - 1)) // n
kh = 104
for i, (lab, val, sub, hero) in enumerate(kpis):
    x = kx0 + i * (kw + gap)
    rrect([x, ky0, x + kw, ky0 + kh], 12, fill=STRIPEBG if hero else PANEL,
          outline="#d9d6ff" if hero else LINE, width=1)
    text((x + 16, ky0 + 14), lab, font(SB, 12), fill=MUTED)
    text((x + 16, ky0 + 36), val, font(B, 31), fill=STRIPE if hero else INK)
    if sub:
        text((x + 16, ky0 + 78), sub, font(R, 13), fill=MUTED)

# ---------- two columns: spend bars + risk ----------
cy0 = ky0 + kh + 26
col_gap = 24
left_w = int((W - PX - 32 - col_gap) * 0.6)
right_x = PX + left_w + col_gap
right_w = W - 32 - right_x
ch = 300

text((PX, cy0), "SPEND BY CATEGORY", font(SB, 13), fill=STRIPE)
text((right_x, cy0), "RISK & SAVINGS", font(SB, 13), fill=STRIPE)
cb = cy0 + 26
rrect([PX, cb, PX + left_w, cb + ch], 12, fill=PANEL, outline=LINE, width=1)
rrect([right_x, cb, right_x + right_w, cb + ch], 12, fill=PANEL, outline=LINE, width=1)

bars = [("software", 420), ("travel", 310), ("meals & ent.", 182),
        ("advertising", 150), ("office supplies", 96), ("professional", 74)]
mx = max(v for _, v in bars)
bx, by = PX + 22, cb + 26
lblw, barw = 168, left_w - 168 - 150
for i, (k, v) in enumerate(bars):
    yy = by + i * 42
    text((bx, yy), k, font(R, 16), fill=MUTED)
    track = [bx + lblw, yy + 4, bx + lblw + barw, yy + 16]
    rrect(track, 6, fill=PANEL2)
    fillw = int(barw * v / mx)
    rrect([bx + lblw, yy + 4, bx + lblw + fillw, yy + 16], 6, fill=STRIPE)
    text((bx + lblw + barw + 16, yy), f"${v}k", font(SB, 16), fill=INK)

rl = [("15 high-risk · $128k exposure · AUC 0.91", RED, RED),
      ("$84k redundant SaaS · $22k duplicates", INK, STRIPE),
      ("$96k / yr idle-cash yield", INK, STRIPE),
      ("credit limit $250k → $400k  (low risk)", INK, STRIPE)]
rx, ry = right_x + 20, cb + 28
for i, (s, col, dot) in enumerate(rl):
    yy = ry + i * 64
    d.ellipse([rx, yy + 5, rx + 12, yy + 17], fill=dot)
    text((rx + 28, yy + 1), s, font(R, 16), fill=col)
    if i < len(rl) - 1:
        d.line([rx, yy + 44, right_x + right_w - 20, yy + 44], fill=LINE2, width=1)

# ---------- flagged transactions table ----------
ty0 = cb + ch + 30
text((PX, ty0), "RECENT FLAGGED TRANSACTIONS", font(SB, 13), fill=STRIPE)
tb = ty0 + 26
tw = W - 32 - PX
rrect([PX, tb, PX + tw, H - 24], 12, fill=PANEL, outline=LINE, width=1)
cols = [("date", PX + 20), ("merchant", PX + 150), ("employee", PX + 430),
        ("category", PX + 700), ("amount", PX + 980), ("flags", PX + 1140), ("fraud", tw + PX - 90)]
for c, cx in cols:
    text((cx, tb + 14), c.upper(), font(SB, 12), fill=MUTED)
d.line([PX + 12, tb + 40, PX + tw - 12, tb + 40], fill=LINE2, width=1)

rows = [("2026-06-21", "Aramark", "Theo Park", "meals", "$1,284", "weekend meal", "31%", False),
        ("2026-06-19", "Shell Oil", "Kenji Reed", "fuel", "$412", "blocked category", "44%", False),
        ("2026-06-18", "Sundry LLC", "Mara Voss", "other", "$8,940", "fraud", "88%", True),
        ("2026-06-16", "Delta Air", "Priya Shah", "travel", "$3,120", "no receipt", "27%", False),
        ("2026-06-15", "Quartz Tech", "Sam Olin", "software", "$5,400", "over limit", "39%", False)]
ry = tb + 52
for (dt, mer, emp, cat, amt, flag, fr, isf) in rows:
    text((PX + 20, ry), dt, font(R, 15), fill=INK)
    text((PX + 150, ry), mer, font(R, 15), fill=INK)
    text((PX + 430, ry), emp, font(R, 15), fill=INK)
    text((PX + 700, ry), cat, font(R, 15), fill=INK)
    text((PX + 980, ry), amt, font(SB, 15), fill=INK)
    pill = RED if isf else AMBER
    pbg = "#fdecec" if isf else "#fdf6e3"
    fw = d.textlength(flag, font=font(R, 13)) + 18
    rrect([PX + 1140, ry - 2, PX + 1140 + fw, ry + 22], 7, fill=pbg)
    text((PX + 1149, ry + 1), flag, font(R, 13), fill=pill)
    text((tw + PX - 90, ry), fr, font(SB, 15), fill=INK)
    ry += 42

img.save("stripify_dash.png")
print("saved stripify_dash.png", img.size)
