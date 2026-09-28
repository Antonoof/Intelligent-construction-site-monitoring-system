"""Синтетические сцены стройплощадки для демо-данных: техника, фон, погода, время суток.

Стиль рисунков продолжает демо «План-факт» из frontend/plan-fact-site-monitor.html. Каждая машина
рисуется в своей системе координат (DIMS — ширина и высота рисунка), поэтому рамка разметки
вычисляется точно по размещению: это «эталонная разметка» для демо-детектора.
Сцена — viewBox 320×180, рендер 1280×720 (tools/make_demo_data.py).
"""
from __future__ import annotations

import random

COL = {"y": "#E1A21C", "yd": "#B7811A", "dk": "#2B2F33", "st": "#5A6068", "gl": "#9FC3D8", "rd": "#C4502E",
       "rdd": "#9B3E24", "wh": "#E7E9EB", "dr": "#D2D6DA", "drs": "#A9B1B8", "so": "#8C6B49", "bl": "#3E6FA8",
       "gr": "#4F8F5B", "grd": "#3C7047", "or": "#D98C1F"}

# ширина и высота рисунка машины в её собственных координатах
DIMS = {"excavator": (80, 46), "dump_truck": (66, 32), "bulldozer": (60, 32), "concrete_mixer": (72, 36),
        "concrete_pump": (88, 32), "concrete_pump_up": (88, 68), "crane_manipulator": (76, 42),
        "tower_crane": (170, 96), "roller": (58, 36), "truck": (70, 35), "mobile_crane": (96, 54),
        "loader": (66, 36), "grader": (88, 32), "drilling_rig": (52, 97)}


def wheels(xs, cy, r):
    return "".join(f'<circle cx="{x}" cy="{cy}" r="{r}" fill="{COL["dk"]}"/>'
                   f'<circle cx="{x}" cy="{cy}" r="{r * .4:.1f}" fill="#7B8087"/>' for x in xs)


def d_excavator(pose=-1):
    P = [52, 12, 64, 38] if pose < 0 else [[60, 6, 74, 32], [58, 3, 70, 17], [54, 9, 64, 25]][pose % 3]
    ex, ey, tx, ty = P
    bb = min(8, 46 - ty)
    return (f'<rect x="2" y="36" width="46" height="10" rx="5" fill="{COL["dk"]}"/><circle cx="8" cy="41" r="3" fill="#565B61"/>'
            f'<circle cx="42" cy="41" r="3" fill="#565B61"/><rect x="4" y="21" width="36" height="15" rx="2" fill="{COL["y"]}"/>'
            f'<rect x="2" y="23" width="8" height="12" rx="1" fill="{COL["yd"]}"/><rect x="25" y="10" width="15" height="13" rx="2" fill="{COL["y"]}"/>'
            f'<rect x="28" y="12.5" width="9.5" height="7" rx="1" fill="{COL["gl"]}"/>'
            f'<polyline points="36,22 {ex},{ey} {tx},{ty}" fill="none" stroke="{COL["y"]}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>'
            f'<path d="M{tx - 6},{ty} h11 l-2.5,{bb} h-7 z" fill="#474B50"/>')


def d_dump_truck(loaded=False):
    s = (f'<rect x="3" y="21" width="60" height="5" fill="{COL["dk"]}"/><path d="M3,7 L42,7 L44,21 L5,21 Z" fill="{COL["rd"]}"/>'
         f'<path d="M3,7 L42,7 L42.6,11 L3.5,11 Z" fill="{COL["rdd"]}"/>')
    if loaded:
        s += f'<path d="M7,7 Q22,0 38,7 Z" fill="{COL["so"]}"/>'
    s += (f'<path d="M46,9 L58,9 L63,17 L63,24 L46,24 Z" fill="{COL["rd"]}"/><path d="M49,11 L57,11 L60.5,17 L49,17 Z" fill="{COL["gl"]}"/>'
          + wheels([13, 25, 55], 27, 5))
    return s


def d_bulldozer(up=False):
    b = -4 if up else 0
    return (f'<rect x="3" y="22" width="40" height="10" rx="5" fill="{COL["dk"]}"/><circle cx="9" cy="27" r="2.5" fill="#565B61"/>'
            f'<circle cx="37" cy="27" r="2.5" fill="#565B61"/><rect x="6" y="12" width="34" height="11" rx="2" fill="{COL["y"]}"/>'
            f'<rect x="10" y="3" width="15" height="10" rx="1.5" fill="{COL["y"]}"/><rect x="12.5" y="5" width="10" height="6" rx="1" fill="{COL["gl"]}"/>'
            f'<rect x="31" y="5" width="2.4" height="8" fill="{COL["dk"]}"/><path d="M40,18 L50,{20 + b}" stroke="{COL["yd"]}" stroke-width="3"/>'
            f'<path d="M50,{13 + b} Q55.5,{22 + b} 50,{31 + b} L58,{31 + b} L58,{13 + b} Z" fill="{COL["yd"]}"/>')


def d_concrete_mixer(rot=0):
    return (f'<rect x="3" y="25" width="64" height="5" fill="{COL["dk"]}"/><g transform="rotate({-10 + rot} 26 15)">'
            f'<ellipse cx="26" cy="15" rx="22" ry="10.5" fill="{COL["dr"]}"/><path d="M10,9 L16,21 M20,5.5 L26,24 M31,5 L36,24 M41,7 L44,20" '
            f'stroke="{COL["drs"]}" stroke-width="2"/></g><path d="M3,17 L8,24" stroke="{COL["st"]}" stroke-width="2.5"/>'
            f'<path d="M50,11 L62,11 L68,19 L68,28 L50,28 Z" fill="{COL["wh"]}"/><path d="M53,13 L61,13 L65,19 L53,19 Z" fill="{COL["gl"]}"/>'
            + wheels([14, 27, 59], 31, 5))


def d_concrete_pump(up=False):
    o = 36 if up else 0
    g = (f'<rect x="3" y="{o + 21}" width="80" height="5" fill="{COL["dk"]}"/><rect x="6" y="{o + 10}" width="52" height="12" rx="2" fill="{COL["y"]}"/>'
         f'<path d="M62,{o + 8} L74,{o + 8} L80,{o + 16} L80,{o + 24} L62,{o + 24} Z" fill="{COL["y"]}"/>'
         f'<path d="M65,{o + 10} L73,{o + 10} L77,{o + 16} L65,{o + 16} Z" fill="{COL["gl"]}"/>' + wheels([14, 28, 44, 70], o + 27, 5))
    if not up:
        g += f'<path d="M10,{o + 8} H58 M13,{o + 5} H55 M16,{o + 2} H52" stroke="{COL["yd"]}" stroke-width="2.4" stroke-linecap="round"/>'
    else:
        g += (f'<polyline points="38,{o + 10} 24,5 60,11 84,30" fill="none" stroke="{COL["yd"]}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
              f'<path d="M84,30 V60" stroke="{COL["dk"]}" stroke-width="1.6"/>')
    return g


def d_crane_manipulator(up=False):
    arm, hk = ("48,15 33,3 19,9", (19, 9, 21)) if up else ("48,15 37,10 25,18", (25, 18, 25))
    return (f'<rect x="3" y="31" width="64" height="5" fill="{COL["dk"]}"/><rect x="4" y="26" width="42" height="5" fill="#6E747B"/>'
            f'<rect x="6" y="21.5" width="36" height="3.5" fill="#8C939B"/><rect x="6" y="17.5" width="36" height="3.5" fill="#7E858D"/>'
            f'<path d="M53,17 L64,17 L70,25 L70,34 L53,34 Z" fill="{COL["bl"]}"/><path d="M56,19 L63,19 L66.5,25 L56,25 Z" fill="{COL["gl"]}"/>'
            f'<rect x="46" y="13" width="4" height="18" fill="{COL["bl"]}"/>'
            f'<polyline points="{arm}" fill="none" stroke="{COL["bl"]}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
            f'<path d="M{hk[0]},{hk[1]} V{hk[2]}" stroke="{COL["dk"]}" stroke-width="1"/>' + wheels([14, 27, 61], 37, 5))


def d_tower_crane(t=0):
    tx, hy = 92 + (t * 23) % 64, 28 + (t * 17) % 34
    z = "".join(f"M70,{y}L78,{y + 8}" for y in range(16, 94, 8))
    j = "".join(f"M{x},10L{x + 4},15" for x in range(4, 166, 8))
    return (f'<g fill="none" stroke="#D39B1E" stroke-width="1.4"><path d="M70,96V14M78,96V14{z}"/>'
            f'<path d="M0,10H170M0,15H170{j}"/><path d="M74,2L20,10M74,2L150,10"/></g>'
            f'<rect x="4" y="15" width="14" height="8" fill="#6B7178"/><rect x="68" y="14" width="12" height="7" fill="#D39B1E"/>'
            f'<path d="M{tx},15V{hy}" stroke="{COL["dk"]}" stroke-width="1"/><rect x="{tx - 2.5}" y="{hy}" width="5" height="4" fill="{COL["dk"]}"/>')


def d_roller():
    return (f'<rect x="8" y="14" width="42" height="10" rx="2" fill="{COL["y"]}"/><rect x="36" y="22" width="16" height="6" fill="{COL["yd"]}"/>'
            f'<rect x="30" y="2" width="17" height="13" rx="1.5" fill="none" stroke="{COL["dk"]}" stroke-width="1.6"/>'
            f'<rect x="31" y="1" width="15" height="2.5" fill="{COL["dk"]}"/><rect x="32" y="5" width="12" height="7" fill="{COL["gl"]}" opacity=".85"/>'
            f'<circle cx="13" cy="24" r="11" fill="{COL["dr"]}"/><circle cx="13" cy="24" r="11" fill="none" stroke="{COL["st"]}" stroke-width="1.5"/>'
            f'<circle cx="13" cy="24" r="3.5" fill="{COL["st"]}"/><path d="M4,12 L22,12" stroke="{COL["yd"]}" stroke-width="3"/>'
            + wheels([47], 28, 8))


def d_truck(cargo=True):
    s = (f'<rect x="2" y="24" width="62" height="4" fill="{COL["dk"]}"/><rect x="2" y="13" width="45" height="11" fill="{COL["gr"]}"/>'
         f'<path d="M2,13 H47 M2,18.5 H47 M13,13 V24 M24,13 V24 M36,13 V24" stroke="{COL["grd"]}" stroke-width="1"/>')
    if cargo:
        s += (f'<rect x="5" y="6" width="12" height="7" fill="#B98B57"/><rect x="19" y="6" width="12" height="7" fill="#C49663"/>'
              f'<path d="M5,9.5 H17 M19,9.5 H31" stroke="#8E6A40" stroke-width=".8"/>')
    s += (f'<path d="M50,5 L62,5 L68,14 L68,27 L50,27 Z" fill="#E7E9EB"/><path d="M53,7.5 L61,7.5 L65,14 L53,14 Z" fill="{COL["gl"]}"/>'
          f'<rect x="50" y="20" width="18" height="2" fill="{COL["st"]}"/>' + wheels([12, 24, 58], 30, 5))
    return s


def d_mobile_crane():
    return (f'<rect x="2" y="38" width="90" height="6" fill="{COL["dk"]}"/><rect x="4" y="34" width="72" height="5" fill="#C9352F"/>'
            f'<path d="M78,24 L90,24 L95,32 L95,42 L78,42 Z" fill="#C9352F"/><path d="M81,26 L89,26 L92,32 L81,32 Z" fill="{COL["gl"]}"/>'
            f'<rect x="28" y="24" width="26" height="10" rx="1.5" fill="{COL["y"]}"/><rect x="45" y="18" width="10" height="8" fill="{COL["y"]}"/>'
            f'<rect x="46.5" y="19.5" width="7" height="4.5" fill="{COL["gl"]}"/>'
            f'<path d="M34,26 L7,3" stroke="{COL["y"]}" stroke-width="4.5" stroke-linecap="round"/>'
            f'<path d="M7,3 V22" stroke="{COL["dk"]}" stroke-width="1"/><rect x="4.5" y="22" width="5" height="3.5" fill="{COL["dk"]}"/>'
            f'<path d="M10,44 V50 M70,44 V50" stroke="{COL["st"]}" stroke-width="2"/>' + wheels([14, 26, 60, 72, 86], 46, 5))


def d_loader(up=False):
    by = 4 if up else 16
    return (f'<rect x="22" y="12" width="38" height="12" rx="2" fill="{COL["y"]}"/><rect x="36" y="1" width="14" height="12" rx="1.5" fill="{COL["y"]}"/>'
            f'<rect x="38" y="3" width="10" height="7" fill="{COL["gl"]}"/><rect x="54" y="6" width="3" height="7" fill="{COL["dk"]}"/>'
            f'<path d="M28,16 L10,{by + 6}" stroke="{COL["yd"]}" stroke-width="4" stroke-linecap="round"/>'
            f'<path d="M1,{by} L12,{by} L14,{by + 12} L3,{by + 14} Z" fill="{COL["yd"]}"/>' + wheels([20, 50], 26, 9))


def d_grader():
    return (f'<path d="M6,16 L56,12" stroke="{COL["y"]}" stroke-width="5" stroke-linecap="round"/>'
            f'<rect x="52" y="10" width="34" height="11" rx="2" fill="{COL["y"]}"/><rect x="54" y="0" width="14" height="12" rx="1.5" fill="{COL["y"]}"/>'
            f'<rect x="56" y="2" width="10" height="7" fill="{COL["gl"]}"/><path d="M26,19 L44,19 L46,25 L24,25 Z" fill="#6B7178"/>'
            f'<path d="M30,14 L35,19 M40,13.5 L38,19" stroke="{COL["st"]}" stroke-width="1.5"/>' + wheels([8, 62, 77], 25, 7))


def d_drilling_rig():
    lat = "".join(f"M10,{y}L17,{y + 6}M17,{y}L10,{y + 6}" for y in range(6, 78, 6))
    aug = "".join(f"M10,{y}L17,{y + 3}" for y in range(18, 84, 4))
    return (f'<rect x="2" y="86" width="48" height="10" rx="5" fill="{COL["dk"]}"/><circle cx="8" cy="91" r="3" fill="#565B61"/>'
            f'<circle cx="44" cy="91" r="3" fill="#565B61"/><rect x="18" y="70" width="30" height="16" rx="2" fill="{COL["or"]}"/>'
            f'<rect x="32" y="58" width="15" height="13" rx="1.5" fill="{COL["or"]}"/><rect x="34" y="60" width="11" height="7" fill="{COL["gl"]}"/>'
            f'<rect x="9.5" y="2" width="8" height="82" fill="none" stroke="{COL["or"]}" stroke-width="1.6"/>'
            f'<path d="{lat}" stroke="{COL["or"]}" stroke-width=".7"/><rect x="8" y="0" width="11" height="4" fill="{COL["dk"]}"/>'
            f'<path d="M13.5,4 V96" stroke="{COL["st"]}" stroke-width="3"/><path d="{aug}" stroke="#8C939B" stroke-width="1"/>'
            f'<path d="M17,74 L20,74" stroke="{COL["or"]}" stroke-width="3"/>')


DRAW = {"excavator": d_excavator, "dump_truck": d_dump_truck, "bulldozer": d_bulldozer,
        "concrete_mixer": d_concrete_mixer, "concrete_pump": d_concrete_pump, "crane_manipulator": d_crane_manipulator,
        "tower_crane": d_tower_crane, "roller": d_roller, "truck": d_truck, "mobile_crane": d_mobile_crane,
        "loader": d_loader, "grader": d_grader, "drilling_rig": d_drilling_rig}


def place(cls: str, gx: float, gy: float, s: float, flip: bool = False, **kw) -> tuple[str, tuple]:
    """Разместить машину по точке опоры (середина нижней стороны) → (SVG, рамка в координатах сцены)."""
    key = "concrete_pump_up" if cls == "concrete_pump" and kw.get("up") else cls
    w, h = DIMS[key]
    x, y = gx - w * s / 2, gy - h * s
    g = DRAW[cls](**kw)
    tf = (f"translate({x + w * s:.2f},{y:.2f}) scale({-s},{s})" if flip else f"translate({x:.2f},{y:.2f}) scale({s})")
    return f'<g transform="{tf}">{g}</g>', (x, y, x + w * s, y + h * s)


# ------------------------------------------------------------------ фоны

POSTS = "".join(f"M{x},80V89" for x in range(4, 320, 16))


def _rebar():
    d = ""
    for i in range(13):
        x1, x2 = 26 + i * 22.5, 2 + i * 26.3
        d += f"M{x1:.1f},94L{x2:.1f},180"
    for y in (104, 116, 130, 146, 164):
        f = (y - 94) / 86
        d += f"M{26 - 24 * f:.1f},{y}H{296 + 22 * f:.1f}"
    return d


REBAR = _rebar()


def city(y0=86, seed=1, color="#8F98A3"):
    rnd = random.Random(seed)
    out = ""
    x = -4
    while x < 320:
        w = rnd.randint(18, 46)
        h = rnd.randint(18, 54)
        out += f'<rect x="{x}" y="{y0 - h}" width="{w}" height="{h}" fill="{color}"/>'
        for wy in range(y0 - h + 4, y0 - 4, 7):
            for wx in range(x + 3, x + w - 3, 6):
                if rnd.random() < .55:
                    out += f'<rect x="{wx}" y="{wy}" width="2.4" height="3" fill="#A9B2BC" opacity=".7"/>'
        x += w + rnd.randint(2, 10)
    return out


def fence(y=88):
    return f'<path d="M0,{y}H320{POSTS.replace("80V89", f"{y - 8}V{y + 1}")}" stroke="#6D6860" stroke-width="1.2" fill="none"/>'


def ground(color):
    return f'<rect y="86" width="320" height="94" fill="{color}"/>'


def scene(kind: str, winter=False) -> str:
    soil, soil2, dirt = ("#E4E8EC", "#CBD3DA", "#B9C2CA") if winter else ("#A8896A", "#8A6B4C", "#76593D")
    if kind == "pit":
        return (city(90, 3) + ground(soil) + fence(88) + f'<path d="M16,96H304L318,180H2Z" fill="{soil2}"/>'
                f'<path d="M24,118H300M10,146H310" stroke="{dirt}" stroke-width="1.2" opacity=".75"/><path d="M246,90Q268,72 298,90Z" fill="#9B7D5B"/>')
    if kind == "slab":
        return (city(90, 5) + f'<rect y="86" width="320" height="94" fill="#9C9990"/><path d="M26,94H296L318,180H2Z" fill="#B5B8BA"/>'
                f'<path d="{REBAR}" stroke="#80746A" stroke-width=".6" opacity=".75" fill="none"/><path d="M26,94H296" stroke="#C4904C" stroke-width="3"/>'
                f'<g fill="#C4904C"><rect x="20" y="84" width="40" height="4"/><rect x="22" y="80" width="36" height="4"/></g>')
    if kind == "entrance":
        return (city(90, 8) + f'<rect y="86" width="320" height="94" fill="#A08869"/><path d="M0,112L146,106L166,180H0Z" fill="#5E6165"/>'
                f'<path d="M30,146L52,144M84,141L106,139M126,136L140,135" stroke="#E8E4D8" stroke-width="2"/><path d="M184,104L318,100V180H192Z" fill="#9C8263"/>'
                f'<g fill="#8B939B"><rect x="262" y="112" width="44" height="4"/><rect x="262" y="117" width="44" height="4"/><rect x="264" y="107" width="40" height="4"/></g>'
                f'<path d="M172,104V180" stroke="#6D6860" stroke-width="1.5"/>' + fence(92))
    if kind == "drain":
        return (city(90, 11) + ground("#A58868") + fence(88) + f'<path d="M120,180L178,96H192L150,180Z" fill="#5C4936"/>'
                f'<path d="M126,180L180,100" stroke="#4A3A2B" stroke-width="1"/>'
                f'<g fill="#A9B1B7" stroke="#79828A" stroke-width="1"><circle cx="252" cy="126" r="7"/><circle cx="266" cy="126" r="7"/><circle cx="259" cy="114" r="7"/></g>'
                f'<path d="M60,120Q90,104 118,120Z" fill="#8C6D4E"/>')
    if kind == "camp":
        cab = ""
        for i, (x, y, c) in enumerate([(150, 70, "#3E6FA8"), (184, 70, "#E7E9EB"), (218, 70, "#3E6FA8"), (252, 70, "#E7E9EB"),
                                       (150, 52, "#E7E9EB"), (184, 52, "#3E6FA8"), (218, 52, "#E7E9EB")]):
            cab += (f'<rect x="{x}" y="{y}" width="33" height="18" fill="{c}" stroke="#5A6068" stroke-width=".6"/>'
                    f'<rect x="{x + 5}" y="{y + 5}" width="7" height="6" fill="#9FC3D8"/><rect x="{x + 20}" y="{y + 4}" width="6" height="14" fill="#6E747B"/>')
        return (city(88, 13) + ground("#9E9A92") + cab + f'<path d="M150,52H284" stroke="#5A6068" stroke-width="1"/>'
                f'<rect x="0" y="100" width="320" height="80" fill="#8E8B85"/><path d="M0,100H320" stroke="#6D6860"/>'
                f'<path d="M20,96V112M60,96V112" stroke="#C9352F" stroke-width="2"/><path d="M20,98H60" stroke="#E8E4D8" stroke-width="2" stroke-dasharray="4 3"/>'
                f'<g fill="#7E7A73"><rect x="80" y="104" width="30" height="3"/><rect x="84" y="100" width="22" height="4"/></g>')
    if kind == "overview":
        return ('<rect y="34" width="320" height="146" fill="#A48A6B"/>' + city(36, 21, "#98A1AA").replace('height="', 'height="', 1)
                + '<path d="M0,40H320" stroke="#6D6860" stroke-width=".8"/>'
                  '<path d="M20,70L150,62L170,150L6,168Z" fill="#7F6246"/><path d="M28,78L146,71L162,142L16,158Z" fill="#6E533B" opacity=".6"/>'
                  '<path d="M184,62L300,58L316,138L196,148Z" fill="#B3B5B6"/>'
                  f'<path d="M190,70L304,66M192,80L306,76M194,92L308,88M196,104L310,100M198,118L312,114M200,132L314,128" stroke="#8B7F74" stroke-width=".5"/>'
                  '<path d="M188,62L196,148M212,61L222,146M236,60L246,144M260,59L270,142M284,58L292,140" stroke="#8B7F74" stroke-width=".5"/>'
                  '<g fill="#3E6FA8"><rect x="276" y="150" width="20" height="9"/><rect x="298" y="150" width="20" height="9"/></g>'
                  '<g fill="#E7E9EB"><rect x="276" y="160" width="20" height="9"/><rect x="298" y="160" width="20" height="9"/></g>'
                  '<path d="M0,176L320,174" stroke="#6D6860" stroke-width="1"/>')
    if kind == "school":
        frame = ""
        for fl in range(4):
            y = 118 - fl * 22
            frame += f'<rect x="40" y="{y}" width="190" height="4" fill="#A7ABAE"/>'
            for x in range(44, 230, 26):
                frame += f'<rect x="{x}" y="{y - 18}" width="4" height="18" fill="#9A9EA2"/>'
        frame += '<rect x="40" y="30" width="190" height="4" fill="#C9A15A"/>'
        frame += "".join(f'<rect x="{x}" y="22" width="3" height="8" fill="#6E747B"/>' for x in range(46, 228, 12))
        return (city(96, 17, "#A2AAB2") + f'<rect y="96" width="320" height="84" fill="{soil}"/>' + frame
                + '<rect x="36" y="122" width="198" height="6" fill="#8E9296"/>'
                  f'<path d="M0,132H320" stroke="{soil2}" stroke-width="2"/><path d="M250,140Q280,126 316,140Z" fill="{soil2}"/>')
    if kind == "school_overview":
        frame = ""
        for fl in range(4):
            y = 92 - fl * 12
            frame += f'<rect x="120" y="{y}" width="110" height="2.5" fill="#A7ABAE"/>'
            for x in range(122, 230, 14):
                frame += f'<rect x="{x}" y="{y - 10}" width="2.2" height="10" fill="#9A9EA2"/>'
        return (f'<rect y="40" width="320" height="140" fill="{soil}"/>' + city(42, 23, "#AAB2BA")
                + '<path d="M0,44H320" stroke="#8E9296" stroke-width=".8"/>'
                + frame + '<rect x="116" y="94" width="118" height="4" fill="#8E9296"/>'
                  f'<path d="M20,120L300,116L316,176L4,178Z" fill="{soil2}" opacity=".7"/>'
                  '<g fill="#8B939B"><rect x="36" y="128" width="40" height="3"/><rect x="36" y="132" width="40" height="3"/><rect x="40" y="124" width="32" height="3"/></g>'
                  '<g fill="#B98B57"><rect x="250" y="130" width="14" height="8"/><rect x="266" y="130" width="14" height="8"/><rect x="258" y="122" width="14" height="8"/></g>')
    raise ValueError(kind)


# ------------------------------------------------------------------ свет и погода

SKY = {"day": ("#8DB6E2", "#DCE8F3"), "morning": ("#A9C4DF", "#EADFD2"), "dusk": ("#5B6C9A", "#E7A77A"),
       "night": ("#131B2E", "#2A3550"), "rain": ("#7E8A96", "#B7C0C8"), "fog": ("#C9CED3", "#DDE1E4"),
       "winter": ("#AEBBC8", "#E3E8EE"), "predawn": ("#0E1422", "#1B2233")}


def effects(light: str, weather: str, seed: int) -> str:
    rnd = random.Random(seed)
    s = ""
    if weather == "rain":
        s += ('<defs><pattern id="rn" width="7" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(16)">'
              '<path d="M3.5 0V5" stroke="#DDE6EF" stroke-opacity=".6" stroke-width=".8"/></pattern></defs>'
              '<rect width="320" height="180" fill="#3C4650" fill-opacity=".16"/><rect width="320" height="180" fill="url(#rn)"/>')
    if weather == "snow":
        s += '<rect width="320" height="180" fill="#E8EEF4" fill-opacity=".10"/>'
        s += "".join(f'<circle cx="{rnd.uniform(0, 320):.1f}" cy="{rnd.uniform(0, 180):.1f}" r="{rnd.uniform(.4, 1.1):.2f}" '
                     f'fill="#FFFFFF" opacity="{rnd.uniform(.5, .95):.2f}"/>' for _ in range(260))
    if weather == "fog":
        s += '<rect width="320" height="180" fill="#E4E7EA" fill-opacity=".86"/>'
    if light == "dusk":
        s += '<rect width="320" height="180" fill="#3A2A55" fill-opacity=".22"/>'
    if light == "night_lit":
        s += ('<defs><radialGradient id="fl" cx=".5" cy=".3" r=".7"><stop offset="0" stop-color="#FFD59A" stop-opacity=".45"/>'
              '<stop offset="1" stop-color="#FFD59A" stop-opacity="0"/></radialGradient></defs>'
              '<rect width="320" height="180" fill="#081022" fill-opacity=".55"/><rect width="320" height="180" fill="url(#fl)"/>'
              '<g fill="#FFE8B0"><circle cx="40" cy="60" r="2"/><circle cx="280" cy="58" r="2"/><circle cx="160" cy="30" r="1.6"/></g>')
    if light == "dark":
        s += '<rect width="320" height="180" fill="#050607" fill-opacity=".95"/>'
    return s


def osd(cam: str, when_text: str, name: str) -> str:
    return (f'<g font-family="DejaVu Sans Mono, monospace" font-size="6.2" font-weight="bold">'
            f'<text x="5" y="10" fill="#000" opacity=".55" dx=".4" dy=".4">{cam}  {when_text}</text>'
            f'<text x="5" y="10" fill="#FFFFFF">{cam}  {when_text}</text>'
            f'<text x="315" y="10" text-anchor="end" fill="#000" opacity=".55" dx=".4" dy=".4">{name}</text>'
            f'<text x="315" y="10" text-anchor="end" fill="#FFFFFF">{name}</text></g>')


def compose(kind: str, light: str, weather: str, machines: list[dict], cam: str, when_text: str, cam_name: str,
            seed: int = 0) -> tuple[str, list[dict]]:
    """Собрать SVG сцены и эталонные рамки машин (в координатах 320×180)."""
    winter = weather == "snow" or light == "winter"
    sky_key = {"night_lit": "night", "dark": "night"}.get(light, "rain" if weather == "rain" else light)
    if winter and light == "day":
        sky_key = "winter"
    top, bot = SKY.get(sky_key, SKY["day"])
    body = (f'<defs><linearGradient id="sk" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{top}"/>'
            f'<stop offset="1" stop-color="{bot}"/></linearGradient></defs><rect width="320" height="180" fill="url(#sk)"/>')
    body += scene(kind, winter)
    boxes = []
    # сначала дальние (башенный кран), затем по глубине — ближние машины рисуются поверх
    for mch in sorted(machines, key=lambda d: (d["cls"] != "tower_crane", d["gy"])):
        kw = {k: v for k, v in mch.items() if k in ("pose", "loaded", "up", "t", "cargo", "rot")}
        svg, box = place(mch["cls"], mch["gx"], mch["gy"], mch["s"], mch.get("flip", False), **kw)
        body += svg
        if mch.get("label", True):
            boxes.append({"cls": mch["cls"], "box": box, "conf": mch.get("conf", 0.9)})
    body += effects(light, weather, seed)
    body += osd(cam, when_text, cam_name)
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 180" width="1280" height="720">{body}</svg>', boxes
