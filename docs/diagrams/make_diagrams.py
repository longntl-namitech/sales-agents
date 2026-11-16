"""Draw the README diagrams in a hand-drawn style. Run from the repo root:

    python docs/diagrams/make_diagrams.py

Standard library only. The output is deterministic (fixed seeds), so a re-run changes an SVG
only where the drawing code or its text changed. The font is Patrick Hand (SIL OFL 1.1, see
OFL.txt), cut down to Latin characters and embedded, so GitHub shows it without loading
anything.
"""

import base64
import html
import json
import math
import random
from pathlib import Path

HERE = Path(__file__).parent
INK = "#1f1f1f"
FILLS = {
    "lilac": "#ECE3F8",
    "yellow": "#FFF3C4",
    "green": "#DFF2DC",
    "blue": "#DCEAF8",
    "pink": "#FBE1E3",
    "paper": "#FBFAF6",
}
FONT = "Hand, 'Patrick Hand', 'Comic Sans MS', cursive"

# Line icons in a 40 x 40 box, drawn with the ink stroke and no fill.
ICONS = {
    "doc": '<path d="M9 4 H26 L33 11 V36 H9 Z M26 4 V11 H33 M14 18 H28 M14 24 H28 M14 30 H23"/>',
    "folder": '<path d="M4 11 H15 L19 15 H36 V34 H4 Z"/>',
    "scissors": (
        '<circle cx="11" cy="30" r="5"/><circle cx="27" cy="30" r="5"/>'
        '<path d="M14 26 L31 5 M24 26 L7 5"/>'
    ),
    "bolt": '<path d="M23 3 L9 22 H19 L15 37 L31 15 H21 Z"/>',
    "layers": '<path d="M20 5 L36 13 L20 21 L4 13 Z M4 20 L20 28 L36 20 M4 27 L20 35 L36 27"/>',
    "search": '<circle cx="16" cy="16" r="10"/><path d="M24 24 L35 35"/>',
    "trophy": (
        '<path d="M11 5 H29 V14 A9 9 0 0 1 11 14 Z M11 8 H5 A6 6 0 0 0 12 17 '
        'M29 8 H35 A6 6 0 0 1 28 17 M20 23 V30 M13 35 H27 M15 30 H25"/>'
    ),
    "pencil": '<path d="M8 32 L29 11 L34 16 L13 37 L6 38 Z M25 15 L30 20"/>',
    "sparkle": '<path d="M20 4 L23 17 L36 20 L23 23 L20 36 L17 23 L4 20 L17 17 Z"/>',
    "bubble": '<path d="M5 7 H35 V27 H18 L10 34 V27 H5 Z M11 14 H29 M11 20 H24"/>',
    "chart": '<path d="M5 5 V35 H36 M9 28 L17 19 L24 24 L34 11"/>',
    "box": '<path d="M5 12 L20 5 L35 12 V29 L20 36 L5 29 Z M5 12 L20 19 L35 12 M20 19 V36"/>',
}


class Sketch:
    def __init__(self, width: int, height: int, seed: int):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)
        self.parts: list[str] = []

    def _j(self, amp: float = 1.5) -> float:
        return self.rng.uniform(-amp, amp)

    def _stroke(self, d: str, width: float = 2.4, opacity: float = 1.0, dash: str = "") -> None:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{INK}" stroke-width="{width}" '
            f'stroke-opacity="{opacity}" stroke-linecap="round" stroke-linejoin="round"{extra}/>'
        )

    def _seg(self, x1: float, y1: float, x2: float, y2: float, bow: float = 2.0) -> str:
        """A cubic from the current point to (x2, y2) that bows a little, like a pen line."""
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length
        b1, b2 = self._j(bow), self._j(bow)
        c1 = (x1 + dx / 3 + nx * b1, y1 + dy / 3 + ny * b1)
        c2 = (x1 + 2 * dx / 3 + nx * b2, y1 + 2 * dy / 3 + ny * b2)
        return f"C {c1[0]:.1f} {c1[1]:.1f} {c2[0]:.1f} {c2[1]:.1f} {x2:.1f} {y2:.1f}"

    def _rounded(self, x: float, y: float, w: float, h: float, r: float) -> str:
        j = self._j
        return " ".join(
            [
                f"M {x + r:.1f} {y + j(0.8):.1f}",
                self._seg(x + r, y, x + w - r, y),
                f"Q {x + w:.1f} {y:.1f} {x + w + j(0.8):.1f} {y + r:.1f}",
                self._seg(x + w, y + r, x + w, y + h - r),
                f"Q {x + w:.1f} {y + h:.1f} {x + w - r:.1f} {y + h + j(0.8):.1f}",
                self._seg(x + w - r, y + h, x + r, y + h),
                f"Q {x:.1f} {y + h:.1f} {x + j(0.8):.1f} {y + h - r:.1f}",
                self._seg(x, y + h - r, x, y + r),
                f"Q {x:.1f} {y:.1f} {x + r:.1f} {y:.1f}",
            ]
        )

    def text(
        self,
        x: float,
        y: float,
        s: str,
        size: float,
        anchor: str = "start",
        bold: bool = False,
        rotate: float = 0,
    ) -> None:
        extra = f' stroke="{INK}" stroke-width="{size / 40:.2f}"' if bold else ""
        if rotate:
            extra += f' transform="rotate({rotate} {x:.1f} {y:.1f})"'
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{INK}"{extra}>{html.escape(s)}</text>'
        )

    def icon(self, name: str, x: float, y: float, scale: float = 1.0) -> None:
        self.parts.append(
            f'<g transform="translate({x:.1f} {y:.1f}) scale({scale})" fill="none" '
            f'stroke="{INK}" stroke-width="2.2" stroke-linecap="round" '
            f'stroke-linejoin="round">{ICONS[name]}</g>'
        )

    def _content(self, x, y, title, bullets, icon, title_size, size) -> None:
        titles = [title] if isinstance(title, str) else list(title)
        tx = x + 22
        if icon:
            self.icon(icon, x + 18, y + 20)
            tx = x + 70
        ty = y + 50
        for line in titles:
            self.text(tx, ty, line, title_size, bold=True)
            ty += title_size + 6
        ty += 8
        for bullet in bullets:
            self.text(x + 24, ty, f"• {bullet}", size)
            ty += size + 11

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        title,
        bullets=(),
        icon: str | None = None,
        title_size: float = 30,
        size: float = 22,
        r: float = 22,
    ) -> None:
        self.parts.append(f'<path d="{self._rounded(x, y, w, h, r)}" fill="{FILLS[fill]}"/>')
        self._stroke(self._rounded(x, y, w, h, r))
        self._stroke(self._rounded(x, y, w, h, r), width=1.2, opacity=0.5)
        self._content(x, y, title, bullets, icon, title_size, size)

    def label_box(
        self, x: float, y: float, w: float, h: float, fill: str, label: str, size: float = 22
    ) -> None:
        """A small box with one centred line, for lists of names."""
        self.parts.append(f'<path d="{self._rounded(x, y, w, h, 14)}" fill="{FILLS[fill]}"/>')
        self._stroke(self._rounded(x, y, w, h, 14), width=2.0)
        self.text(x + w / 2, y + h / 2 + size / 3, label, size, anchor="middle")

    def frame(self, x: float, y: float, w: float, h: float, label: str) -> None:
        self.parts.append(f'<path d="{self._rounded(x, y, w, h, 26)}" fill="{FILLS["paper"]}"/>')
        self._stroke(self._rounded(x, y, w, h, 26), width=2.0, dash="14 10")
        self.text(x + 28, y + 44, label, 30, bold=True)

    def cylinder(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        title,
        bullets=(),
        icon: str | None = None,
    ) -> None:
        rx, ry = w / 2, 20
        body = (
            f"M {x} {y + ry} L {x} {y + h - ry} A {rx} {ry} 0 0 0 {x + w} {y + h - ry} "
            f"L {x + w} {y + ry} A {rx} {ry} 0 0 0 {x} {y + ry} Z"
        )
        self.parts.append(f'<path d="{body}" fill="{FILLS[fill]}"/>')
        for width, opacity in ((2.4, 1.0), (1.2, 0.5)):
            top_ry = ry + self._j(1.2)
            self._stroke(
                f"M {x} {y + ry} A {rx} {top_ry:.1f} 0 0 1 {x + w} {y + ry} "
                f"A {rx} {top_ry:.1f} 0 0 1 {x} {y + ry}",
                width,
                opacity,
            )
            self._stroke(
                f"M {x} {y + ry} {self._seg(x, y + ry, x, y + h - ry)} "
                f"A {rx} {ry + self._j(1.2):.1f} 0 0 0 {x + w} {y + h - ry} "
                f"{self._seg(x + w, y + h - ry, x + w, y + ry)}",
                width,
                opacity,
            )
        self._content(x, y + 2 * ry - 8, title, bullets, icon, 30, 22)

    def arrow(self, *points, dash: str = "", head: float = 16) -> None:
        d = f"M {points[0][0]:.1f} {points[0][1]:.1f} " + " ".join(
            self._seg(a[0], a[1], b[0], b[1], bow=1.4)
            for a, b in zip(points, points[1:], strict=False)
        )
        self._stroke(d, dash=dash)
        (x1, y1), (x2, y2) = points[-2], points[-1]
        angle = math.atan2(y2 - y1, x2 - x1)
        for side in (-1, 1):
            a = angle + math.pi + side * 0.45
            self._stroke(
                f"M {x2:.1f} {y2:.1f} L {x2 + head * math.cos(a):.1f} {y2 + head * math.sin(a):.1f}"
            )

    def dashed_rule(self, y: float) -> None:
        self._stroke(f"M 30 {y} {self._seg(30, y, self.width - 30, y, bow=1)}", dash="20 14")

    def person(self, cx: float, top: float, label: str) -> None:
        self._stroke(
            f"M {cx + 18} {top + 18} A 18 18 0 1 1 {cx + 17.9} {top + 17} "
            f"M {cx} {top + 36} {self._seg(cx, top + 36, cx, top + 92)} "
            f"M {cx - 30} {top + 58} {self._seg(cx - 30, top + 58, cx + 30, top + 58)} "
            f"M {cx} {top + 92} L {cx - 26} {top + 134} M {cx} {top + 92} L {cx + 26} {top + 134}"
        )
        self.text(cx, top + 170, label, 26, anchor="middle")

    def heading(self, title: str, subtitle: str) -> None:
        self.text(40, 70, title, 52, bold=True)
        self.text(40, 116, subtitle, 25)

    def svg(self) -> str:
        font = base64.b64encode((HERE / "PatrickHand-Latin.ttf").read_bytes()).decode()
        style = (
            "@font-face{font-family:Hand;src:url(data:font/ttf;base64,"
            f"{font}) format('truetype');}}text{{font-family:{FONT};}}"
        )
        return "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} '
                f'{self.height}" width="{self.width}" height="{self.height}">',
                f"<style>{style}</style>",
                f'<rect width="{self.width}" height="{self.height}" fill="#FFFFFF"/>',
                *self.parts,
                "</svg>",
            ]
        )




def architecture() -> Sketch:
    """The outbound path: Supervisor, four subagents, then approval, guardrail and the mock CRM."""
    s = Sketch(2000, 1300, seed=11)
    s.heading(
        "sales-agents: the outbound path",
        "A subagent gets only a lead id. Approval and guardrail are code in front of the CRM write.",
    )
    s.box(60, 180, 380, 170, "lilac", "Lead", ["synthetic, picked by id"], "doc")
    s.box(560, 180, 600, 170, "yellow", "Supervisor", ["Deep Agent (create_deep_agent)", "delegates, owns no rules"], "layers")
    s.arrow((450, 265), (548, 265))
    s.text(1280, 230, "yellow: calls the model", 26)
    s.text(1280, 270, "green: plain code, no model", 26)
    s.text(1280, 310, "blue: shared stores", 26)

    s.frame(40, 420, 1920, 330, "Subagents (Deep Agents)")
    subagents = [
        ("Researcher", ["research_lead", "synthetic company store"], "search"),
        ("Qualifier", ["hot, warm or cold + reason", "rubric gates applied in code"], "trophy"),
        ("Writer", ["grounded first email", "save_draft"], "pencil"),
        ("CRM agent", ["upsert_lead", "record_outreach"], "box"),
    ]
    w, gap = 430, 40
    centers = []
    for i, (title, bullets, icon) in enumerate(subagents):
        x = 80 + i * (w + gap)
        s.box(x, 500, w, 210, "yellow", title, bullets, icon, size=23)
        centers.append(x + w / 2)
    for cx in centers:
        s.arrow((860, 352), (cx, 492))

    s.cylinder(80, 820, 900, 190, "blue", "LeadDesk", ["lead, company, label and draft by lead id", "subagents read the lead and hand results back"])
    s.arrow((centers[0], 712), (centers[0], 818), dash="10 8")
    s.arrow((centers[1], 712), (centers[1], 818), dash="10 8")
    s.cylinder(1020, 820, 430, 190, "blue", "Product docs", ["keyword search", "the only facts allowed"])
    s.arrow((centers[2], 712), (centers[2], 818), dash="10 8")
    s.box(1490, 820, 430, 190, "green", "Human approval", ["interrupt_on record_outreach", "approve, edit or reject"], "bubble")
    s.arrow((centers[3], 712), (centers[3], 812))

    s.frame(80, 1070, 900, 190, "Where you decide")
    for i, label in enumerate(["terminal", "web page", "Telegram"]):
        s.label_box(120 + i * 280, 1150, 240, 70, "paper", label)
    s.arrow((930, 1066), (930, 1045), (1465, 1045), (1465, 930), (1484, 930), dash="10 8", head=12)
    s.cylinder(1020, 1080, 430, 190, "blue", "Mock CRM", ["FastAPI over sqlite", "tools over MCP"])
    s.box(1490, 1080, 430, 190, "green", "Guardrail", ["opt-out line, no promises", "numbers and claims in docs"], "sparkle")
    s.arrow((1705, 1012), (1705, 1072))
    s.arrow((1488, 1175), (1462, 1175))
    return s


def layout() -> Sketch:
    """Where each part lives in the repository."""
    s = Sketch(2000, 860, seed=12)
    s.heading("Repository layout", "src/sales_agents, grouped by job. Folders outside the package sit underneath.")
    s.frame(40, 160, 1920, 520, "src/sales_agents/")
    row1 = [
        ("yellow", "agents/", ["supervisor, researcher, qualifier", "writer, replies, routing", "reactivation, rfp, baseline"], "layers"),
        ("blue", "tools/", ["crm (MCP client + gates)", "enrichment (company store)", "kb (product docs search)"], "search"),
        ("blue", "crm_mock/", ["app.py: FastAPI", "mcp_server.py: MCP tools", "store.py: sqlite"], "box"),
        ("green", "eval/", ["harness, runner, metrics", "tokens (token guard)", "report, hand_review"], "chart"),
    ]
    row2 = [
        ("lilac", "Fronts", ["cli.py, web.py", "telegram_bot.py, telegram_front.py", "runner.py (shared by web and Telegram)"], "bubble"),
        ("green", "Rules in code", ["guardrail.py, approver.py", "rubric.py, schemas.py"], "sparkle"),
        ("lilac", "Synthetic data", ["datagen, draftgen, replygen", "reactivationgen, rfpgen"], "folder"),
    ]
    w, gap = 445, 31
    for i, (fill, title, bullets, icon) in enumerate(row1):
        s.box(70 + i * (w + gap), 230, w, 200, fill, title, bullets, icon, size=22)
    w2 = (1860 - 2 * gap) / 3
    for i, (fill, title, bullets, icon) in enumerate(row2):
        s.box(70 + i * (w2 + gap), 460, w2, 190, fill, title, bullets, icon, size=22)
    for i, label in enumerate(["datasets/", "results/", "tests/", "docs/"]):
        s.label_box(70 + i * (w + gap), 730, w, 80, "paper", label, size=28)
    return s


def results_chart() -> Sketch:
    """Precision and recall from results/measurements.json, never typed by hand."""
    path = HERE.parents[1] / "results" / "measurements.json"
    rows_by_name = {r["name"]: r for r in json.loads(path.read_text(encoding="utf-8"))}

    def row(label: str, name: str, metric: str):
        m = rows_by_name[name]["metrics"]
        return (label, m[f"precision{metric}"], m[f"recall{metric}"])

    rows = [
        ("Lead qualification, hot label", None, None),
        row("Rule baseline (ceiling), n=60", "Rule baseline", " (hot)"),
        row("Qualifier (LLM), n=60", "Qualifier (LLM)", " (hot)"),
        row("Qualifier + Researcher, n=60", "Qualifier + Researcher", " (hot)"),
        row("Supervisor (subagents), n=30", "Supervisor (subagents)", " (hot)"),
        ("Lead reactivation", None, None),
        row("Selector (LLM), n=40", "Reactivation selector (LLM)", ""),
        row("Baseline, label and days, n=40", "Reactivation baseline (label and days only)", ""),
    ]
    return dot_plot(
        "Precision and recall",
        "One run each on synthetic sets that share templates with the prompts. 1.000 is not a quality claim.",
        ("precision", "recall"),
        rows,
        0.4,
        1.0,
    )


def dot_plot(title: str, subtitle: str, series: tuple[str, str], rows, lo: float, hi: float):
    """Cleveland dot plot: rows are (label, a, b), or (group heading, None, None).

    Use dots, not bars, when the axis does not start at zero: a cut bar axis exaggerates gaps.
    """
    x0, x1 = 560, 1680
    s = Sketch(2000, 300 + 62 * len(rows), seed=4)
    s.heading(title, subtitle)

    def px(v: float) -> float:
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)

    def circle(cx: float, cy: float) -> None:
        s.parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="13" fill="{FILLS["blue"]}"/>')
        s._stroke(f"M {cx + 13:.1f} {cy:.1f} A 13 13 0 1 1 {cx + 12.9:.1f} {cy - 1:.1f}")

    def square(cx: float, cy: float) -> None:
        x, y = cx - 13, cy - 13
        s.parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="26" height="26" fill="{FILLS["pink"]}"/>'
        )
        s._stroke(s._rounded(x, y, 26, 26, 3), width=2.2)

    # shape and fill both differ, so the two series read apart without colour
    circle(1180, 160)
    s.text(1206, 168, series[0], 26)
    square(1343, 160)
    s.text(1370, 168, series[1], 26)
    top = y = 215
    for label, a, b in rows:
        if a is None:
            s.text(40, y + 40, label, 28, bold=True)
            y += 62
            continue
        cy = y + 31
        s.text(60, cy + 8, label, 25)
        s._stroke(f"M {x0} {cy} {s._seg(x0, cy, x1, cy, bow=0.6)}", width=1.0, opacity=0.25)
        square(px(b), cy)  # square first: on equal values the circle still shows inside it
        circle(px(a), cy)
        s.text(1960, cy + 8, f"{a:.3f} / {b:.3f}", 24, anchor="end")
        y += 62
    ticks = 5
    for i in range(ticks + 1):
        v = lo + i * (hi - lo) / ticks
        s._stroke(f"M {px(v):.1f} {top} L {px(v):.1f} {y}", width=1.0, opacity=0.35, dash="6 8")
        s.text(px(v), y + 34, f"{v:.2f}", 22, anchor="middle")
    return s


def main() -> None:
    for name, sketch in [("architecture", architecture()), ("layout", layout()), ("results", results_chart())]:
        (HERE / f"{name}.svg").write_text(sketch.svg(), encoding="utf-8")
        print(f"wrote {name}.svg")


if __name__ == "__main__":
    main()
