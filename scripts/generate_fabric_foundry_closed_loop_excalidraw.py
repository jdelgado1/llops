#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("docs/fabric-foundry-closed-loop.excalidraw")

_elements = []
_counter = 1


def eid(prefix: str) -> str:
    global _counter
    value = f"{prefix}-{_counter:04d}"
    _counter += 1
    return value


def base(el_type: str, x: float, y: float, w: float, h: float, stroke: str, bg: str = "transparent") -> dict:
    return {
        "id": eid(el_type),
        "type": el_type,
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "angle": 0,
        "strokeColor": stroke,
        "backgroundColor": bg,
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": {"type": 3} if el_type == "rectangle" else None,
        "seed": 100000 + _counter,
        "version": 1,
        "versionNonce": 200000 + _counter,
        "isDeleted": False,
        "boundElements": [],
        "updated": 1,
        "link": None,
        "locked": False,
    }


def rect(x, y, w, h, stroke="#374151", bg="#ffffff", opacity=100, sw=2):
    el = base("rectangle", x, y, w, h, stroke, bg)
    el["opacity"] = opacity
    el["strokeWidth"] = sw
    _elements.append(el)
    return el["id"]


def text(x, y, body, size=20, color="#111827", w=None, align="left", bold=False):
    lines = body.split("\n")
    if w is None:
        w = max(220, min(620, max(len(line) for line in lines) * size * 0.55))
    h = len(lines) * size * 1.25
    el = base("text", x, y, w, h, color, "transparent")
    el.update({
        "strokeWidth": 1,
        "roundness": None,
        "boundElements": None,
        "text": body,
        "fontSize": size,
        "fontFamily": 2,
        "textAlign": align,
        "verticalAlign": "top",
        "containerId": None,
        "originalText": body,
        "lineHeight": 1.25,
        "baseline": max(size, h - 5),
    })
    if bold:
        # Excalidraw has no explicit bold property in this JSON flavor; larger size/color carries hierarchy.
        pass
    _elements.append(el)
    return el["id"]


def arrow(x1, y1, x2, y2, color="#374151", width=2, dashed=False, label=None, label_dx=0, label_dy=0):
    el = base("arrow", x1, y1, abs(x2 - x1), abs(y2 - y1), color, "transparent")
    el.update({
        "roundness": {"type": 2},
        "strokeWidth": width,
        "strokeStyle": "dashed" if dashed else "solid",
        "points": [[0, 0], [x2 - x1, y2 - y1]],
        "lastCommittedPoint": None,
        "startBinding": None,
        "endBinding": None,
        "startArrowhead": None,
        "endArrowhead": "arrow",
    })
    _elements.append(el)
    if label:
        text((x1 + x2) / 2 + label_dx, (y1 + y2) / 2 + label_dy, label, size=16, color=color, w=280, align="center")
    return el["id"]


# Title
text(250, 28, "Fabric + Foundry LLMOps Closed Loop", size=44, color="#1f2937", w=980, align="center")
text(360, 82, "Teacher-gold evaluation, Fabric data spine, retraining export, and promotion governance", size=18, color="#4b5563", w=760, align="center")

# Swimlanes
rect(40, 135, 330, 760, "#2563eb", "#dbeafe", opacity=32)
rect(400, 135, 610, 760, "#ea580c", "#ffedd5", opacity=30)
rect(1040, 135, 330, 760, "#9333ea", "#f3e8ff", opacity=30)
rect(1400, 135, 330, 760, "#16a34a", "#dcfce7", opacity=30)

text(78, 155, "AZURE AI FOUNDRY", size=28, color="#1d4ed8", w=260, align="center")
text(550, 155, "MICROSOFT FABRIC LAKEHOUSE", size=28, color="#c2410c", w=360, align="center")
text(1090, 155, "POWER BI + HTML", size=28, color="#7e22ce", w=240, align="center")
text(1445, 155, "DATA AGENT", size=28, color="#15803d", w=230, align="center")

# Foundry blocks
rect(70, 210, 270, 118, "#1d4ed8", "#eff6ff")
text(90, 226, "1. Model execution\n• gpt-5.4 teacher\n• gpt-4.1-nano students\n• tool-call outputs", 18, "#111827", 230)

rect(70, 370, 270, 138, "#1d4ed8", "#eff6ff")
text(90, 386, "2. Foundry produces artifacts\n• eval_results\n• eval_details\n• traces\n• model outputs", 18, "#111827", 230)

rect(70, 690, 270, 132, "#1d4ed8", "#eff6ff")
text(90, 706, "7. Next fine-tune\nGlobalStandard SFT\nBase = previous student\nDeploy candidate", 18, "#111827", 230)

# Fabric spine main layers
rect(435, 210, 540, 138, "#b45309", "#fff7ed")
text(455, 224, "BRONZE: raw ingestion", 22, "#92400e", 300)
text(455, 260, "• raw traces\n• raw Foundry eval artifacts\n• raw model outputs\nFiles/llmops/raw/...", 17, "#111827", 480)

rect(435, 390, 540, 156, "#b45309", "#fff7ed")
text(455, 404, "SILVER: normalized model/eval data", 22, "#92400e", 400)
text(455, 440, "• model versions\n• eval runs\n• eval results\n• golden/reference examples\n• row-count contracts + manifests", 17, "#111827", 490)

rect(435, 590, 540, 190, "#b45309", "#fff7ed")
text(455, 604, "GOLD: decision and retraining layer", 22, "#92400e", 420)
text(455, 640, "• model quality\n• promotion status\n• trace performance\n• training cost estimates\n• retraining candidates\n• dashboard-ready tables", 17, "#111827", 490)

rect(505, 815, 400, 54, "#c2410c", "#fed7aa")
text(520, 828, "Files/llmops/foundry_exports/dev3-student-v3/train.jsonl", 18, "#7c2d12", 370, align="center")

# Reporting blocks
rect(1070, 245, 270, 140, "#7e22ce", "#faf5ff")
text(1090, 262, "Executive scorecard", 22, "#6b21a8", 230)
text(1090, 298, "Power BI visuals\nAccuracy trend\nPromotion status\nCost / latency", 17, "#111827", 230)

rect(1070, 450, 270, 132, "#7e22ce", "#faf5ff")
text(1090, 468, "HTML scorecard", 22, "#6b21a8", 230)
text(1090, 504, "Portable static summary\nLoop worked\nCandidate rejected\nNext action", 17, "#111827", 230)

# Data agent
rect(1430, 260, 260, 200, "#15803d", "#f0fdf4")
text(1450, 278, "Data Agent", 24, "#166534", 220)
text(1450, 320, "Uses semantic model to answer:\n• Which run improved?\n• Why rejected?\n• Cost and p95?\n• Which examples failed?", 17, "#111827", 220)

rect(1430, 540, 260, 140, "#15803d", "#f0fdf4")
text(1450, 558, "Semantic model", 22, "#166534", 220)
text(1450, 594, "Tables/views over gold + reporting layers\nNo raw-file joins in reports", 17, "#111827", 220)

# Notes
rect(410, 925, 610, 105, "#dc2626", "#fef2f2")
text(435, 942, "Governance note", 24, "#b91c1c", 260)
text(435, 982, "Golden dataset is versioned and reviewed. Fabric can generate retraining candidates, but it does not silently mutate the official golden dataset.", 17, "#111827", 560)

rect(1040, 925, 630, 105, "#334155", "#f8fafc")
text(1065, 942, "Decision note", 24, "#1f2937", 240)
text(1065, 982, "Student-v2 improved from 0/10 to 1/10 on the hard slice, but is rejected for promotion. Student-v3 is queued from Fabric retraining candidates.", 17, "#111827", 580)

# Arrows and labels
arrow(340, 270, 430, 270, "#dc2626", 3, False, "Foundry -> Fabric\nraw artifacts", 0, -44)
arrow(340, 435, 430, 270, "#dc2626", 3)
arrow(705, 348, 705, 390, "#c2410c", 3)
arrow(705, 546, 705, 590, "#c2410c", 3)
arrow(975, 685, 1070, 315, "#9333ea", 3, False, "reporting\nconsumers", 12, -20)
arrow(975, 685, 1070, 515, "#9333ea", 3)
arrow(975, 685, 1430, 610, "#16a34a", 3, False, "semantic model\nquestions", 40, -55)
arrow(1430, 360, 1340, 315, "#16a34a", 2, True)
arrow(705, 780, 705, 812, "#c2410c", 3)
arrow(505, 842, 340, 755, "#dc2626", 3, False, "Fabric -> Foundry\nnext train.jsonl", -85, -60)
arrow(205, 690, 205, 510, "#1d4ed8", 2, False)
arrow(205, 328, 205, 370, "#1d4ed8", 2)
arrow(205, 822, 480, 700, "#059669", 2, True, "evaluate candidate\nthen gate", 0, 22)

# Footer
text(70, 1048, "Red arrows mark the cross-system handoffs. Fabric is the data spine; Foundry remains the model execution and fine-tuning layer.", 18, "#374151", 1260, align="center")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({
    "type": "excalidraw",
    "version": 2,
    "source": "https://excalidraw.com",
    "elements": _elements,
    "appState": {
        "theme": "light",
        "viewBackgroundColor": "#ffffff",
        "currentItemStrokeColor": "#111827",
        "currentItemBackgroundColor": "transparent",
        "currentItemFillStyle": "solid",
        "currentItemStrokeWidth": 2,
        "currentItemStrokeStyle": "solid",
        "currentItemRoughness": 0,
        "currentItemOpacity": 100,
        "currentItemFontFamily": 2,
        "currentItemFontSize": 20,
        "currentItemTextAlign": "left",
        "currentItemStartArrowhead": None,
        "currentItemEndArrowhead": "arrow",
        "scrollX": 0,
        "scrollY": 0,
        "zoom": {"value": 0.65},
        "gridSize": 20,
    },
    "files": {},
}, indent=2), encoding="utf-8")
print(f"Wrote {OUT} with {len(_elements)} elements")
