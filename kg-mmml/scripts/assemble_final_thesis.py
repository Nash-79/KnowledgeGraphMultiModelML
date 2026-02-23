from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
THESIS_DIR = ROOT / "docs" / "thesis"
OUT_MD = THESIS_DIR / "FINAL_THESIS_MANUSCRIPT.md"


SECTIONS = [
    ("Abstract", "Abstract.md"),
    ("Chapter 1: Introduction", "Chapter_1_Introduction.md"),
    ("Chapter 2: Literature Review", "Chapter_2_Literature_Review.md"),
    ("Chapter 3: Methodology", "Chapter_3_Methodology.md"),
    ("Chapter 4: Implementation", "Chapter_4_Implementation.md"),
    ("Chapter 5: Results", "Chapter_5_Results.md"),
    ("Chapter 6: Discussion", "Chapter_6_Discussion.md"),
    ("Chapter 7: Conclusion", "Chapter_7_Conclusion.md"),
    ("References", "References.md"),
    ("Appendix A: Code Listings", "Appendix_A_Code_Listings.md"),
    ("Appendix B: Metric Tables", "Appendix_B_Metric_Tables.md"),
    ("Appendix C: Decision Gates", "Appendix_C_Decision_Gates.md"),
    ("Appendix D: Reproducibility", "Appendix_D_Reproducibility.md"),
    ("Appendix E: Citation-to-Claim Trace", "Appendix_E_Citation_to_Claim_Trace.md"),
]


MOJIBAKE_FIXES = {
    "â†’": "->",
    "â‰¥": ">=",
    "â‰¤": "<=",
    "â‰ˆ": "~=",
    "âˆ’": "-",
    "â€“": "-",
    "â€”": "-",
    "â€˜": "'",
    "â€™": "'",
    "â€œ": '"',
    "â€": '"',
    "â€¦": "...",
    "Ã—": "x",
    "Â": "",
    "Å¡": "s",
    "Ä": "c",
    "Ä‡": "c",
    "Ã¡": "a",
}


def clean_text(text: str) -> str:
    cleaned = text
    for bad, good in MOJIBAKE_FIXES.items():
        cleaned = cleaned.replace(bad, good)
    return cleaned


def strip_top_heading(text: str, fallback: str) -> str:
    lines = text.splitlines()
    if not lines:
        return f"# {fallback}\n"

    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and lines[i].lstrip().startswith("#"):
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        body = "\n".join(lines[i:]).strip()
        return body

    return text.strip()


def build_document() -> str:
    parts: list[str] = []
    parts.append("# Integrating Knowledge Graphs with Multi-Modal Machine Learning")
    parts.append("")
    parts.append("## Front Matter")
    parts.append("")
    parts.append("- Student Name: Naresh Mepani")
    parts.append("- Student ID: 24026935")
    parts.append("- Degree: MSc Project (Research-focused)")
    parts.append("- Date: 2026-02-09")
    parts.append("")
    parts.append("## Note")
    parts.append("")
    parts.append("This is the assembled master manuscript from thesis chapter sources.")
    parts.append("Use this file as the single source for Word/PDF final formatting.")
    parts.append("")
    parts.append("---")
    parts.append("")

    for title, rel in SECTIONS:
        src = THESIS_DIR / rel
        if not src.exists():
            parts.append(f"## {title}")
            parts.append("")
            parts.append(f"[Missing source file: {rel}]")
            parts.append("")
            parts.append("---")
            parts.append("")
            continue

        raw = src.read_text(encoding="utf-8", errors="ignore")
        raw = clean_text(raw)
        body = strip_top_heading(raw, title)

        parts.append(f"## {title}")
        parts.append("")
        parts.append(body)
        parts.append("")
        parts.append("---")
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    manuscript = build_document()
    OUT_MD.write_text(manuscript, encoding="utf-8")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
