"""Grounding helpers for PyMuPDF: locate cited text in a PDF and return evidence.

Library use:
    from grounding import locate, ground_citations, to_json, snippet, highlight
    from grounding import chunks_with_coordinates, chunk_metadata, chunk_images, locate_in_chunk

Command line:
    python grounding.py report.pdf --quote "Net revenue increased 12%" --page 1
    python grounding.py report.pdf --quote "within 14 days" --fuzzy
"""
import json
import re
import sys
from decimal import Decimal

import pymupdf


def locate(page, cited_text, fuzzy=False):
    """Return the quads of cited_text on a page, or [] if not found.

    fuzzy=True also lets numbers match as digits or words ("14 days" finds "fourteen days").
    """
    text = " ".join(cited_text.split())          # collapse whitespace and line breaks
    quads = page.search_for(text, quads=True)    # case-insensitive, joins hyphenated words
    if quads:
        return quads
    return _locate_by_words(page, text, fuzzy)   # fallback for small differences


def _norm(word):
    return re.sub(r"[^\w%$.]", "", word).lower().rstrip(".")


def _locate_by_words(page, text, fuzzy=False):
    """Match the quote word by word, ignoring punctuation, case and hyphenation.

    With fuzzy=True, numbers match whether written as digits or words ("14 days" finds
    "fourteen days", "twenty-one" finds "21", "12 percent" finds "12%").
    """
    target = [t for t, _ in _tokens(((w, None) for w in text.split()), fuzzy)]
    words = page.get_text("words", flags=pymupdf.TEXT_DEHYPHENATE)
    tokens = _tokens(((w[4], pymupdf.Rect(w[:4])) for w in words), fuzzy)
    n = len(target)
    if not n:
        return []
    for i in range(len(tokens) - n + 1):
        if all(tokens[i + k][0] == target[k] for k in range(n)):
            quads, seen = [], set()
            for _, rects in tokens[i:i + n]:
                for r in rects:
                    if tuple(r) not in seen:      # parts of one hyphenated word share a rect
                        seen.add(tuple(r))
                        quads.append(r.quad)
            return quads
    return []


# --- numbers written as words ----------------------------------------------

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine".split())}
_TEENS = {w: i + 10 for i, w in enumerate(
    "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: (i + 2) * 10 for i, w in enumerate(
    "twenty thirty forty fifty sixty seventy eighty ninety".split())}
_SCALES = {"thousand": 10**3, "million": 10**6, "billion": 10**9, "trillion": 10**12}
_NUMBER_WORDS = {**_UNITS, **_TEENS, **_TENS, "hundred": 100, **_SCALES}
_DIGITS = re.compile(r"^\$?\d+(\.\d+)?$")


def _split_compound(word):
    """'twentyone' (from 'twenty-one' or a line-end hyphen) -> ['twenty', 'one']."""
    for tens in _TENS:
        rest = word[len(tens):]
        if word.startswith(tens) and rest in _UNITS and rest != "zero":
            return [tens, rest]
    return [word]


def _tokens(words, fuzzy=False):
    """Normalize (text, rect) words into (token, [rects]); fuzzy turns number words into digits."""
    if not fuzzy:
        return [(t, [r] if r is not None else []) for t, r in ((_norm(w), r) for w, r in words) if t]
    out = []
    for text, rect in words:
        rects = [rect] if rect is not None else []
        parts = [_norm(p) for p in text.split("-")]
        if len(parts) > 1 and any(p in _NUMBER_WORDS or _DIGITS.match(p) for p in parts):
            pieces = parts                       # "fourteen-day" -> fourteen, day
        else:
            pieces = [_norm(text)]
        for p in pieces:
            for q in _split_compound(p) if p else []:
                out.append((q, rects))
        if out and text.rstrip(")\"'”’").endswith((",", ";", ":", ".", "!", "?")):
            out.append(("", []))                 # punctuation ends a number: "five, six"
    tokens = [t for t in _words_to_numbers(out) if t[0]]
    return _merge_percent(tokens)


def _words_to_numbers(tokens):
    """Replace each run of number words with one digit token covering all their rects."""
    out, i = [], 0
    while i < len(tokens):
        word, rects = tokens[i]
        nxt = tokens[i + 1][0] if i + 1 < len(tokens) else ""
        if _DIGITS.match(word) and (nxt in _SCALES or nxt == "hundred"):
            # "205 million" / "$4.1 billion" -> "205000000" / "$4100000000"
            sign = "$" if word.startswith("$") else ""
            value = Decimal(word.lstrip("$")) * _NUMBER_WORDS[nxt]
            text = str(int(value)) if value == value.to_integral() else str(value.normalize())
            out.append((sign + text, rects + tokens[i + 1][1]))
            i += 2
            continue
        value, end = _parse_number(tokens, i)
        if end > i:
            out.append((str(value), [r for _, rs in tokens[i:end] for r in rs]))
            i = end
        else:
            out.append(tokens[i])
            i += 1
    return out


def _parse_number(tokens, i):
    """Parse number words from tokens[i]. Returns (value, end); end == i if none.

    Only well-formed sequences are joined, so "five six" stays two numbers while
    "two hundred and five" or "three thousand forty" become one.
    """
    total = current = 0
    prev, end, j = None, i, i
    while j < len(tokens):
        w = tokens[j][0]
        if w in _UNITS or w in _TEENS:
            after_tens = prev == "tens" and w in _UNITS and w != "zero"
            if prev not in (None, "hundred", "scale", "and") and not after_tens:
                break
            current += _UNITS.get(w, _TEENS.get(w))
            prev = "unit"
        elif w in _TENS:
            if prev not in (None, "hundred", "scale", "and"):
                break
            current += _TENS[w]
            prev = "tens"
        elif w == "hundred":
            if prev != "unit" or current % 100 == 0 or current >= 100:
                break
            current *= 100
            prev = "hundred"
        elif w in _SCALES:
            if prev not in ("unit", "tens", "hundred"):
                break
            total += current * _SCALES[w]
            current = 0
            prev = "scale"
        elif w == "and" and prev in ("hundred", "scale"):
            prev = "and"                         # only kept if a number word follows
            j += 1
            continue
        else:
            break
        j += 1
        end = j
    return total + current, end


def _merge_percent(tokens):
    """'12 percent' and 'twelve per cent' -> '12%', to match a printed '12%'."""
    out, i = [], 0
    while i < len(tokens):
        word, rects = tokens[i]
        nxt = [t for t, _ in tokens[i + 1:i + 3]]
        if _DIGITS.match(word) and nxt[:1] == ["percent"]:
            out.append((word + "%", rects + tokens[i + 1][1]))
            i += 2
        elif _DIGITS.match(word) and nxt == ["per", "cent"]:
            out.append((word + "%", rects + tokens[i + 1][1] + tokens[i + 2][1]))
            i += 3
        else:
            out.append((word, rects))
            i += 1
    return out


def ground_citations(response, docs, fuzzy=False):
    """Map each page_location citation in a Claude response to quads on the page.

    response: the Message returned by client.messages.create()
    docs: the PyMuPDF documents, in the order they were sent to Claude
    fuzzy: also match numbers written as digits or words (see locate)
    """
    results = []
    for block in response.content:
        if block.type != "text" or not getattr(block, "citations", None):
            continue
        for cit in block.citations:
            if cit.type != "page_location":
                continue
            doc = docs[cit.document_index]
            # Claude's page numbers are 1-based; end_page_number is exclusive
            for pno in range(cit.start_page_number - 1, cit.end_page_number - 1):
                quads = locate(doc[pno], cit.cited_text, fuzzy)
                if quads:
                    results.append({"claim": block.text, "cited_text": cit.cited_text,
                                    "doc": cit.document_index, "page": pno, "quads": quads})
                    break
            else:
                results.append({"claim": block.text, "cited_text": cit.cited_text,
                                "doc": cit.document_index, "page": None, "quads": []})
    return results


def to_json(results):
    """Page number (1-based) and bounding box in PDF points for each citation."""
    out = []
    for r in results:
        item = {"claim": r["claim"], "cited_text": r["cited_text"], "found": bool(r["quads"])}
        if r["quads"]:
            box = pymupdf.Rect()
            for q in r["quads"]:
                box |= q.rect
            item["page_number"] = r["page"] + 1
            item["rect"] = [round(v, 1) for v in box]   # x0, y0, x1, y1
        out.append(item)
    return out


def snippet(docs, r, margin=24, dpi=150, full_width=False):
    """Render the cited region, highlighted, with some surrounding context.

    The document itself is left unchanged.
    """
    page = docs[r["doc"]][r["page"]]
    box = pymupdf.Rect()
    for q in r["quads"]:
        box |= q.rect
    clip = box + (-margin, -margin, margin, margin)
    for block in page.get_text("dict")["blocks"]:          # extend to whole lines
        for line in block.get("lines", []):
            if clip.intersects(line["bbox"]):
                clip |= line["bbox"]
    clip += (-4, -4, 4, 4)
    if full_width:
        clip.x0, clip.x1 = page.rect.x0, page.rect.x1
    clip &= page.rect
    annot = page.add_highlight_annot(r["quads"])            # temporary highlight
    pix = page.get_pixmap(clip=clip, dpi=dpi)
    page.delete_annot(annot)
    return pix


def highlight(docs, results):
    """Add a highlight annotation per located citation, with the claim as its note."""
    for r in results:
        if not r["quads"]:
            continue
        page = docs[r["doc"]][r["page"]]
        annot = page.add_highlight_annot(r["quads"])
        annot.set_info(content=r["claim"])
        annot.update()


# --- RAG chunks with page coordinates -------------------------------------

def chunks_with_coordinates(path, max_chars=1500):
    """Split a PDF into Markdown chunks that remember where each part came from.

    A new chunk starts at each section header, or when a chunk would exceed max_chars.
    Each chunk records the page number (1-based) and bbox of every layout box it contains.
    """
    doc = pymupdf.open(path)
    chunks, current = [], None
    for page_dict in doc.to_markdown(page_chunks=True):
        page_number = page_dict["metadata"]["page_number"]
        text = page_dict["text"]
        for box in page_dict["page_boxes"]:
            start, stop = box["pos"]
            box_text = text[start:stop]
            is_header = box["class"] in ("title", "section-header")
            new_section = is_header and current and current["has_body"]
            too_long = current and len(current["text"]) + len(box_text) > max_chars
            if current is None or new_section or too_long:
                current = {"text": "", "boxes": [], "has_body": False}
                chunks.append(current)
            current["text"] += box_text
            current["has_body"] |= not is_header
            current["boxes"].append({"page": page_number, "bbox": list(box["bbox"])})
    return chunks


def chunk_metadata(path, chunk):
    """Flat metadata that vector stores accept: strings and numbers only."""
    return {
        "source": path,
        "page": chunk["boxes"][0]["page"],          # first page, for filtering
        "boxes": json.dumps(chunk["boxes"]),        # every page and bbox
    }


def chunk_images(metadata, dpi=150, margin=12):
    """Render the page regions a chunk came from: one image per page it touches."""
    doc = pymupdf.open(metadata["source"])
    regions = {}
    for b in json.loads(metadata["boxes"]):
        regions.setdefault(b["page"], pymupdf.Rect())
        regions[b["page"]] |= pymupdf.Rect(b["bbox"])
    images = []
    for page_number, rect in regions.items():
        page = doc[page_number - 1]
        clip = (rect + (-margin, -margin, margin, margin)) & page.rect
        images.append(page.get_pixmap(clip=clip, dpi=dpi))
    return images


def locate_in_chunk(doc, metadata, quote):
    """Find a quote inside a retrieved chunk's regions only. Returns (page_number, quads)."""
    for b in json.loads(metadata["boxes"]):
        page = doc[b["page"] - 1]
        quads = page.search_for(quote, clip=pymupdf.Rect(b["bbox"]), quads=True)
        if quads:
            return b["page"], quads
    return None, []


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Locate a quote in a PDF and print its coordinates.")
    ap.add_argument("pdf")
    ap.add_argument("--quote", required=True, help="text to locate")
    ap.add_argument("--page", type=int, help="1-based page to search; default: all pages")
    ap.add_argument("--snippet", help="save an evidence image to this PNG path")
    ap.add_argument("--fuzzy", action="store_true",
                    help='match numbers as digits or words ("14 days" finds "fourteen days")')
    args = ap.parse_args(argv)

    doc = pymupdf.open(args.pdf)
    pages = [args.page - 1] if args.page else range(doc.page_count)
    result = {"claim": None, "cited_text": args.quote, "doc": 0, "page": None, "quads": []}
    for pno in pages:
        quads = locate(doc[pno], args.quote, fuzzy=args.fuzzy)
        if quads:
            result.update(page=pno, quads=quads)
            break
    print(json.dumps(to_json([result])[0], indent=2))
    if args.snippet and result["quads"]:
        snippet([doc], result).save(args.snippet)
    return 0 if result["quads"] else 1


if __name__ == "__main__":
    sys.exit(main())
