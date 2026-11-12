"""Grounding helpers for PyMuPDF: locate cited text in a PDF and return evidence.

Library use:
    from grounding import locate, ground_citations, to_json, snippet, highlight
    from grounding import chunks_with_coordinates, chunk_metadata, chunk_images, locate_in_chunk

Command line:
    python grounding.py report.pdf --quote "Net revenue increased 12%" --page 1
"""
import json
import re
import sys

import pymupdf


def locate(page, cited_text):
    """Return the quads of cited_text on a page, or [] if not found."""
    text = " ".join(cited_text.split())          # collapse whitespace and line breaks
    quads = page.search_for(text, quads=True)    # case-insensitive, joins hyphenated words
    if quads:
        return quads
    return _locate_by_words(page, text)          # fallback for small differences


def _norm(word):
    return re.sub(r"[^\w%$.]", "", word).lower().rstrip(".")


def _locate_by_words(page, text):
    """Match the quote word by word, ignoring punctuation, case and hyphenation."""
    target = [_norm(w) for w in text.split() if _norm(w)]
    words = page.get_text("words", flags=pymupdf.TEXT_DEHYPHENATE)
    tokens = [(_norm(w[4]), pymupdf.Rect(w[:4])) for w in words]
    tokens = [t for t in tokens if t[0]]
    n = len(target)
    for i in range(len(tokens) - n + 1):
        if all(tokens[i + k][0] == target[k] for k in range(n)):
            return [r.quad for _, r in tokens[i:i + n]]
    return []


def ground_citations(response, docs):
    """Map each page_location citation in a Claude response to quads on the page.

    response: the Message returned by client.messages.create()
    docs: the PyMuPDF documents, in the order they were sent to Claude
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
                quads = locate(doc[pno], cit.cited_text)
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
    args = ap.parse_args(argv)

    doc = pymupdf.open(args.pdf)
    pages = [args.page - 1] if args.page else range(doc.page_count)
    result = {"claim": None, "cited_text": args.quote, "doc": 0, "page": None, "quads": []}
    for pno in pages:
        quads = locate(doc[pno], args.quote)
        if quads:
            result.update(page=pno, quads=quads)
            break
    print(json.dumps(to_json([result])[0], indent=2))
    if args.snippet and result["quads"]:
        snippet([doc], result).save(args.snippet)
    return 0 if result["quads"] else 1


if __name__ == "__main__":
    sys.exit(main())
