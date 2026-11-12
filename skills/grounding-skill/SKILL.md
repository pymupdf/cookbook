---
name: pymupdf-grounding
description: Ground AI answers in PDF documents with PyMuPDF. Locate cited or quoted text on the page, keep page coordinates with RAG chunks, return exact PDF coordinates, and render evidence snippets or highlighted PDFs. Use when an LLM answer cites a PDF (for example Claude citations with page_location), or when building a RAG pipeline over PDFs that must show, verify or store where each claim comes from.
license: Skill content MIT. PyMuPDF itself is dual-licensed (AGPL-3.0 or commercial license from Artifex).
metadata:
  version: "0.2"
  requires: "pymupdf>=2.1"
  maintainer: "Artifex Software"
---

# Grounding AI answers in PDFs with PyMuPDF

Grounding links each claim in an AI answer to the exact place in a source document that supports it. PyMuPDF reads the position of every character in a PDF, so it can turn a quoted passage into exact page coordinates without any extra model.

The core result is always **a page number and a rectangle in PDF coordinates**. Images, highlights and web overlays are ways of displaying that result.

This skill grows with the PyMuPDF grounding article series. Version 0.2 covers grounding LLM citations and keeping page coordinates with RAG chunks, for born-digital PDFs.

## Setup

```bash
pip install "pymupdf>=2.1"
```

Tested helpers are in `scripts/grounding.py`. Copy the functions you need rather than rewriting them:

| Function | Purpose |
|---|---|
| `locate(page, text)` | Quads of the text on one page, or `[]` |
| `ground_citations(response, docs)` | Locate every `page_location` citation in a Claude response |
| `to_json(results)` | Page number and rect in PDF points for each citation |
| `snippet(docs, r, margin=24)` | Image of the cited region with surrounding lines; leaves the document unchanged |
| `highlight(docs, results)` | Highlight annotations with the claim as the note, for a saved copy |
| `chunks_with_coordinates(path)` | Markdown chunks split at section headers, each with the page and bbox of every layout box |
| `chunk_metadata(path, chunk)` | Flat metadata for vector stores: `source`, first `page`, `boxes` as a JSON string |
| `chunk_images(metadata)` | Images of the page regions a retrieved chunk came from |
| `locate_in_chunk(doc, metadata, quote)` | Find a quote only inside a retrieved chunk's regions |

Quick check from the command line (exit code 1 when the quote is not found):

```bash
python scripts/grounding.py report.pdf --quote "Net revenue increased 12%" --page 1 --snippet evidence.png
```

## Workflow A: ground LLM citations

1. **Get the citations.** With Claude, send the PDF as a `document` block with `"citations": {"enabled": true}`. Each PDF citation has `cited_text`, `document_index`, `start_page_number` (1-based) and `end_page_number` (exclusive). For other models, any quote plus a page hint works: call `locate()` directly.
2. **Locate each quote.** `locate()` uses `Page.search_for(text, quads=True)`, which is case-insensitive, matches across line breaks and rejoins words hyphenated at line ends. If that finds nothing, it falls back to a word-by-word match that ignores punctuation and case.
3. **Return coordinates.** `to_json()` gives `page_number` (1-based) and `rect` `[x0, y0, x1, y1]` in PDF points. Store this with the answer as the audit record.
4. **Show the evidence (optional).** Use `snippet()` for chat and review screens, or `highlight()` plus `doc.save(new_path)` for a file reviewers can open anywhere.

## Workflow B: RAG chunks with page coordinates

1. **Chunk from layout boxes, not from plain text.** `doc.to_markdown(page_chunks=True)` returns `page_boxes` for each page; each box has `class`, `bbox` and `pos`, and `text[pos[0]:pos[1]]` is the Markdown from that box. `chunks_with_coordinates()` builds chunks from these boxes. A generic text splitter loses the positions.
2. **Store coordinates as metadata.** Use `chunk_metadata()`. Vector stores need flat metadata, so the boxes are a JSON string.
   - LangChain: `Document(page_content=c["text"], metadata=chunk_metadata(path, c))`.
   - LlamaIndex: `TextNode(text=c["text"], metadata=..., excluded_embed_metadata_keys=["boxes"], excluded_llm_metadata_keys=["boxes"])`, so coordinates stay out of embeddings and prompts.
3. **Show the source of a retrieved chunk** with `chunk_images(hit.metadata)`, or send page numbers and boxes to the front end.
4. **Narrow to the exact sentence** with `locate_in_chunk()` when the answer quotes the chunk.

The PyMuPDF loaders for LangChain (`langchain-pymupdf4llm`) and LlamaIndex (`LlamaMarkdownReader`) do not pass `page_boxes` through today. Build chunks with `chunks_with_coordinates()` instead.

## Coordinates

- Units are points: 72 per inch.
- PyMuPDF measures from the top-left corner of the unrotated page.
- `page_boxes` bboxes are rounded to whole points; quads from `search_for()` are not.
- The PDF file format measures from the bottom-left. Convert with `rect * page.transformation_matrix` when another tool expects that.
- To draw on a page image rendered at `dpi`, multiply the values by `dpi / 72`.
- PyMuPDF page numbers are 0-based; Claude's and the JSON output's are 1-based.

## Rules

- Never present a citation as evidence when it was not found. Report `"found": false` and flag it for review.
- Do not save over the user's original PDF. Write highlighted output to a new file.
- The PDF needs a text layer. Scanned pages need OCR first, and Claude cannot cite them without one.
- If the same text appears more than once on a page, `search_for()` returns every occurrence. Check the match count when that matters.
- Paraphrased quotes are not matched. Do not loosen matching to force a result.
- `page_boxes` positions are per layout box (paragraph, heading, list item, table). Use `locate_in_chunk()` or `locate()` for sentence-level positions.

## Guides

Each article in the series has a runnable notebook in the PyMuPDF cookbook repository.

- Cite Claude's answers at the exact spot in the PDF. [link to be added]
- Keep page coordinates with your RAG chunks. [link to be added]

## License

PyMuPDF is available under the GNU AGPL-3.0 or a commercial license from Artifex. Applications that include PyMuPDF must follow the AGPL unless they have a commercial license. See https://artifex.com/licensing.
