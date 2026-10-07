# Golden evaluation set

`golden.jsonl` holds questions about the indexed regulations, each labelled with the
document sections that answer it. Retrieval quality (recall@k, MRR) is measured
against it, so **errors in this file become errors in every metric**. It is curated
by hand and versioned in git.

## Record format (one JSON object per line)

```json
{
  "id": "q017",
  "question": "Is there a required minimum insulation resistance for trolleybus circuits?",
  "lang": "en",
  "type": "numeric",
  "gold": [{"doc_id": "unece-r107-rev9", "scope": "Annex 12", "section": "3.9"}],
  "answer": "Yes: at least 10 MΩ for each basic, supplementary and overall double insulation, tested at 1,000 V DC.",
  "source": "claude-session-2",
  "status": "draft",
  "notes": ""
}
```

| Field | Meaning |
|---|---|
| `id` | Unique, `q001`, `q002`, … (human-written questions: `h001`, …) |
| `question` | What a real user would type. See the writing rules below. |
| `lang` | `en` or `ru` |
| `type` | One of the types below |
| `gold` | Sections that answer the question. Empty list `[]` for `unanswerable`. |
| `gold[].doc_id` | Document id, e.g. `unece-r107-rev9` |
| `gold[].scope` | Exactly as in `blocks.jsonl`: `Regulation`, `Annex 3`, `Annex 1 - Part 1 - Appendix 2`, … |
| `gold[].section` | Section number as in `blocks.jsonl` `section_number`, e.g. `7.6.1.1`. Use the **most specific** section(s) containing the answer. For unnumbered content (e.g. Annex 4 diagrams) use `null` and give `page` instead. |
| `gold[].page` | Optional; required when `section` is `null`. |
| `answer` | Short reference answer **in your own words**, with values and units. `null` for `unanswerable`. |
| `source` | Who wrote it: `human`, `claude-session-2`, … |
| `status` | `draft` → `verified` / `rejected`. Only `verified` items count in metrics. |
| `notes` | Anything a reviewer should know (ambiguities, why a question is unanswerable). |

There is no `split` field: the evaluation tool assigns dev/test (≈70/30)
deterministically from the `id`, so authors cannot influence it.

### How gold matching works

A retrieved chunk counts as relevant if it has the same `doc_id` and `scope` and
covers the gold section **or any of its subsections** (gold `3.9` matches a chunk
covering `3.9.1`). For `section: null`, the chunk must be in the same scope and
include the given page. Label the section that actually *contains* the answer,
not its parent: gold `7` would match half of Annex 3.

If the answer needs **several** sections (comparison, cross-reference), list them all.

## Question types (target mix for ~50 questions)

| type | ~count | What it tests |
|---|---|---|
| `numeric` | 12 | A specific value, dimension, force, time, count |
| `definition` | 6 | What a term means in the regulation |
| `applicability` | 6 | Whether / to which vehicles a requirement applies; exemptions |
| `comparison` | 5 | Differences between classes, categories or cases; needs ≥2 sections |
| `cross_reference` | 5 | The answer needs a section **and** something it refers to elsewhere ("see Annex X") |
| `vocabulary_mismatch` | 6 | Asked in everyday/engineering words the regulation does **not** use |
| `unanswerable` | 8 | Plausible bus/vehicle question the documents do **not** answer |

About **20% of all questions in Russian** (`lang: "ru"`), spread across types.
Spread questions across scopes in proportion to their substance: the main
Regulation, Annex 3 (largest), Annex 8, Annex 11, Annex 12, Annex 13, Annex 6/7.
Annex 1 consists of blank model forms: at most 1–2 questions there.

## Writing rules (to keep the set honest)

1. **Write like a user, not like the document.** Imagine an engineer at a bus
   manufacturer asking a colleague. Do not copy any phrase of 3+ consecutive
   content words from the gold text; do not mention section or paragraph numbers.
   The tooling measures word overlap with the gold text and flags high values.
2. **One clear intent per question.** It must be possible to say whether a passage answers it.
3. **The gold must really contain the answer.** Verify by reading the text; do not
   label from section titles alone.
4. **Unanswerable questions must be plausible**: realistic for the domain, not
   trivia. Verify the absence by searching the document text for the key terms,
   and say in `notes` what you checked.
5. **Do not consult the retrieval system** while writing (no `scripts/search.py`,
   no Qdrant, no `chunks.jsonl`, no code under `src/regrag/retrieval/`). Read the
   document from `data/processed/<doc_id>/blocks.jsonl` (fields `scope`,
   `section_number`, `text`, `page_number`, `role`; use `role == "body"`) or the PDF.
   Questions written while looking at system output end up tuned to the system.

## Where the source text is

- `data/processed/unece-r107-rev9/blocks.jsonl`: parsed UN Regulation No. 107
  (Revision 9), one block per line, in document order.
- `data/raw/R107r9e.pdf`: the original PDF (tables and figures are easier to read here).
