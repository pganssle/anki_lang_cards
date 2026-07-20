"""Generate a side-by-side HTML comparison of two (or more) output CSVs.

This is for comparing the sentences different models (or prompts) generate
for the same words: run the pipeline once per configuration with different
``--output`` paths, then point this script at the resulting CSVs. Words
present in every input file are shown next to each other, one column per
file; sentences that appear in every file are dimmed so the differences
stand out.

With JavaScript enabled the page shows one word at a time with prev/next
controls (and a button to switch to the full table); without JavaScript it
degrades to the full table.
"""

import argparse
import html
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Final

from anki_lang_cards import _types as types

_STYLE: Final[str] = """\
body { font-family: sans-serif; max-width: 75rem; margin: 1rem auto; padding: 0 1rem; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ccc; vertical-align: top; padding: 0.5rem; text-align: left; }
.meta p { margin: 0; }
.meta .headword { font-weight: bold; }
.model { color: #888; font-size: 0.8em; margin: 0 0 0.4rem; }
.sentence { margin: 0 0 0.6rem; }
.sentence .es { margin: 0; }
.sentence .en { margin: 0; color: #555; font-size: 0.9em; }
.sentence.shared .es, .sentence.shared .en { color: #bbb; }
.controls { display: none; gap: 0.5rem; align-items: center; margin: 0.5rem 0;
            position: sticky; top: 0; background: #fff; padding: 0.5rem 0; }
.cycle-mode .controls { display: flex; }
.cycle-mode tbody tr { display: none; }
.cycle-mode tbody tr.current { display: table-row; }
"""

_SCRIPT: Final[str] = """\
(function () {
    var body = document.body;
    var rows = Array.prototype.slice.call(document.querySelectorAll("tbody tr"));
    var counter = document.getElementById("counter");
    var toggle = document.getElementById("toggle");
    var idx = 0;

    function update() {
        rows.forEach(function (row, i) {
            row.classList.toggle("current", i === idx);
        });
        counter.textContent = (idx + 1) + " / " + rows.length;
    }

    function step(delta) {
        idx = (idx + delta + rows.length) % rows.length;
        update();
    }

    document.getElementById("prev").addEventListener("click", function () { step(-1); });
    document.getElementById("next").addEventListener("click", function () { step(1); });
    toggle.addEventListener("click", function () {
        var cycling = body.classList.toggle("cycle-mode");
        toggle.textContent = cycling ? "Table view" : "One word at a time";
    });
    document.addEventListener("keydown", function (event) {
        if (!body.classList.contains("cycle-mode")) { return; }
        if (event.key === "ArrowLeft") { step(-1); }
        if (event.key === "ArrowRight") { step(1); }
    });

    body.classList.add("cycle-mode");
    update();
})();
"""


def load_words(path: pathlib.Path) -> Mapping[str, types.Word]:
    """Load a pipeline CSV, keyed by headword (first occurrence wins)."""
    words: dict[str, types.Word] = {}
    for word in types.Word.words_from_csv(path):
        words.setdefault(word.word, word)
    return words


def common_headwords(word_maps: Sequence[Mapping[str, types.Word]]) -> Sequence[str]:
    """Headwords present in every file, in the order of the first file."""
    return tuple(
        headword
        for headword in word_maps[0]
        if all(headword in word_map for word_map in word_maps[1:])
    )


def _render_cell(word: types.Word, shared_spanish: frozenset[str]) -> str:
    parts: list[str] = []
    if word.model_used:
        parts.append(f'<p class="model">{html.escape(word.model_used)}</p>')
    for sentence in word.sentences:
        shared = " shared" if sentence.spanish in shared_spanish else ""
        parts.append(
            f'<div class="sentence{shared}">'
            f'<p class="es">{html.escape(sentence.spanish)}</p>'
            f'<p class="en">{html.escape(sentence.english)}</p>'
            f"</div>"
        )
    return f"<td>{''.join(parts)}</td>"


def _render_row(headword: str, words: Sequence[types.Word]) -> str:
    shared_spanish = frozenset.intersection(
        *(frozenset(s.spanish for s in word.sentences) for word in words)
    )
    exemplar = words[0]
    meta = (
        f'<td class="meta"><p class="headword">{html.escape(headword)}</p>'
        f"<p>{html.escape(str(exemplar.part_of_speech))}</p>"
        f"<p>{html.escape(exemplar.meaning)}</p></td>"
    )
    cells = "".join(_render_cell(word, shared_spanish) for word in words)
    return f"<tr>{meta}{cells}</tr>"


def render_comparison(
    word_maps: Sequence[Mapping[str, types.Word]],
    headwords: Sequence[str],
    labels: Sequence[str],
) -> str:
    header_cells = "".join(
        f"<th>{html.escape(label)}</th>" for label in ("Word", *labels)
    )
    rows = "\n".join(
        _render_row(headword, [word_map[headword] for word_map in word_maps])
        for headword in headwords
    )
    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Model output comparison</title>
<style>
{_STYLE}
</style>
</head>
<body>
<div class="controls">
<button id="prev" type="button">&#9664;</button>
<span id="counter"></span>
<button id="next" type="button">&#9654;</button>
<button id="toggle" type="button">Table view</button>
</div>
<table>
<thead><tr>{header_cells}</tr></thead>
<tbody>
{rows}
</tbody>
</table>
<script>
{_SCRIPT}
</script>
</body>
</html>
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "files",
        type=pathlib.Path,
        nargs="+",
        help="Two or more output CSVs to compare.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Column labels for the files; defaults to the file paths.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("output/model_comparison.html"),
        help="Path of the HTML file to write.",
    )
    args = parser.parse_args(argv)

    if len(args.files) < 2:
        parser.error("at least two files are required")
    labels: Sequence[str] = args.labels or [str(path) for path in args.files]
    if len(labels) != len(args.files):
        parser.error("--labels must match the number of files")

    word_maps = [load_words(path) for path in args.files]
    headwords = common_headwords(word_maps)
    if not headwords:
        print("No words appear in all of the input files.", file=sys.stderr)
        return 1

    all_headwords = set().union(*word_maps)
    skipped = sorted(all_headwords - set(headwords))
    print(f"{len(headwords)} words common to all files.")
    if skipped:
        print(f"Skipped {len(skipped)} words not present in every file:")
        for headword in skipped:
            print(f"  {headword}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_comparison(word_maps, headwords, labels), encoding="utf-8"
    )
    print(f"Written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
