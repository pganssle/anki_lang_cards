# Reviewing word rows

What to check in each exported row, roughly in order of how often each has
gone wrong. The examples are real misses from earlier runs.

The rows come from `scripts/review_words.py export`. Each has the headword,
part of speech, meaning, article, its sentences (with where each came from:
`book: ...` means the model took it from the user's reading, `model: ...`
means generated), `csv_matches` (other rows in this run that are probably
the same word), `deck_matches` (notes already in the deck that probably are)
and, for the word CSV, `contexts`: the passages from the book the
highlights came from. The passage is the ground truth for the sense used.
A sentence with no source at all usually means the model cited a passage
that does not exist; judge the row on its own merits, and treat those
sentences like generated ones.

## The headword

The headword should be the dictionary form of the word *as it is used in the
sentences*, since that is what the card teaches.

- **Reflexive verbs.** If the sense only exists pronominally, the headword
  has the `se`: `irse` ("me voy a ir"), `mesarse` (to tear at one's hair —
  `mesar` alone was once defined as "to stroke", the opposite). If the base
  verb and the reflexive mean different things, they are different cards;
  if they mean the same (`compadecer`/`compadecerse`), they are one.
- **Gendered nouns and adjectives** show both forms compactly: `seco/a`,
  `español/a`, `mirón/ona`. Two rows `mirón/a` and `mirón/ona` are the same
  word written two ways; make them agree.
- **Notes and qualifiers** go in one parenthetical after the headword:
  `fiambre (comida)`, `sortear (obstáculo)`.
- **Phrases.** If the word only appears inside a fixed expression, the
  expression is the headword: `de tapadillo` rather than a noun `tapadillo`
  whose definition admits it is only used in that phrase; `hincar el diente
  (a)`. Conversely, an ellipsis from one sentence is not an idiom (`bastante
  con que`).
- **Typos and accents**: `apañárselas`, not `apañarselas`.
- **Plurals and inflected forms.** The headword is the singular and the
  infinitive, even when the highlight was `gatos` or `vayamos`.

## Part of speech and article

Valid codes are the ones the prompts list: `nm`, `nf`, `nm/f`, `vtr`, `vi`,
`v phr`, `adv`, `adj`, `adj mf`, `interj`, `conj`, `pron`, `prep`. Check the
code against the sentence (`no obstante` is `conj`/`adv`, not both on two
rows; `candil` is a noun, not an adjective). Nouns have an article (`el`,
`la`, `el/la` for `nm/f`); nothing else does.

## The meaning

The meaning should be the sense used in the sentences, plus the main sense if
that one is much more common — a card that only teaches an obscure sense is a
trap.

- **Over-fitted to one sentence.** `retozar` defined as "to laze about" from
  one lazy-morning scene (it is *to frolic*); `hurón` as "network watcher"
  from a sci-fi metaphor (it is *ferret*); `mascar` as "to spoon-feed" (it is
  *to chew*; that sense only exists in "dárselo todo mascado").
- **Another word's definition.** `aquejar` was given `achacar`'s meaning
  because the author used it oddly. When the book's usage is nonstandard,
  define the standard sense and say so.
- **Missing the everyday sense.** `remangarse` without *to roll up one's
  sleeves*; `tropezón` without *a stumble*; `asta` without *horn, flagpole*.
- **Invented detail.** `florete` as "foil, a helmet with a pointed spike" —
  the second half came from one ambiguous scene.
- **Typos in the English** ("unncessary", "grop", "glothing") and leftover
  HTML entities (`&quot;`).

If you are not sure of a meaning, do not guess a correction: flag it in your
notes for the user, saying what you suspect.

## Problems the owner has flagged by hand

These came up again and again in the cards the owner red-flagged, and no
script can catch them:

- **Sentences using a sense the definition lacks.** `acceder`,
  `comprometer`, `marcador` (also "marker"), `llevarse` (also "to get along
  with"), `carril` (also "track"): the extra sentences show a second sense
  the definition never mentions. Extend the definition, or move the sentence
  to a card for that sense.
- **Definitions that are wrong or off.** `techo` is "ceiling" more than
  "roof"; `hurón` is "ferret"; "no pares de…" was translated as "stop".
- **Part of speech and number.** `escrito/a` used as a noun in its sentence;
  `guacho/a` adjective or noun; `albores` and `cagarrinas`, used only in the
  plural.
- **Idioms filed under the literal word.** "Sarna con gusto no pica" doesn't
  use the literal meaning of `sarna`; "el toque de queda" is "curfew", which
  `toque` alone doesn't teach. Flag these for a phrase card.

## Whether it should be a card at all

Drop rows that teach nothing useful, with the reason: words invented by one
author (`bacapear`), guesses the meaning itself admits to ("likely short for
'adolescente'"), proper nouns, and noun entries whose only content is an
idiom that has its own card (`clavija` next to `apretar las clavijas`).
Things that are real but very narrow (a regional ice-cream flavour) are the
user's call — flag them rather than dropping.

## Duplicates

`csv_matches` and `deck_matches` come from loose string matching (accents,
gender suffixes, parentheticals, a pronominal `se` and a plural ending are
ignored), so they are leads, not verdicts — and they miss variants like `ser
la gota que colma el vaso` / `la gota que desborda el vaso` and irregular
forms, so keep an eye out yourself. They are computed from the headwords as
exported, before your corrections.

- In the **word CSV** (before generation), do not drop duplicates: make
  their headwords agree so the generation step's merge machinery groups them,
  and mention the pair in your notes. Be deliberate about deck matches: a row
  whose headword and part of speech match a deck note exactly is folded into
  that note as an update without anyone being asked, so this is the only
  point at which to notice a genuinely different word hiding behind the same
  spelling; if you find one, flag it for the user rather than guessing how to
  keep it apart.
- In the **diff**, a row that duplicates a deck note at a *different* rank
  will create a second note on import. Do not change ranks; flag it, naming
  both ranks.

## Sentences

Each card should end up with sentences that show the meaning on the card:
for a new card, aim for at least three good ones, the first of which is the
clearest. When removing weak sentences leaves fewer, write replacements.

- **Too short to give context**: "La cerveza es cebada.", "¿Tienes
  mejillones?", "Menuda zorra." A learner should be able to guess the word's
  meaning from the sentence. Replace these.
- **A different sense than the card**: `agredir` defined as "to assault"
  with the figurative "Tanta química me agrede". Replace, or keep it as an
  extra if the main sentence shows the card's sense.
- **Errors in model-generated sentences**: "Nececitamos", a missing plural.
  Fix in place.
- **"Book" sentences that are not in the book.** A `book:` sentence is the
  model's extraction from the passage, which it is allowed to shorten or
  complete, so it is not verbatim; but it should be recognizably from the
  passage in `contexts`. One that is not was invented and mislabelled: if it
  is good, keep it but re-attribute it with `"source": "reviewer"` in the
  edit; if not, replace it. Leave faithful extracts alone apart from real
  errors — those are the user's reading, which is the point of the card.
- **Translation** should carry the sense, not be word-for-word, and should
  not change the meaning.

When you write a sentence, follow the rules the pipeline's own prompts use,
so yours read like the rest of the deck:

- Idiomatic Spanish in the configured dialect (`[sentences] dialect` in
  `config.toml`), the way a published book from that region would use it,
  without forcing slang in.
- Spanish punctuation: dialogue with the raya (—Dime algo que no sepa —dijo
  Ramón.), «guillemets» for quoting a word inline, never English quotation
  marks; normal English punctuation in the translation.
- For verbs, vary mood and tense across the card's sentences. A useful or
  interesting fact in a sentence is a bonus.
- Vulgar and sexual vocabulary is normal vocabulary; write sentences for it
  matter-of-factly.

## Content worth flagging

Sentences quoted from books can carry violence, sexual content or slurs the
user may not want to see while studying. Do not censor them; list them under
"Needs your decision" so the user can choose.

## Reviewing a chunk

Whether you review a chunk yourself or hand it to a subagent, the output is
the same two files next to the chunk:

- `chunk-NNN.corrections.json`: a JSON list in the format
  `python scripts/review_words.py --help` documents. Key rows by `row` (the
  `row` number shown in the chunk, which counts from 0 across the whole file)
  in the word CSV and by `rank` in the diff, repeat the current `word` as a guard,
  and give a short `reason` the user will read in the changelog. Only correct
  what you are confident about.
- `chunk-NNN.notes.md`: everything that needs the user's judgment, one line
  per item, starting with the headword (and rank, if it has one).

An example correction:

```json
[
  {
    "row": 14,
    "word": "mesar",
    "reason": "only used pronominally; means to tear at (hair), not to stroke",
    "set": {"word": "mesarse", "part_of_speech": "vtr", "meaning": "to tear at (one's hair)"}
  },
  {
    "row": 22,
    "word": "cebada",
    "reason": "main sentence too short to show the meaning",
    "sentences": [
      {"index": 0, "remove": true},
      {"add": true, "first": true,
       "spanish": "La cerveza se elabora con cebada malteada, agua y lúpulo.",
       "english": "Beer is made from malted barley, water and hops."}
    ]
  }
]
```
