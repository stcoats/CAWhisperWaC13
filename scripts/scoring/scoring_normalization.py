#!/usr/bin/env python3
"""Shared lexical normalization and uncertainty-aware edit scoring."""

from __future__ import annotations

import re
import unicodedata

from regional_spelling_variants import DICTIONARY as REGIONAL_SPELLING_DATA


SCORING_VERSION = "context-gold-orthography-v6"
UNCLEAR_TOKEN = "<unclear>"
CLITIC_S_TOKEN = "<clitic-s>"
VERBATIM_TOKEN = re.compile(
    r"[a-z0-9]+(?:['’][a-z0-9]+)*(?:-[a-z0-9]+(?:['’][a-z0-9]+)*)*-?",
    re.IGNORECASE,
)
BRACKET_EVENT = re.compile(r"\[([^\]]+)\]")
SPACED_NASAL_FILLER = re.compile(r"\bm+\s+h+m+\b", re.IGNORECASE)
SPACED_UH_HUH = re.compile(r"\buh+\s+huh+\b", re.IGNORECASE)
UNCLEAR_EVENT = re.compile(r"^unclear(?:\s+(\d+))?$", re.IGNORECASE)
SPOKEN_INITIALISM = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Z](?:\s*[.-]\s*|\s+)){1,7}[A-Z]\.?(?![A-Za-z0-9])"
)

FILLERS = {
    "ah", "eh", "er", "erm", "huh", "mm", "uh", "uh-huh", "um",
}
NASAL_FILLER_VARIANTS = {
    "hm", "hmm", "hmmm", "m-hm", "m-hmm", "mhm", "mm", "mm-hm",
    "mm-hmm", "mmhm", "mmm", "mmmm",
}
BRACKETED_LEXICAL_EVENTS = FILLERS | {"hmm", "mmm"}

# These are inaudible writing conventions, not lexical differences. Keep the
# map explicit and conservative: do not merge audible forms such as
# "aluminium/aluminum", "gonna/going to", or "st/saint".
CANONICAL_FORMS = {
    # Titles and common written abbreviations.
    "mr": "mister",
    "mister": "mister",
    "mrs": "missus",
    "missus": "missus",
    "ms": "miz",
    "miz": "miz",
    "dr": "doctor",
    "doctor": "doctor",
    "prof": "professor",
    "professor": "professor",
    "rev": "reverend",
    "reverend": "reverend",
    "sgt": "sergeant",
    "sergeant": "sergeant",
    "jr": "junior",
    "junior": "junior",
    "sr": "senior",
    "senior": "senior",
    "cr": "councilor",
    "councillor": "councilor",
    "councilor": "councilor",
    # Safe US/Australian/New Zealand spelling equivalents.
    "acknowledgement": "acknowledgment",
    "acknowledgements": "acknowledgments",
    "behaviour": "behavior",
    "behaviours": "behaviors",
    "cancelled": "canceled",
    "cancelling": "canceling",
    "catalogue": "catalog",
    "catalogues": "catalogs",
    "catalogued": "cataloged",
    "cataloguing": "cataloging",
    "centre": "center",
    "centres": "centers",
    "centred": "centered",
    "centring": "centering",
    "cheque": "check",
    "cheques": "checks",
    "colour": "color",
    "colours": "colors",
    "coloured": "colored",
    "colouring": "coloring",
    "defence": "defense",
    "defences": "defenses",
    "enrol": "enroll",
    "enrols": "enrolls",
    "enrolled": "enrolled",
    "enrolling": "enrolling",
    "enrolment": "enrollment",
    "enrolments": "enrollments",
    "favour": "favor",
    "favours": "favors",
    "favoured": "favored",
    "favouring": "favoring",
    "favourite": "favorite",
    "favourites": "favorites",
    "flavour": "flavor",
    "flavours": "flavors",
    "fulfil": "fulfill",
    "fulfils": "fulfills",
    "fulfilled": "fulfilled",
    "fulfilling": "fulfilling",
    "fulfilment": "fulfillment",
    "grey": "gray",
    "honour": "honor",
    "honours": "honors",
    "honoured": "honored",
    "honouring": "honoring",
    "honourable": "honorable",
    "humour": "humor",
    "kerb": "curb",
    "kerbs": "curbs",
    "labour": "labor",
    "labours": "labors",
    "laboured": "labored",
    "labouring": "laboring",
    "licence": "license",
    "licences": "licenses",
    "litre": "liter",
    "litres": "liters",
    "metre": "meter",
    "metres": "meters",
    "neighbour": "neighbor",
    "neighbours": "neighbors",
    "neighbourhood": "neighborhood",
    "neighbourhoods": "neighborhoods",
    "offence": "offense",
    "offences": "offenses",
    "practice": "practice",
    "practise": "practice",
    "practised": "practiced",
    "practising": "practicing",
    "pretence": "pretense",
    "programme": "program",
    "programmes": "programs",
    "theatre": "theater",
    "theatres": "theaters",
    "traveller": "traveler",
    "travellers": "travelers",
    "travelled": "traveled",
    "travelling": "traveling",
    "tyre": "tire",
    "tyres": "tires",
}

# The upstream table is intentionally broad. These entries are regional
# vocabulary or distinct pronunciations, rather than merely different ways of
# spelling the same audible lexical item, so they must remain ASR differences.
REGIONAL_SPELLING_EXCLUSIONS = {
    "aeroplane", "airplane",
    "bougie", "bogie",
    "louth", "loth",
    "mum", "mom",
    "rolland", "roland",
    "titbit", "tidbit", "titbits", "tidbits",
}
REGIONAL_CANONICAL_FORMS = {}
for regional_form, us_form in (
    line.split("\t") for line in REGIONAL_SPELLING_DATA.splitlines() if line.strip()
):
    if regional_form not in REGIONAL_SPELLING_EXCLUSIONS and us_form not in REGIONAL_SPELLING_EXCLUSIONS:
        REGIONAL_CANONICAL_FORMS[regional_form] = us_form
        REGIONAL_CANONICAL_FORMS[us_form] = us_form


def _ize_family(token: str) -> str:
    """Canonicalize productive -ise/-isation spellings conservatively."""
    replacements = (
        ("isations", "izations"),
        ("isation", "ization"),
        ("isers", "izers"),
        ("iser", "izer"),
        ("ising", "izing"),
        ("ised", "ized"),
        ("ises", "izes"),
        ("ise", "ize"),
    )
    # Restrict the productive rule to common, genuinely equivalent families;
    # words such as "promise" and "surprise" must remain untouched.
    roots = (
        "author", "categor", "central", "character", "civil", "colon",
        "commercial", "computer", "critic", "emphas", "final", "formal",
        "general", "hospital", "legal", "local", "maxim", "minim",
        "modern", "normal", "organ", "priorit", "public", "real",
        "recogn", "special", "standard", "summar", "visual",
    )
    for source, target in replacements:
        if token.endswith(source) and token[: -len(source)].startswith(roots):
            return token[: -len(source)] + target
    return token


def canonical_word(token: str) -> str:
    token = token.lower().replace("’", "'")
    # Macron and other combining-mark choices are orthographic, not separate
    # ASR events: Māori/Maori and kōrero/korero must compare equal. Raw text is
    # retained elsewhere; this fold applies only inside the scorer.
    token = "".join(
        character
        for character in unicodedata.normalize("NFKD", token)
        if unicodedata.category(character) != "Mn"
    )
    return CANONICAL_FORMS.get(
        token,
        REGIONAL_CANONICAL_FORMS.get(token, _ize_family(token)),
    )


def _collapse_spoken_initialisms(text: str) -> str:
    """Make L T P, L.T.P., and L-T-P compare as the single token LTP."""
    def replace(match: re.Match[str]) -> str:
        letters = "".join(character for character in match.group(0) if character.isalpha())
        # Preserve audible repetitions such as "I I" and "A A".
        if len(set(letters)) == 1:
            return match.group(0)
        return letters

    return SPOKEN_INITIALISM.sub(replace, text)


def _ordinary_tokens(text: str) -> list[str]:
    text = SPACED_NASAL_FILLER.sub("mm-hmm", text)
    text = SPACED_UH_HUH.sub("uh-huh", text)
    raw = [match.group(0) for match in VERBATIM_TOKEN.finditer(text)]
    result = []
    for token in raw:
        token = token.lower().replace("’", "'")
        if token in NASAL_FILLER_VARIANTS:
            result.append("mm")
        elif token == "uh-huh":
            result.append(token)
        elif "-" in token and not token.endswith("-"):
            result.extend(canonical_word(part) for part in token.split("-") if part)
        elif token.endswith("-"):
            result.append(canonical_word(token[:-1]) + "-")
        else:
            result.append(canonical_word(token))
    return result


def canonical_tokens(text: str, conventional: bool = False) -> list[str]:
    """Tokenize while preserving verbatim events and masking unclear spans."""
    text = _collapse_spoken_initialisms(text)
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    text = "".join(
        character
        for character in unicodedata.normalize("NFKD", text)
        if unicodedata.category(character) != "Mn"
    )
    result: list[str] = []
    cursor = 0
    for event in BRACKET_EVENT.finditer(text):
        result.extend(_ordinary_tokens(text[cursor : event.start()]))
        label = " ".join(event.group(1).strip().split())
        unclear = UNCLEAR_EVENT.fullmatch(label)
        if unclear:
            count = int(unclear.group(1) or 1)
            if not 1 <= count <= 50:
                raise ValueError(f"invalid unclear-span length: [{label}]")
            result.extend([UNCLEAR_TOKEN] * count)
        elif label in BRACKETED_LEXICAL_EVENTS:
            result.extend(_ordinary_tokens(label))
        # All other bracketed non-speech labels are excluded lexically.
        cursor = event.end()
    result.extend(_ordinary_tokens(text[cursor:]))
    if conventional:
        expanded = []
        for token in result:
            if token != UNCLEAR_TOKEN and token.endswith("'s") and len(token) > 2:
                expanded.extend((token[:-2], CLITIC_S_TOKEN))
            else:
                expanded.append("them" if token == "em" else token)
        result = expanded
        result = [
            token for token in result
            if token == UNCLEAR_TOKEN or (token not in FILLERS and not token.endswith("-"))
        ]
        result = [
            token for index, token in enumerate(result)
            if index == 0 or token == UNCLEAR_TOKEN or token != result[index - 1]
        ]
    return result


def scoreable_units(items: list[str]) -> int:
    return sum(token != UNCLEAR_TOKEN for token in items)


def tokens_equivalent(reference: str, hypothesis: str) -> bool:
    return (
        reference == hypothesis
        or reference == CLITIC_S_TOKEN and hypothesis in {"is", "has"}
        or hypothesis == CLITIC_S_TOKEN and reference in {"is", "has"}
    )


def alignment_pairs(reference: list[str], hypothesis: list[str]) -> list[tuple[int | None, int | None]]:
    """Minimum-edit alignment; each unclear reference word masks zero/one hypothesis word."""
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    distance = [[0] * cols for _ in range(rows)]
    operation = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        if reference[i - 1] == UNCLEAR_TOKEN:
            distance[i][0], operation[i][0] = distance[i - 1][0], "W0"
        else:
            distance[i][0], operation[i][0] = distance[i - 1][0] + 1, "D"
    for j in range(1, cols):
        distance[0][j], operation[0][j] = j, "I"
    priority = {"M": 0, "W1": 1, "W0": 2, "S": 3, "D": 4, "I": 5}
    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == UNCLEAR_TOKEN:
                choices = [
                    (distance[i - 1][j - 1], "W1"),
                    (distance[i - 1][j], "W0"),
                    (distance[i][j - 1] + 1, "I"),
                ]
            else:
                choices = [
                    (
                        distance[i - 1][j - 1] + (not tokens_equivalent(reference[i - 1], hypothesis[j - 1])),
                        "M" if tokens_equivalent(reference[i - 1], hypothesis[j - 1]) else "S",
                    ),
                    (distance[i - 1][j] + 1, "D"),
                    (distance[i][j - 1] + 1, "I"),
                ]
            distance[i][j], operation[i][j] = min(choices, key=lambda item: (item[0], priority[item[1]]))
    pairs = []
    i, j = len(reference), len(hypothesis)
    while i or j:
        op = operation[i][j]
        if op in {"M", "S", "W1"}:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif op in {"D", "W0"}:
            pairs.append((i - 1, None)); i -= 1
        elif op == "I":
            pairs.append((None, j - 1)); j -= 1
        else:
            raise RuntimeError(f"invalid alignment backtrace at {i}, {j}")
    return list(reversed(pairs))


def edit_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    substitutions = deletions = insertions = 0
    for ref_index, hyp_index in alignment_pairs(reference, hypothesis):
        if ref_index is None:
            insertions += 1
        elif reference[ref_index] == UNCLEAR_TOKEN:
            continue
        elif hyp_index is None:
            deletions += 1
        elif not tokens_equivalent(reference[ref_index], hypothesis[hyp_index]):
            substitutions += 1
    return substitutions, deletions, insertions


NORMALIZATION_DESCRIPTION = (
    "case and punctuation are ignored; a vendored comprehensive UK/Australian/US "
    "spelling-variant table, macron/diacritic variants, and written title forms are canonicalized; "
    "spoken initialism variants such as L T P/L.T.P./LTP are equivalent; "
    "leading straight/curly apostrophes in reduced forms such as 'em are ignored "
    "without merging the audible forms em/them in verbatim scoring; normalized "
    "scoring equates em/them and contracted 's with its is/has expansion; nasal backchannel and "
    "uh-huh spellings are canonicalized; bracketed non-speech events are excluded; "
    "[unclear] and [unclear N] reference words mask zero or one hypothesis word each; "
    "audible lexical distinctions, reduced forms, repetitions, and trailing fragments "
    "remain distinct in verbatim scoring"
)
