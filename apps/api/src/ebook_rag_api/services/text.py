import re

WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "need",
    "needs",
    "note",
    "notes",
    "of",
    "on",
    "or",
    "say",
    "says",
    "should",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "would",
    "with",
}

NON_ANCHOR_TERMS = {
    "appear",
    "article",
    "book",
    "call",
    "discuss",
    "happen",
    "mention",
    "nickname",
    "occur",
    "policy",
    "say",
    "tell",
    "thing",
    "use",
    "write",
}


def normalize_query_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_term(token: str) -> str:
    normalized = token.lower()
    if len(normalized) > 5 and normalized.endswith("ies"):
        normalized = normalized[:-3] + "y"
    elif len(normalized) > 4 and normalized.endswith("ing"):
        normalized = normalized[:-3]
        if len(normalized) >= 2 and normalized[-1] == normalized[-2]:
            normalized = normalized[:-1]
    elif len(normalized) > 3 and normalized.endswith("ed"):
        normalized = normalized[:-2]
        if len(normalized) >= 2 and normalized[-1] == normalized[-2]:
            normalized = normalized[:-1]
    elif len(normalized) > 3 and normalized.endswith("es"):
        normalized = normalized[:-2]
    elif len(normalized) > 3 and normalized.endswith("s"):
        normalized = normalized[:-1]
    elif len(normalized) > 5 and normalized.endswith("ion"):
        normalized = normalized[:-3]
    if len(normalized) > 5 and normalized.endswith("e"):
        normalized = normalized[:-1]
    return normalized


def tokenize_terms(text: str, *, drop_stopwords: bool = False) -> set[str]:
    terms: set[str] = set()
    for raw_token in TOKEN_RE.findall(text.lower()):
        if drop_stopwords and raw_token in STOPWORDS:
            continue
        token = normalize_term(raw_token)
        if len(token) <= 1:
            continue
        if drop_stopwords and token in STOPWORDS:
            continue
        terms.add(token)
    return terms


def normalized_token_sequence(text: str, *, drop_stopwords: bool = False) -> list[str]:
    sequence: list[str] = []
    for raw_token in TOKEN_RE.findall(text.lower()):
        if drop_stopwords and raw_token in STOPWORDS:
            continue
        token = normalize_term(raw_token)
        if len(token) <= 1:
            continue
        if drop_stopwords and token in STOPWORDS:
            continue
        sequence.append(token)
    return sequence


def contains_normalized_phrase(query: str, text: str) -> bool:
    normalized_query = " ".join(normalized_token_sequence(query))
    normalized_text = " ".join(normalized_token_sequence(text))
    return bool(normalized_query and normalized_query in normalized_text)


def extract_anchor_terms(text: str) -> set[str]:
    anchor_terms: set[str] = set()
    fallback_terms: set[str] = set()

    for raw_token in TOKEN_RE.findall(text):
        lowered = raw_token.lower()
        if lowered in STOPWORDS:
            continue
        normalized = normalize_term(lowered)
        if len(normalized) <= 1 or normalized in STOPWORDS:
            continue
        if normalized not in NON_ANCHOR_TERMS and len(raw_token) >= 4:
            fallback_terms.add(normalized)
        if (
            normalized not in NON_ANCHOR_TERMS
            and (
                any(character.isdigit() for character in raw_token)
                or len(raw_token) >= 6
                or raw_token[:1].isupper()
            )
        ):
            anchor_terms.add(normalized)

    return anchor_terms or fallback_terms


def extract_constraint_terms(text: str) -> set[str]:
    constraint_terms: set[str] = set()

    for raw_token in TOKEN_RE.findall(text):
        lowered = raw_token.lower()
        if lowered in STOPWORDS:
            continue
        normalized = normalize_term(lowered)
        if len(normalized) <= 1 or normalized in STOPWORDS:
            continue
        if any(character.isdigit() for character in raw_token):
            constraint_terms.add(normalized)
            continue
        if normalized in NON_ANCHOR_TERMS:
            continue
        if raw_token[:1].islower() and len(normalized) >= 4:
            constraint_terms.add(normalized)

    return constraint_terms
