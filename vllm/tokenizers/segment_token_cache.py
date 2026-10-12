# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Segment-level prompt tokenization cache for the DeepSeek-V4 chat encoder.

The DeepSeek-V4 encoder renders a whole conversation into one flat string,
which is then tokenized on every request. In an agent workload consecutive
requests of one trajectory share almost the entire prompt (about 18K prompt
tokens per request of which about 400 are new on the L20 coding-agent runs),
so the same text is re-tokenized dozens of times in the API server process.

A HF fast tokenizer splits its input on added tokens before normalization and
pre-tokenization and tokenizes each in-between piece independently. Splitting
the rendered prompt at a subset of those added-token occurrences and encoding
the pieces separately therefore produces exactly the same ids as encoding the
whole string, as long as

* every split marker is an added token that encodes to exactly one id,
* the marker does not strip surrounding whitespace (lstrip/rstrip) and is not
  single_word, and
* no other added token can contain or straddle an occurrence of the marker.

The conditions are checked when the cache is built; a marker that fails is
dropped, and if nothing is left no cache is built and the caller keeps the
plain encode. The ids of each piece are kept in an LRU keyed by the piece text,
so a request only tokenizes text the process has not seen before.
"""

import re
import threading
from array import array
from collections import OrderedDict
from collections.abc import Iterable

from vllm.logger import init_logger

logger = init_logger(__name__)

# Role and structure markers emitted by deepseek_v4_encoding.encode_messages.
DSV4_SEGMENT_MARKERS: tuple[str, ...] = (
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
    "<｜latest_reminder｜>",
    "<think>",
    "</think>",
)


def _overlaps_other_added_token(marker: str, added_vocab: dict[str, int]) -> bool:
    for other in added_vocab:
        if other == marker:
            continue
        if marker in other or other in marker:
            return True
        for k in range(1, min(len(marker), len(other))):
            if other.endswith(marker[:k]) or marker.endswith(other[:k]):
                return True
    return False


class SegmentTokenCache:
    """Equivalent of ``tokenizer.encode(text, add_special_tokens=False)``."""

    def __init__(self, tokenizer, markers: Iterable[str], max_chars: int):
        self.tokenizer = tokenizer
        self.max_chars = max_chars

        added_vocab = tokenizer.get_added_vocab()
        decoder = getattr(tokenizer, "added_tokens_decoder", {}) or {}

        usable: list[str] = []
        self._marker_id: dict[str, int] = {}
        for marker in markers:
            token_id = added_vocab.get(marker)
            if token_id is None:
                continue
            info = decoder.get(token_id)
            if info is not None and (
                getattr(info, "lstrip", False)
                or getattr(info, "rstrip", False)
                or getattr(info, "single_word", False)
            ):
                continue
            if tokenizer.encode(marker, add_special_tokens=False) != [token_id]:
                continue
            usable.append(marker)
            self._marker_id[marker] = token_id

        # The tokenizer matches added tokens leftmost-longest over its whole
        # added vocabulary, so a marker is only a safe split point if no other
        # added token can swallow or straddle one of its occurrences.
        usable = [m for m in usable if not _overlaps_other_added_token(m, added_vocab)]
        if not usable:
            raise ValueError("no usable segment markers for this tokenizer")

        self._splitter = re.compile(
            "("
            + "|".join(re.escape(m) for m in sorted(usable, key=len, reverse=True))
            + ")"
        )
        self.markers: tuple[str, ...] = tuple(usable)

        # ids are kept as C int arrays: a list of Python ints costs about 36
        # bytes per token, which would turn the character budget into GiBs.
        self._cache: OrderedDict[str, array] = OrderedDict()
        self._cached_chars = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        with self._lock:
            cache = self._cache
            for i, piece in enumerate(self._splitter.split(text)):
                if not piece:
                    continue
                if i % 2:
                    out.append(self._marker_id[piece])
                    continue
                ids = cache.get(piece)
                if ids is None:
                    self.misses += 1
                    ids = array(
                        "i", self.tokenizer.encode(piece, add_special_tokens=False)
                    )
                    cache[piece] = ids
                    self._cached_chars += len(piece)
                    while self._cached_chars > self.max_chars and len(cache) > 1:
                        old_key, _ = cache.popitem(last=False)
                        self._cached_chars -= len(old_key)
                else:
                    cache.move_to_end(piece)
                    self.hits += 1
                out.extend(ids)
        return out


def build_segment_token_cache(
    tokenizer, markers: Iterable[str], max_chars: int
) -> SegmentTokenCache | None:
    """Return a cache, or None when this tokenizer cannot be split safely."""
    try:
        cache = SegmentTokenCache(tokenizer, markers, max_chars)
    except Exception as e:
        logger.warning("Segment token cache disabled: %s", e)
        return None
    logger.info("Segment token cache enabled, markers: %s", ", ".join(cache.markers))
    return cache
