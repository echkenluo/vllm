# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The segment cache must return exactly what a whole-prompt encode returns.

A wrong split would change the prompt ids silently, so the tests compare the
cached path with a plain encode on growing agent-style conversations, and check
that markers which a fast tokenizer would not split on are refused.
"""

import re
from types import SimpleNamespace

import pytest

from vllm.tokenizers.deepseek_v4 import get_deepseek_v4_tokenizer
from vllm.tokenizers.segment_token_cache import (
    DSV4_SEGMENT_MARKERS,
    SegmentTokenCache,
)

ADDED = {m: 1000 + i for i, m in enumerate(DSV4_SEGMENT_MARKERS)}


class SplittingTokenizer:
    """Mimics a HF fast tokenizer: split on added tokens, then encode pieces.

    Ordinary text is encoded with a context-dependent rule (a token per word,
    the id depends on the previous word), so encoding a piece without its
    left context would give different ids. The cache is only correct if it
    splits exactly where the tokenizer itself splits.
    """

    def __init__(self, added=ADDED, flags=None):
        self.added = dict(added)
        self.added_tokens_decoder = {
            i: SimpleNamespace(**(flags or {}).get(t, {})) for t, i in self.added.items()
        }
        self.encode_calls = 0
        self._split = re.compile(
            "("
            + "|".join(re.escape(t) for t in sorted(self.added, key=len, reverse=True))
            + ")"
        )

    def get_added_vocab(self):
        return dict(self.added)

    def encode(self, text, add_special_tokens=False, **kwargs):
        self.encode_calls += 1
        out = []
        for i, piece in enumerate(self._split.split(text)):
            if not piece:
                continue
            if i % 2:
                out.append(self.added[piece])
                continue
            prev = 0
            for word in re.findall(r"\S+|\s+", piece):
                tok = (hash(word) ^ prev) % 997
                out.append(tok)
                prev = tok
        return out


def _conversation(turns):
    text = "<｜begin▁of▁sentence｜>You are a coding agent."
    for t in range(turns):
        text += f"<｜User｜>run the tests in module {t} and show me the log"
        text += f"<｜Assistant｜><think>step {t}: read files</think>ran {t} tests ok"
        text += "<｜end▁of▁sentence｜>"
    return text + "<｜latest_reminder｜>be brief<｜Assistant｜><think>"


def test_matches_plain_encode_on_growing_conversation():
    tok = SplittingTokenizer()
    cache = SegmentTokenCache(tok, DSV4_SEGMENT_MARKERS, max_chars=1 << 20)
    for turns in range(1, 30):
        prompt = _conversation(turns)
        assert cache.encode(prompt) == tok.encode(prompt)
    # Later turns re-use earlier segments instead of re-encoding them.
    assert cache.hits > cache.misses


def test_lru_eviction_keeps_results_exact():
    tok = SplittingTokenizer()
    cache = SegmentTokenCache(tok, DSV4_SEGMENT_MARKERS, max_chars=200)
    for turns in (5, 12, 3, 12, 20):
        prompt = _conversation(turns)
        assert cache.encode(prompt) == tok.encode(prompt)
    assert cache._cached_chars <= 200 or len(cache._cache) == 1


def test_refuses_markers_the_tokenizer_would_not_split_on():
    flags = {m: {"lstrip": True} for m in DSV4_SEGMENT_MARKERS}
    with pytest.raises(ValueError):
        SegmentTokenCache(SplittingTokenizer(flags=flags), DSV4_SEGMENT_MARKERS, 1000)
    # A marker contained in a longer added token is not a safe split point.
    added = dict(ADDED)
    added["<｜User｜>x"] = 2000
    cache = SegmentTokenCache(SplittingTokenizer(added), DSV4_SEGMENT_MARKERS, 1000)
    assert "<｜User｜>" not in cache.markers


def test_chat_template_uses_cache_only_when_enabled(monkeypatch):
    messages = [{"role": "user", "content": "hello"}]
    # get_deepseek_v4_tokenizer returns a copy of the tokenizer it wraps, so
    # encode calls are counted on the returned object.
    monkeypatch.delenv("DSV4_SEGMENT_TOKEN_CACHE", raising=False)
    plain = get_deepseek_v4_tokenizer(SplittingTokenizer())
    expected = plain.apply_chat_template(messages)
    calls = plain.encode_calls
    plain.apply_chat_template(messages)
    # Without the switch every request is encoded again.
    assert plain.encode_calls == calls + 1

    monkeypatch.setenv("DSV4_SEGMENT_TOKEN_CACHE", "1")
    cached = get_deepseek_v4_tokenizer(SplittingTokenizer())
    assert cached.apply_chat_template(messages) == expected
    calls = cached.encode_calls
    assert cached.apply_chat_template(messages) == expected
    # The second identical request encodes nothing new.
    assert cached.encode_calls == calls
    # Truncation still goes through the plain encode.
    cached.apply_chat_template(messages, truncation=True, max_length=8)
    assert cached.encode_calls == calls + 1
