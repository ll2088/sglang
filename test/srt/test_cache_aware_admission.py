from dataclasses import dataclass

import pytest

from sglang.srt.mem_cache.cache_aware_admission import (
    CacheAwareAdmissionConfig,
    reorder_cache_aware_admission,
)


@dataclass(frozen=True)
class _Req:
    name: str
    cached: int = 0
    eligible: bool = True


def _reorder(items, config, usage=1.0):
    return reorder_cache_aware_admission(
        items,
        config,
        cache_usage=usage,
        is_eligible=lambda req: req.eligible,
        num_cached_tokens=lambda req: req.cached,
    )


def test_window_zero_keeps_arrival_order():
    items = [_Req('a', 0), _Req('b', 10)]
    assert _reorder(items, CacheAwareAdmissionConfig()) == items


def test_threshold_skips_reordering():
    items = [_Req('cold', 0), _Req('warm', 10)]
    config = CacheAwareAdmissionConfig(window=2, threshold=0.5)

    assert _reorder(items, config, usage=0.49) == items
    assert [req.name for req in _reorder(items, config, usage=0.5)] == [
        'warm',
        'cold',
    ]


def test_only_eligible_slots_move():
    items = [
        _Req('cold', 0),
        _Req('blocked', 99, eligible=False),
        _Req('warm', 10),
        _Req('tie', 10),
    ]

    result = _reorder(items, CacheAwareAdmissionConfig(window=4))

    assert [req.name for req in result] == ['warm', 'blocked', 'tie', 'cold']


def test_reordering_is_confined_to_window():
    items = [_Req('cold', 0), _Req('warm', 10), _Req('tail', 100)]

    result = _reorder(items, CacheAwareAdmissionConfig(window=2))

    assert [req.name for req in result] == ['warm', 'cold', 'tail']


def test_probe_is_read_only_and_limited_to_eligible_slots():
    items = [_Req('a', 1), _Req('b', 2, eligible=False), _Req('c', 3)]
    original = list(items)
    probed = []

    def probe(req):
        probed.append(req.name)
        return req.cached

    result = reorder_cache_aware_admission(
        items,
        CacheAwareAdmissionConfig(window=3),
        cache_usage=1.0,
        is_eligible=lambda req: req.eligible,
        num_cached_tokens=probe,
    )

    assert probed == ['a', 'c']
    assert items == original
    assert result is not items
    assert [req.name for req in result] == ['c', 'b', 'a']


@pytest.mark.parametrize(
    'kwargs',
    [{'window': -1}, {'threshold': -0.1}, {'threshold': 1.1}],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        CacheAwareAdmissionConfig(**kwargs)
