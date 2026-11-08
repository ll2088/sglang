'''Cache-aware admission ordering for the waiting queue.

Within a bounded look-ahead window, requests whose prefix is already resident
in the KV cache can be admitted ahead of cold requests. Reordering is confined
to slots occupied by eligible requests, so blocked or partially computed
requests keep their exact position.

The helper is intentionally side-effect free: callers inject the cache probe,
which must only read prefix-cache state and must not allocate blocks or emit
events.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

T = TypeVar('T')


@dataclass(frozen=True)
class CacheAwareAdmissionConfig:
    '''Configuration for bounded cache-aware admission ordering.'''

    window: int = 0
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.window < 0:
            raise ValueError('window must be >= 0')
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError('threshold must be in [0.0, 1.0]')

    @property
    def enabled(self) -> bool:
        return self.window > 0


def reorder_cache_aware_admission(
    waiting: Sequence[T],
    config: CacheAwareAdmissionConfig,
    *,
    cache_usage: float,
    is_eligible: Callable[[T], bool],
    num_cached_tokens: Callable[[T], int],
) -> list[T]:
    '''Return `waiting` in cache-aware admission order.

    Only the first `config.window` items are considered. Within that window,
    eligible items are permuted among their own slots by descending resident
    prefix length, with arrival order breaking ties. Ineligible items are never
    moved, and items beyond the window keep their position.

    A new list is always returned; the input sequence is not mutated. The
    probe is called once per eligible slot and only after the threshold check.
    '''
    items = list(waiting)
    if not config.enabled or cache_usage < config.threshold:
        return items

    head = items[: config.window]
    slots = [i for i, request in enumerate(head) if is_eligible(request)]
    if len(slots) < 2:
        return items

    cached = [num_cached_tokens(head[i]) for i in slots]
    order = sorted(range(len(slots)), key=lambda pos: (-cached[pos], pos))
    if all(order[pos] == pos for pos in range(len(order))):
        return items

    reordered = list(head)
    for slot, source_pos in zip(slots, order):
        reordered[slot] = head[slots[source_pos]]
    return reordered + items[config.window :]
