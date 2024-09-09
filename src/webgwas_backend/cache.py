from collections import OrderedDict
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUResultsCache(Generic[K, V]):
    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.cache: OrderedDict[K, V] = OrderedDict()

    def __setitem__(self, key: K, value: V) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value

    def __getitem__(self, key: K) -> V:
        if key in self.cache:
            self.cache.move_to_end(key)
        return self.cache[key]

    def __len__(self) -> int:
        return len(self.cache)

    def __contains__(self, key: K) -> bool:
        return key in self.cache

    def get(self, key: K, default: V | None = None) -> V | None:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return default
