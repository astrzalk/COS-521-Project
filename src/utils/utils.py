#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copy and Pasted this code from:
# https://stackoverflow.com/questions/4197805/python-for-loop-look-ahead
from itertools import tee, islice, zip_longest

def get_next(some_iterable, window=1):
    """
    Makes an interable return the current and the next element
    simultaneously per each iteration.
    Examples usage:
    ---------------
    >>> a = [1,2,3,4]
    >>> for cur, next in get_next(a):
    ...     if next is None:
    ...         break
    ...     print(cur * next)
    ...
    2
    6
    12
    """
    items, nexts = tee(some_iterable, 2)
    nexts = islice(nexts, window, None)
    return zip_longest(items, nexts)

