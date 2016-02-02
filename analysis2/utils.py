"""
Useful functions
"""

import itertools

def loop_iterator(ranges):
    items = [[x for x in range(n)] for n in ranges]
    for it in itertools.product(*items):
        yield it

if __name__ == "__main__":
    pass

