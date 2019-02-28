import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
from typing import Callable
import time


class ParallelExecutor(object):
    def __init__(self, pool='thread', *args, **kwargs):
        self.pool = None
        self.pool_cls = None
        if pool == 'thread':
            self.pool_cls = ThreadPool
        elif pool == 'process':
            self.pool_cls = multiprocessing.Pool
        self.pool_args = args
        self.pool_kwargs = kwargs
        self.handlers = []

    def dispatch(self, fn: Callable, *args, **kwargs):
        handler = self.pool.apply_async(fn, args=args, kwds=kwargs)
        self.handlers.append(handler)
        return handler

    def iterate_results(self):
        for h in self.handlers:
            yield h.get()

    def __del__(self):
        if self.pool:
            self.pool.close()

    def __enter__(self):
        if self.pool_cls:
            self.pool = self.pool_cls(*self.pool_args, **self.pool_kwargs)
        return self

    def __exit__(self, *exc):
        if self.pool:
            self.pool.close()
            self.pool.join()
        return False


def __square(a):
    time.sleep(2)
    return a ** 2


if __name__ == '__main__':
    p = ParallelExecutor('thread')
    with p:
        for i in range(100):
            p.dispatch(__square, i)

    for res in p.iterate_results():
        print(res)
