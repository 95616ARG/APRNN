import time
from sytorch.pervasives import *

async def _await(coro):
    return await coro

def _prepare_pipeline(*args):
    pipeline = []

    for arg in args:
        if isinstance(arg, asyncio.Queue):
            assert len(pipeline) == 0 or not isinstance(pipeline[-1], asyncio.Queue)
        elif callable(arg):
            if len(pipeline) == 0 or not isinstance(pipeline[-1], asyncio.Queue):
                pipeline.append(asyncio.Queue())
        else:
            raise TypeError
        pipeline.append(arg)

    if len(pipeline) == 0 or not isinstance(pipeline[-1], asyncio.Queue):
        pipeline.append(asyncio.Queue())

    return tuple(pipeline)

class _close_signal: ...

async def async_pipe(src: asyncio.Queue, fn, dst: asyncio.Queue) -> None:
    closed = False
    while not closed:
        obj = await src.get()
        print(f"pipe received {obj}.")

        if obj is _close_signal:
            print(f"pipe received close signal.")
            closed = True

        else:
            if asyncio.iscoroutinefunction(fn):
                obj = await fn(obj)
            else:
                obj = fn(obj)

        await dst.put(obj)
        print(f"pipe passed {obj}.")
        src.task_done()

    assert src.empty()
    print("pipe returned.")
    return

class AsyncPipeline:
    def __init__(self, *args):
        """
        Parameters
        ==========
        args: [Queue, ] { Callable [, Queue] }
        """
        self.pipeline = _prepare_pipeline(*args)
        self.tasks = ()
        self.open()

    @property
    def coroutines(self):
        return tuple(task.get_coro() for task in self.tasks)

    def exceptions(self):
        es = []
        for task in self.tasks:
            try:
                if task.exception() is not None:
                    es.append(task.exception())
            except asyncio.InvalidStateError:
                pass
        return tuple(es)

    def _handle_exception(self):
        for e in self.exceptions():
            raise e

    async def put(self, args):
        return await self.pipeline[0].put(args)

    def put_nowait(self, args):
        return self.pipeline[0].put_nowait(args)

    async def get(self):
        return await self.pipeline[-1].get()

    def get_nowait(self):
        return self.pipeline[-1].get_nowait()

    def done(self):
        return all(task.done() for task in self.tasks)

    def results(self):
        return tuple(task.result() for task in self.tasks)

    def open(self):
        # self.join()
        assert self.done()
        self.tasks = tuple(
            asyncio.create_task(async_pipe(*self.pipeline[i:i+3]))
                if i+3 <= len(self.pipeline) else
            _assert(i == len(self.pipeline) - 1)
            for i in range(0, len(self.pipeline)-2, 2)
        )

    def close(self):
        return self.put_nowait(_close_signal)

    async def join(self):
        if self.done():
            self._handle_exception()
            return

        self.close()
        for task in self.tasks:
            print(f'await {task}.')
            await task
        self._handle_exception()
        return

    def join_sync(self):
        raise NotImplementedError("WIP.")
        if self.done():
            self._handle_exception()
            return self.results()

        self.close()

        # NOTE(anonymous): For now manually join tasks because
        # `asyncio.run_until_complete()` doesn't work with Jupyter Notebook.
        # There might be a better way to join under all conditions.
        asyncio.run_until_complete()
        while True:
            if self.done():
                self._handle_exception()
                return self.results()
            else:
                self._handle_exception()
                print('wait')
                asyncio.sleep(1.)

    @property
    def queues(self):
        return self.pipeline[0::2]

    @property
    def funcs(self):
        return self.pipeline[1::2]

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.done() and self.empty():
            raise StopAsyncIteration
        else:
            return await self.get()
