import warnings

__all__ = [
    'within_no_symbolic_context',
    'set_symbolic_enabled',
    'is_symbolic_enabled',
    'no_symbolic',
]

_within_no_symbolic_context: bool=False
def within_no_symbolic_context() -> bool:
    return _within_no_symbolic_context

_symbolic_enabled: bool=True
def set_symbolic_enabled(enabled: bool=True) -> None:
    # if within_no_symbolic_context():
    #     warnings.warn(
    #         "set_symbolic_enabled: overriding the global symbolic mode within "
    #         "a local no_symbolic context."
    #     )
    global _symbolic_enabled
    _symbolic_enabled = enabled

def is_symbolic_enabled() -> bool:
    return _symbolic_enabled

class no_symbolic:
    def __init__(self):
        self.prev_move = False
        self.prev_within_ctx = False

    def __enter__(self):
        self.prev_mode = is_symbolic_enabled()
        set_symbolic_enabled(False)

        global _within_no_symbolic_context
        self.prev_within_ctx = _within_no_symbolic_context
        _within_no_symbolic_context = True

    def __exit__(self, exc_type, exc_value, exc_traceback):
        global _within_no_symbolic_context
        _within_no_symbolic_context = self.prev_within_ctx

        set_symbolic_enabled(self.prev_mode)
