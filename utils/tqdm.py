from tqdm import tqdm as o_tqdm
import contextlib
import inspect


@contextlib.contextmanager
def _redirect_to_tqdm():
    # Store builtin print
    old_print = print

    def new_print(*args, **kwargs):
        # If tqdm.tqdm.write raises error, use builtin print
        try:
            tqdm.write(*args, **kwargs)
        except:
            old_print(*args, **kwargs)

    try:
        # Globaly replace print with new_print
        inspect.builtins.print = new_print
        yield
    finally:
        inspect.builtins.print = old_print


def tqdm(*args, **kwargs):
    """see tqdm.tqdm for arguments"""
    with _redirect_to_tqdm():
        for x in o_tqdm(*args, **kwargs):
            yield x
