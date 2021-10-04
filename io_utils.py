import os
import sys
import tempfile
import threading
from shutil import copyfileobj, rmtree
from contextlib import contextmanager

import ctypes
from ctypes.util import find_library
try:
    libc = ctypes.cdll.msvcrt # Windows
except OSError:
    libc = ctypes.cdll.LoadLibrary(find_library('c'))

class AutoDeleteDir:
    """
    Stores a directory path and delete that
    directory once this object goes out of scope,
    using __del__(). CPython only.
    """ 
    def __init__(self, dirpath=None):
        if dirpath is None:
            dirpath = tempfile.mkdtemp()
        self.dirpath = dirpath
        self.skip_delete = False
    
    def __del__(self):
        if not self.skip_delete:
            rmtree(self.dirpath)

    def __str__(self):
        return self.dirpath


class TemporaryNamedPipe:
    """
    Represents a unix 'named pipe', a.k.a. fifo
    The pipe is created in a temporary directory and deleted upon cleanup() (or __exit__).

    Example:

        with TemporaryNamedPipe() as pipe:
            def write_hello():
                with pipe.open_stream('r') as f:
                    f.write("Hello")
            threading.Thread(target=write_hello).start()

            subprocess.call(f'cat {pipe.path}')
    """
    def __init__(self, basename='temporary-pipe.bin'):
        self.state = 'uninitialized'
        assert '/' not in basename
        self.tmpdir = tempfile.mkdtemp()
        self.path = f"{self.tmpdir}/{basename}"

        os.mkfifo(self.path)
        self.state = 'pipe_exists'
        self.writer_thread = None

    def cleanup(self):
        if self.path:
            os.unlink(self.path)
            os.rmdir(self.tmpdir)
            self.path = None

    def __del__(self):
        self.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()

    def start_writing_stream(self, stream):
        def write_input():
            with open(self.path, 'wb') as f:
                copyfileobj(stream, f)
        self.writer_thread = threading.Thread(target=write_input)
        self.writer_thread.start()
        return self.writer_thread

    def open_stream(self, mode):
        return TemporaryNamedPipe.Stream(mode, self)

    class Stream:
        """
        An open stream to a parent TemporaryNamedPipe.
        Retains a reference to the parent so the pipe isn't deleted before stream is closed.
        """
        def __init__(self, mode, parent_pipe):
            self._parent_pipe = parent_pipe
            self._file = open(parent_pipe.path, mode)
            self.closed = False
            self.mode = mode
            self.name = self._file.name

        def close(self):
            if self._parent_pipe:
                self._file.close()
                self._parent_pipe = None  # Release parent pipe
                self.closed = True

        def __del__(self):
            self.close()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def __iter__(self):
            return self._file.__iter__()

        def fileno(self, *args, **kwargs): return self._file.fileno(*args, **kwargs)
        def flush(self, *args, **kwargs): return self._file.flush(*args, **kwargs)
        def isatty(self, *args, **kwargs): return self._file.isatty(*args, **kwargs)
        def readable(self, *args, **kwargs): return self._file.readable(*args, **kwargs)
        def readline(self, *args, **kwargs): return self._file.readline(*args, **kwargs)
        def readlines(self, *args, **kwargs): return self._file.readlines(*args, **kwargs)
        def seek(self, *args, **kwargs): return self._file.seek(*args, **kwargs)
        def tell(self, *args, **kwargs): return self._file.tell(*args, **kwargs)
        def truncate(self, *args, **kwargs): return self._file.truncate(*args, **kwargs)
        def writable(self, *args, **kwargs): return self._file.writable(*args, **kwargs)

        def read(self, *args, **kwargs): return self._file.read(*args, **kwargs)
        def readall(self, *args, **kwargs): return self._file.readall(*args, **kwargs)
        def readinto(self, *args, **kwargs): return self._file.readinto(*args, **kwargs)
        def write(self, *args, **kwargs): return self._file.write(*args, **kwargs)


@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    Context manager.

    Redirects a file object or file descriptor to a new file descriptor.

    Example:
    with open('my-stdout.txt', 'w') as f:
        with stdout_redirected(f):
            print('Writing to my-stdout.txt')

    Motivation: In pure-Python, you can redirect all print() statements like this:

        sys.stdout = open('myfile.txt')

        ...but that doesn't redirect any compiled printf() (or std::cout) output
        from C/C++ extension modules.

    This context manager uses a superior approach, based on low-level Unix file
    descriptors, which redirects both Python AND C/C++ output.

    Lifted from the following link (with minor edits):
    https://stackoverflow.com/a/22434262/162094
    (MIT License)
    """
    if stdout is None:
        stdout = sys.stdout

    stdout_fd = fileno(stdout)

    try:
        if fileno(to) == stdout_fd:
            # Nothing to do; early return
            yield stdout
            return
    except ValueError:  # filename
        pass

    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        flush(stdout)  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            flush(stdout)
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


def flush(stream):
    try:
        libc.fflush(None)  # Flush all C stdio buffers
        stream.flush()
    except (AttributeError, ValueError, IOError):
        pass  # unsupported


def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd
