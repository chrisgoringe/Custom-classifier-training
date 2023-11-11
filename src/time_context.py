import time

class Timer:
    depth = 0
    def __init__(self, label, format="{:>10} {:>9.2f} s", callback=print):
        self.label = label
        self.callback = callback
        self.format = format

    def __enter__(self):
        Timer.depth += 1
        self._starttime = time.monotonic()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.callback(">"*Timer.depth + self.format.format(self.label, time.monotonic()-self._starttime))
        Timer.depth -= 1
        
