import time

class Timer:
    depth = 0
    max_depth = 2
    default_callback = print
    default_informat = " {:>20}"
    default_outformat = " {:>20} {:>9.2f} s"
    def __init__(self, label, outformat=None, informat=None, callback=None):
        self.label = label
        self.callback = callback or self.default_callback
        self.outformat = outformat or self.default_outformat
        self.informat = informat or self.default_informat

    def __enter__(self):
        Timer.depth += 1
        if Timer.depth <= Timer.max_depth:
            self.callback(">"*Timer.depth+self.informat.format(self.label))
        self._starttime = time.monotonic()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if Timer.depth <= Timer.max_depth:
            self.callback("<"*Timer.depth + self.outformat.format(self.label, time.monotonic()-self._starttime))
        Timer.depth -= 1
        
