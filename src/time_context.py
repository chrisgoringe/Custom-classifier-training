import time

class Timer:
    depth = 0
    max_depth = 2
    default_callback = print
    default_informat = " {:>20}"
    default_outformat = " {:>20} {:>9.2f} s"
    stack = []

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
        Timer.stack.append(self)
        return self.message

    def __exit__(self, exc_type, exc_val, exc_tb):
        if Timer.depth <= Timer.max_depth:
            self.callback("<"*Timer.depth + self.outformat.format(self.label, time.monotonic()-self._starttime))
        Timer.depth -= 1
        Timer.stack = Timer.stack[:-1]
        
    @classmethod
    def message(cls, m):
        if m: cls.stack[-1].callback(" "*Timer.depth+" "+m)
        return time.monotonic()-cls.stack[-1]._starttime
    
if len(Timer.stack)==0:
    Timer.stack = [Timer(""),]