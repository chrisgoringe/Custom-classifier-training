import argparse

class CommentArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        if arg_line.startswith('#'): return [] 
        line = "=".join(a.strip() for a in arg_line.split('='))
        return [line,] if len(line) else []