"""Entry point for `python -m zemble.daemon ...`, which is how a client spawns one."""

import sys

from zemble.daemon.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
