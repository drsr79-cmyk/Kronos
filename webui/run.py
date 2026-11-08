#!/usr/bin/env python3
"""Start the local Kronos research UI. Does not install packages or open a debugger."""

import os
import sys

_WEBUI = os.path.dirname(os.path.abspath(__file__))
if _WEBUI not in sys.path:
    sys.path.insert(0, _WEBUI)


def main():
    os.environ.pop("FLASK_DEBUG", None)
    os.environ.pop("FLASK_ENV", None)
    from app import main as serve

    serve()


if __name__ == "__main__":
    main()
