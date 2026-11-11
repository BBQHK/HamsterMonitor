#!/usr/bin/env python3
"""Entrypoint for the hamster cage hardware service.

Kept as a thin launcher so existing systemd units and scripts that point at
hardware/start.py keep working. Application wiring lives in app.py.
"""

from app import main

if __name__ == "__main__":
    main()
