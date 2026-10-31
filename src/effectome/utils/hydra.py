"""Hydra CLI with Python 3.14 help-string compatibility.

Hydra 1.3 passes a lazy object as shell-completion help. Python 3.14
validates help strings during argument creation and rejects that object.
See https://github.com/facebookresearch/hydra/issues/3121.
Remove this shim when the stable Hydra parser handles Python 3.14 directly.
"""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace
from typing import Any

import hydra
from hydra._internal import utils as hydra_utils


class _HelpStringParser(argparse.ArgumentParser):
    def add_argument(self, *args: str, **kwargs: Any) -> argparse.Action:
        help_text = kwargs.get("help")
        if help_text is not None and not isinstance(help_text, str):
            kwargs["help"] = str(help_text)
        return super().add_argument(*args, **kwargs)


if sys.version_info >= (3, 14):
    # Replace only Hydra's module-local argparse reference, leaving the standard
    # library and other argument parsers untouched.
    hydra_utils.argparse = SimpleNamespace(  # type: ignore[assignment]
        ArgumentParser=_HelpStringParser,
        Namespace=argparse.Namespace,
    )

main = hydra.main
