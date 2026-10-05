"""torchsympy package.

This package is intended to be used as a standalone library:

	import torchsympy
	lt = torchsympy.TorchSymPy()

The implementation lives in :mod:`torchsympy.main` and is re-exported here for
convenient imports.
"""

from .main import (
	MAX_ELEMENTS_PER_EVALUATION,
	MEMORY_FRACTION_PER_EVALUATION,
	TEMPORARIES_PER_EVALUATION,
	TorchExpr,
	TorchSymPy,
	clear_caches,
	elements_budget,
	setup_logging,
)

try:
	from importlib.metadata import version as _pkg_version

	# Distribution name may differ from import package name.
	try:
		__version__ = _pkg_version("torchsympy")
	except Exception:
		__version__ = _pkg_version("torchsympy")
except Exception:
	__version__ = "0.4.3"

__all__ = [
	"MAX_ELEMENTS_PER_EVALUATION",
	"MEMORY_FRACTION_PER_EVALUATION",
	"TEMPORARIES_PER_EVALUATION",
	"TorchExpr",
	"TorchSymPy",
	"clear_caches",
	"elements_budget",
	"setup_logging",
	"__version__",
]
