"""DimGrad: a source-to-source compiler that checks the units of derivatives and emits C."""
from .glue.compiler import check_source, compile_source
from .glue.errors import DimGradError

__all__ = ["check_source", "compile_source", "DimGradError"]
