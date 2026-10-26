import importlib.util
import sys
from pathlib import Path

# The package code imports itself as `knowledge`. In this repository the package
# sits at the repository root, so register the root as the `knowledge` package
# when it is not already importable (for example in a monorepo checkout).
REPO_ROOT = Path(__file__).resolve().parent.parent

if importlib.util.find_spec("knowledge") is None:
    spec = importlib.util.spec_from_file_location(
        "knowledge",
        REPO_ROOT / "__init__.py",
        submodule_search_locations=[str(REPO_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["knowledge"] = module
    spec.loader.exec_module(module)
