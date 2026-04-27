import pathlib

import pytest


@pytest.fixture
def tmp_corpus(tmp_path: pathlib.Path) -> pathlib.Path:
    """A temporary directory pre-populated with sample docs."""
    (tmp_path / "intro.md").write_text(
        "# Introduction\n\n## Overview\n\nSome overview text.\n\n## Details\n\nDetail text.\n"
    )
    (tmp_path / "guide.md").write_text(
        "# User Guide\n\n## Installation\n\n```bash\npip install foo\n```"
        "\n\n## Usage\n\nUsage info.\n"
    )
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.md").write_text("# Nested Doc\n\nSome flat content.\n")
    return tmp_path
