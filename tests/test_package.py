import pathlib
import re

import lebrel_encrypted

ROOT = pathlib.Path(__file__).resolve().parents[1]


def declared_version():
    return re.search(r'^version = "([^"]+)"$', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.MULTILINE).group(1)


def test_the_package_states_the_version_it_is_published_as():
    assert lebrel_encrypted.__version__ == declared_version()


def test_the_changelog_opens_with_the_published_version():
    latest = re.search(r"^## (\S+)", (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), re.MULTILINE).group(1)
    assert latest == declared_version()
