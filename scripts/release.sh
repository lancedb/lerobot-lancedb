#!/usr/bin/env bash
# Build and publish lerobot-lancedb to PyPI (or TestPyPI with --test).
#
# Usage:
#     PYPI_TOKEN=pypi-... scripts/release.sh           # publish to real PyPI
#     PYPI_TOKEN=pypi-... scripts/release.sh --test    # publish to TestPyPI
#     scripts/release.sh --build-only                  # build + check, skip upload
#
# Token is read from $PYPI_TOKEN (TestPyPI: $TEST_PYPI_TOKEN). The script never
# writes the token to disk; twine receives it via TWINE_PASSWORD.
#
# After publish, verify with:
#     pip install --upgrade lerobot-lancedb && python -c "import lerobot_lancedb; print(lerobot_lancedb.__version__ if hasattr(lerobot_lancedb,'__version__') else 'ok')"

set -euo pipefail

cd "$(dirname "$0")/.."

mode="publish"          # publish | test | build-only
repo_url=""             # twine --repository-url (TestPyPI override)
token_var="PYPI_TOKEN"

for arg in "$@"; do
    case "$arg" in
        --test)
            mode="test"
            repo_url="https://test.pypi.org/legacy/"
            token_var="TEST_PYPI_TOKEN"
            ;;
        --build-only)
            mode="build-only"
            ;;
        -h|--help)
            sed -n '2,15p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

# Read version from pyproject.toml (avoids drift between tag and metadata).
version=$(python -c "import tomllib,pathlib; print(tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['version'])")
name=$(python -c "import tomllib,pathlib; print(tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['name'])")
echo "==> Releasing ${name} ${version} (mode=${mode})"

# Refuse to publish an uncommitted tree — too easy to ship the wrong bytes.
if [[ "$mode" != "build-only" ]] && [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: working tree is dirty. Commit or stash before publishing." >&2
    git status --short >&2
    exit 1
fi

echo "==> Cleaning dist/ and build/"
rm -rf dist/ build/ src/*.egg-info

echo "==> Building sdist + wheel"
python -m build

echo "==> Running twine check"
python -m twine check dist/*

if [[ "$mode" == "build-only" ]]; then
    echo "==> build-only mode; skipping upload."
    ls -lh dist/
    exit 0
fi

# Resolve token. Don't echo it.
if [[ -z "${!token_var:-}" ]]; then
    echo "ERROR: \$${token_var} is not set. Export it before running this script." >&2
    exit 1
fi

echo "==> Uploading to ${mode} ($([[ -n $repo_url ]] && echo $repo_url || echo pypi.org))"
twine_args=(--non-interactive --username __token__)
if [[ -n "$repo_url" ]]; then
    twine_args+=(--repository-url "$repo_url")
fi
TWINE_PASSWORD="${!token_var}" python -m twine upload "${twine_args[@]}" dist/*

echo "==> Upload complete."
if [[ "$mode" == "test" ]]; then
    echo "    Verify: pip install --index-url https://test.pypi.org/simple/ ${name}==${version}"
else
    echo "    Verify: pip install ${name}==${version}"
fi
echo "    Tag the release: git tag v${version} && git push origin v${version}"
