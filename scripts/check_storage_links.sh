#!/bin/bash
# Verify (and, where safe, repair) the per-machine storage symlinks.
#
# The links are NOT in git -- they point into this machine's scratch, and the one
# time `release/data` was committed, everyone who pulled lost the directory it
# shadowed (.gitignore, docs/setup.md). So the expected targets live in LOCAL git
# config instead, which is per-machine by construction:
#
#   git config --add simdist.storagelink "release/data=/scratch/<you>/simdist/data"
#   git config --add simdist.storagelink "exps=/scratch/<you>/simdist/exps"
#
# Run standalone, or automatically after checkout/merge/rebase via
# `git config core.hooksPath scripts/githooks`.
#
# Exit 0 = all links fine (or repaired). Exit 1 = something needs a human.
set -uo pipefail

cd "$(git rev-parse --show-toplevel)" || exit 0
rc=0
n=0

while IFS= read -r entry; do
  [ -n "$entry" ] || continue
  n=$((n + 1))
  path="${entry%%=*}"
  target="${entry#*=}"
  if [ "$path" = "$entry" ] || [ -z "$target" ]; then
    echo "storage-links: malformed simdist.storagelink entry: '$entry' (want path=target)" >&2
    rc=1
    continue
  fi

  # A TRACKED path here is the original incident: git will replace the link.
  if git ls-files --error-unmatch -- "$path" >/dev/null 2>&1; then
    echo "storage-links: DANGER: '$path' is TRACKED by git but is a storage link." >&2
    echo "               git will overwrite it on checkout. Untrack it now:" >&2
    echo "                 git rm --cached -r '$path' && git commit -m 'untrack $path'" >&2
    rc=1
  fi

  if [ -L "$path" ]; then
    actual=$(readlink "$path")
    if [ "$actual" != "$target" ]; then
      echo "storage-links: '$path' points to '$actual', expected '$target'" >&2
      rc=1
    fi
  elif [ -e "$path" ]; then
    # a real file/dir where a link belongs -- do NOT touch it, it may be data
    echo "storage-links: '$path' exists but is NOT a symlink. Leaving it alone." >&2
    echo "               Expected: $path -> $target" >&2
    rc=1
  else
    if [ -d "$target" ]; then
      mkdir -p "$(dirname "$path")" 2>/dev/null
      ln -s "$target" "$path" && \
        echo "storage-links: RESTORED missing link $path -> $target" >&2
    else
      echo "storage-links: '$path' is missing AND its target '$target' does not exist." >&2
      echo "               Nothing was restored. Check whether scratch was purged." >&2
      rc=1
    fi
    continue
  fi

  if [ ! -d "$path/" ]; then
    echo "storage-links: '$path' -> '$(readlink "$path")' does not resolve to a" >&2
    echo "               directory. The link is intact but its target is gone" >&2
    echo "               (scratch purge?)." >&2
    rc=1
  fi
done < <(git config --get-all simdist.storagelink)

if [ "$n" -eq 0 ]; then
  echo "storage-links: no links registered on this machine. If you use" >&2
  echo "  release/data / exps symlinks, register them (see the header of" >&2
  echo "  scripts/check_storage_links.sh) so they are checked automatically." >&2
fi

exit "$rc"
