#!/usr/bin/env bash
# Removes "Co-Authored-By: Claude", "Claude-Session:" and "Generated with [Claude Code]"
# lines from commit messages in every non-fork, non-archived repo you own on GitHub.
#
#   ./remove-claude-coauthor.sh            # dry run: only lists affected repos (changes nothing)
#   ./remove-claude-coauthor.sh --apply    # backs up, rewrites history, force-pushes
#
# Needs: git and the GitHub CLI (brew install gh && gh auth login)

set -uo pipefail

OWNER="${OWNER:-amey000}"
APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

PATTERN='^(co-authored-by: claude|claude-session:|.*generated with \[claude code\])'
WORK="$(mktemp -d)"
BACKUP="$HOME/claude-coauthor-backup-$(date +%Y%m%d-%H%M%S)"
export FILTER_BRANCH_SQUELCH_WARNING=1

command -v gh >/dev/null 2>&1 || { echo "GitHub CLI not found. Run: brew install gh && gh auth login"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "Not logged in. Run: gh auth login"; exit 1; }

echo "Scanning repos owned by $OWNER..."
REPOS="$(gh repo list "$OWNER" --limit 1000 --source --no-archived --json nameWithOwner -q '.[].nameWithOwner')"

affected=0; done_ok=0; failed=""

while IFS= read -r repo; do
  [ -z "$repo" ] && continue
  dir="$WORK/${repo//\//__}.git"
  if ! git clone --quiet --mirror "https://github.com/$repo.git" "$dir" 2>/dev/null; then
    echo "  ! $repo: could not clone (empty repo?) - skipped"
    continue
  fi
  git -C "$dir" config --unset remote.origin.mirror

  n="$(git -C "$dir" log --branches --tags -i -E --grep="$PATTERN" --format=%H | sort -u | wc -l | tr -d ' ')"
  [ "$n" = "0" ] && continue

  affected=$((affected + 1))
  echo "  * $repo: $n commit(s) with Claude attribution"

  [ "$APPLY" = "1" ] || continue

  mkdir -p "$BACKUP"
  git -C "$dir" bundle create --quiet "$BACKUP/${repo//\//__}.bundle" --branches --tags

  if ! git -C "$dir" filter-branch -f \
        --msg-filter "grep -v -i -E '$PATTERN' | git stripspace" \
        --tag-name-filter cat -- --branches --tags >/dev/null 2>&1; then
    echo "    ! rewrite failed - nothing pushed"; failed="$failed $repo"; continue
  fi

  left="$(git -C "$dir" log --branches --tags -i -E --grep="$PATTERN" --format=%H | wc -l | tr -d ' ')"
  if [ "$left" != "0" ]; then
    echo "    ! $left commit(s) still match after rewrite - nothing pushed"; failed="$failed $repo"; continue
  fi

  if git -C "$dir" push --quiet --force origin 'refs/heads/*:refs/heads/*' 2>"$dir.err" &&
     git -C "$dir" push --quiet --force --tags origin 2>>"$dir.err"; then
    echo "    -> cleaned and force-pushed"
    done_ok=$((done_ok + 1))
  else
    echo "    ! push refused (branch protection?):"; sed 's/^/      /' "$dir.err" | head -5
    failed="$failed $repo"
  fi
done <<< "$REPOS"

echo
if [ "$APPLY" = "1" ]; then
  echo "Done: $done_ok of $affected affected repo(s) cleaned."
  [ -n "$failed" ] && echo "Needs attention:$failed"
  [ -d "$BACKUP" ] && echo "Backups of the original history: $BACKUP"
  echo "Re-clone (or 'git fetch && git reset --hard origin/<branch>') any local copies of these repos."
else
  echo "$affected repo(s) affected. Nothing was changed."
  [ "$affected" -gt 0 ] && echo "Run again with --apply to rewrite and force-push them."
fi
rm -rf "$WORK"
