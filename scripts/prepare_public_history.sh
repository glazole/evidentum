#!/usr/bin/env bash
# Prepare a new, backed-up mirror; never push or change repository visibility.
set -euo pipefail

repo_url="${1:-https://github.com/glazole/evidentum.git}"
release_dir="${2:-evidentum-public-release}"

command -v git >/dev/null
git filter-repo --version >/dev/null 2>&1 || {
  echo 'Install git-filter-repo first: python -m pip install git-filter-repo' >&2
  exit 1
}
if [[ -e "$release_dir" ]]; then
  echo 'Destination already exists; choose a new directory.' >&2
  exit 1
fi
mkdir -p "$release_dir"
release_dir="$(cd "$release_dir" && pwd)"
mirror="$release_dir/evidentum.git"

git clone --mirror "$repo_url" "$mirror"
git -C "$mirror" bundle create "$release_dir/private-backup.bundle" --all
git -C "$mirror" for-each-ref --format='%(refname) %(objectname)' \
  refs/heads refs/tags > "$release_dir/original-refs.txt"

# Remove only the explicitly authorized backup branches from this new mirror.
for branch in backup_2026-04-17_19-28 backup_2026-04-18_18-24 backup-20260516-2238; do
  ref="refs/heads/$branch"
  if git -C "$mirror" show-ref --verify --quiet "$ref"; then
    git -C "$mirror" update-ref -d "$ref"
  fi
done

# Globs match paths including nested directories and all historic PDF filenames.
git -C "$mirror" filter-repo --force --invert-paths \
  --path-glob '*.[pP][dD][fF]'

if git -C "$mirror" rev-list --objects --all | \
    grep -Ei '\.pdf$' > "$release_dir/remaining-pdfs.txt"; then
  echo 'PDF paths remain; inspect remaining-pdfs.txt. Nothing has been pushed.' >&2
  exit 1
fi

# filter-repo removes origin intentionally. Restore it only on the new mirror.
git -C "$mirror" remote remove origin 2>/dev/null || true
git -C "$mirror" remote add origin "$repo_url"

# Prepare a single explicit, leased push for all recorded branches and tags.
# The command is a review artifact and is NEVER executed by this script.
push_script="$release_dir/push-reviewed-refs.sh"
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\n'
  printf '# Review all rewritten refs before running. This replaces remote history.\n'
  printf 'git -C %q push --atomic origin' "$mirror"
  while read -r ref old_sha; do
    printf ' %q' "--force-with-lease=$ref:$old_sha"
  done < "$release_dir/original-refs.txt"
  while read -r ref old_sha; do
    if git -C "$mirror" show-ref --verify --quiet "$ref"; then
      printf ' %q' "$ref:$ref"
    else
      printf ' %q' ":$ref"
    fi
  done < "$release_dir/original-refs.txt"
  printf '\n'
} > "$push_script"

printf 'Prepared mirror: %s\nPrivate backup: %s\nReview-only push script: %s\n' \
  "$mirror" "$release_dir/private-backup.bundle" "$push_script"
echo 'Keep the backup private. No remote refs or visibility settings were changed.'
