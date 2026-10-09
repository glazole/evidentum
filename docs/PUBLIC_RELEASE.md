# Public release preparation

## Audit on 2026-10-09

Baseline `main`: `6d86a79184736862a78d40de04443e4c88f5d0da`.

The audit enumerated all seven branch histories through the connected GitHub API:
89 unique reachable commits, 87 unique trees, and 190 unique text blobs. Commit
lists were below the 100-item page limit and recursive trees were not truncated.
No tracked `.env`, private-key file or notebook was found in those trees.

Pattern checks covered provider-token formats, private-key headers, credential
assignments and private IP addresses. Reviewed matches were function calls and
documentation placeholders. No evident live credential was identified. This is a
bounded static review, not a guarantee: PDF contents, inaccessible/unreachable Git
objects, tags, GitHub attachments, releases and Actions artifacts were not audited.

Six PDFs are removed from the proposed tip. History contains ten distinct PDF
blobs, including old filenames and older document versions. A normal merge does
not erase them from history or other branches.

Application files under `app/`, migrations, prompts, provider clients and existing
tests are unchanged. Original Yandex translation defaults are preserved. Nginx and
Compose changes remove project-specific network/domain assumptions while retaining
the primary API/UI startup and the optional full HTTPS/Certbot deployment.

## Finish before changing visibility

1. Apply the proposed packaging/documentation changes while the repository is private.
2. The preparation script deletes only these backup branches from its new mirror:
   `backup_2026-04-17_19-28`, `backup_2026-04-18_18-24`,
   `backup-20260516-2238`. Feature, homelab and hardening branches are not backups
   by name and are not automatically deleted.
3. Prepare a fresh mirror and private recovery bundle outside the working checkout:

   ```bash
   python -m pip install git-filter-repo
   bash /path/to/evidentum/scripts/prepare_public_history.sh \
     https://github.com/glazole/evidentum.git /path/to/new-release-directory
   ```

4. Inspect the cleaned mirror and generated `push-reviewed-refs.sh`, which also
   deletes those three remote backup refs when run. The preparation
   script never runs that push. Its generated command uses atomic updates and a
   separate force-with-lease condition for every branch/tag; a changed remote ref
   requires a fresh mirror and review. Keep `private-backup.bundle` private.
5. Replace the reviewed remote refs, verify every remaining branch is PDF-free,
   and check any tags and PR refs before opening the repository. Old commit links
   and GitHub cached views may remain accessible after rewriting refs; removal
   from GitHub's retained objects can require GitHub Support.
6. Change visibility only after those checks. Existing local clones should be
   replaced or carefully reset so old PDF-bearing history is not pushed back.

The connected GitHub API used for this preparation supports commits, trees and
branch updates, but exposes no branch-deletion operation or authenticated Git clone.
Deleting remote backup branches and a complete history rewrite require an available
Git client with repository access or an explicitly approved browser fallback.

## Validation

The nine existing unit tests passed in the preparation environment. Both Compose files passed Docker Compose v2 configuration validation. Ignore rules
and source-file preservation were checked. The history-preparation script passed
a fixture test covering renamed PDFs across branches/tags, a recovery bundle and
unchanged application code. Nginx template substitution preserves request variables.
Container and nginx startup remain unverified because this environment has no Docker
daemon or nginx executable.
