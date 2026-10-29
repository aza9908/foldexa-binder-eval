#!/usr/bin/env bash
# Populate the pinned upstream repositories under external/.
#
#   scripts/setup_upstream.sh          ipsae + bindcraft  (~12 MB)
#   scripts/setup_upstream.sh --all    adds boltzgen and rf3-foundry (~300 MB)
#   scripts/setup_upstream.sh --list   show what is pinned and what is present
#
# Only ipsae is needed to run the test suite in full; the reference
# regression in tests/test_ipsae.py skips without it. bindcraft is small and
# is what the default thresholds are checked against. boltzgen and
# rf3-foundry are large and only needed if you are running those tools from
# this checkout rather than their own environments.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SMALL=(external/ipsae external/bindcraft)
LARGE=(external/boltzgen external/rf3-foundry)

if [[ ! -f .gitmodules ]]; then
    echo "error: .gitmodules not found; run this from a git checkout." >&2
    exit 1
fi

list_status() {
    printf '%-26s %-52s %s\n' "PATH" "URL" "STATUS"
    while read -r path url; do
        if [[ -n "$(ls -A "$path" 2>/dev/null)" ]]; then
            status="present"
        else
            status="not cloned"
        fi
        printf '%-26s %-52s %s\n' "$path" "$url" "$status"
    done < <(
        git config --file .gitmodules --get-regexp '^submodule\..*\.path$' |
        while read -r key path; do
            name="${key#submodule.}"; name="${name%.path}"
            url="$(git config --file .gitmodules --get "submodule.${name}.url")"
            echo "$path $url"
        done
    )
}

if [[ "${1:-}" == "--list" ]]; then
    list_status
    exit 0
fi

targets=("${SMALL[@]}")
if [[ "${1:-}" == "--all" ]]; then
    targets+=("${LARGE[@]}")
elif [[ -n "${1:-}" ]]; then
    echo "error: unknown option '$1' (expected --all or --list)" >&2
    exit 1
fi

echo "Fetching ${#targets[@]} submodule(s). This needs network access to github.com."
for path in "${targets[@]}"; do
    if [[ -n "$(ls -A "$path" 2>/dev/null)" ]]; then
        echo "  $path already present, skipping"
        continue
    fi
    echo "  $path ..."
    if ! git submodule update --init --depth 1 -- "$path"; then
        echo "  warning: could not fetch $path; continuing" >&2
    fi
done

echo
list_status
echo
echo "Done. Run 'pytest' to include the ipSAE reference regression."
