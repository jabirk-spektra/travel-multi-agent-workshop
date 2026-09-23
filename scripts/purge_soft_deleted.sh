#!/bin/sh
# Purges soft-deleted Foundry (Cognitive Services) accounts left behind by a previous deployment of
# this azd environment. POSIX twin of Purge-SoftDeleted.ps1 -- see that file for the rationale.

val() { v=$(azd env get-value "$1" 2>/dev/null) && printf '%s' "$v" | tr -d '\r' || printf ''; }

RG_NAME=$(val AZURE_RESOURCE_GROUP); [ -z "$RG_NAME" ] && RG_NAME="rg-$(val AZURE_ENV_NAME)"

az cognitiveservices account list-deleted --query "[].[id,name,location]" -o tsv 2>/dev/null | tr -d '\r' |
while IFS="$(printf '\t')" read -r ID NAME LOC; do
  case "$(printf '%s' "$ID" | tr 'A-Z' 'a-z')" in
    *"/resourcegroups/$(printf '%s' "$RG_NAME" | tr 'A-Z' 'a-z')/"*)
      echo "Purging soft-deleted Foundry account $NAME ($LOC) from a previous deployment..."
      az cognitiveservices account purge -g "$RG_NAME" -n "$NAME" -l "$LOC"
      ;;
  esac
done
