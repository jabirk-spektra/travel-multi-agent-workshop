#!/bin/sh
# Finds an Azure region where this tenant can actually create a Fabric capacity and saves it to
# the azd environment as FABRIC_CAPACITY_LOCATION. Called from the azd preprovision hook.
# POSIX twin of Resolve-FabricRegion.ps1 -- see that file for the full rationale.

val() { v=$(azd env get-value "$1" 2>/dev/null) && printf '%s' "$v" | tr -d '\r' || printf ''; }

[ "$(val DEPLOY_ANALYTICS)" = "false" ] && exit 0

EXISTING=$(val FABRIC_CAPACITY_LOCATION)
VERIFIED=$(val FABRIC_CAPACITY_LOCATION_VERIFIED)
if [ -n "$EXISTING" ] && [ "$EXISTING" = "$VERIFIED" ]; then
  echo "Fabric capacity region: $EXISTING (verified)"
  exit 0
fi

ENV_NAME=$(val AZURE_ENV_NAME)
APP_LOC=$(val AZURE_LOCATION)
SUB_ID=$(val AZURE_SUBSCRIPTION_ID)
RG_NAME=$(val AZURE_RESOURCE_GROUP); [ -z "$RG_NAME" ] && RG_NAME="rg-$ENV_NAME"
ADMIN=$(val OWNER_EMAIL); [ -z "$ADMIN" ] && ADMIN=$(az account show --query user.name -o tsv 2>/dev/null)

CANDIDATES="$EXISTING $APP_LOC westus eastus eastus2 centralus northcentralus southcentralus westus2 westus3 \
westcentralus canadacentral brazilsouth northeurope westeurope uksouth francecentral germanywestcentral \
swedencentral switzerlandnorth norwayeast australiaeast southeastasia eastasia japaneast koreacentral \
centralindia southafricanorth uaenorth"

echo ""
echo "Analytics is enabled -> detecting a region where this tenant can create a Fabric capacity..."

az provider register --namespace Microsoft.Fabric --wait >/dev/null 2>&1
if [ "$(az group show -n "$RG_NAME" --query properties.provisioningState -o tsv 2>/dev/null)" = "Deleting" ]; then
  echo "  Resource group $RG_NAME is still being deleted; waiting for that to finish..."
  az group wait -n "$RG_NAME" --deleted --timeout 1800 2>/dev/null
fi
if [ "$(az group exists -n "$RG_NAME" 2>/dev/null)" != "true" ]; then
  az group create -n "$RG_NAME" -l "$APP_LOC" --tags "azd-env-name=$ENV_NAME" -o none 2>/dev/null
fi

BASE="https://management.azure.com/subscriptions/$SUB_ID/resourceGroups/$RG_NAME/providers/Microsoft.Fabric/capacities"
FOUND=""
TRIED=" "
for REGION in $CANDIDATES; do
  case "$TRIED" in *" $REGION "*) continue ;; esac
  TRIED="$TRIED$REGION "
  PROBE="fabprobe$(od -An -N2 -tu2 /dev/urandom | tr -d ' ')"
  BODY="{\"location\":\"$REGION\",\"sku\":{\"name\":\"F2\",\"tier\":\"Fabric\"},\"properties\":{\"administration\":{\"members\":[\"$ADMIN\"]}}}"
  if OUT=$(az rest --method put --url "$BASE/$PROBE?api-version=2023-11-01" --body "$BODY" 2>&1); then
    # Accepted the request -- make sure it actually provisions before trusting the region.
    STATE=""; I=0
    while [ $I -lt 36 ] && [ "$STATE" != "Succeeded" ] && [ "$STATE" != "Failed" ]; do
      sleep 5; I=$((I + 1))
      STATE=$(az rest --method get --url "$BASE/$PROBE?api-version=2023-11-01" --query properties.provisioningState -o tsv 2>/dev/null | tr -d '\r')
    done
    az resource delete -g "$RG_NAME" -n "$PROBE" --resource-type Microsoft.Fabric/capacities >/dev/null 2>&1
    if [ "$STATE" = "Succeeded" ]; then FOUND="$REGION"; break; fi
    echo "  $REGION -> accepted but did not provision (state: $STATE)"
    continue
  fi
  REASON=$(printf '%s' "$OUT" | grep -m1 .)
  if ! printf '%s' "$OUT" | grep -qiE 'location|region|LocationNotAvailable|NoRegisteredProviderFound'; then
    # Not a placement rejection (e.g. RG being deleted, auth, quota) -- probing more regions won't help.
    echo "WARNING: Fabric region detection stopped: $REASON"
    break
  fi
  echo "  $REGION -> not allowed ($REASON)"
done

if [ -n "$FOUND" ]; then
  if [ -n "$EXISTING" ] && [ "$EXISTING" != "$FOUND" ]; then
    echo "  Saved region '$EXISTING' is not usable on this tenant; replacing it."
  fi
  azd env set FABRIC_CAPACITY_LOCATION "$FOUND"
  azd env set FABRIC_CAPACITY_LOCATION_VERIFIED "$FOUND"
  echo "Fabric capacity region set to: $FOUND"
else
  echo "WARNING: No candidate region accepted a Fabric capacity. Set one manually with"
  echo "  azd env set FABRIC_CAPACITY_LOCATION <region>  (Fabric portal -> ? -> About -> 'Your data is stored in')"
  echo "or skip analytics with: azd env set DEPLOY_ANALYTICS false"
fi
