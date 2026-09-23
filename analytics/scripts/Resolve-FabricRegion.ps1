<#
.SYNOPSIS
  Finds an Azure region where this tenant can actually create a Fabric capacity and saves it
  to the azd environment as FABRIC_CAPACITY_LOCATION. Called from the azd preprovision hook.

.DESCRIPTION
  Fabric capacity placement is governed per-tenant: some tenants (commonly lab/trial tenants)
  only accept capacities in the Power BI home region and reject everything else with
  "Location needs to match the PowerBI cluster location". That region can't be read without a
  Power BI license, so we probe instead: PUT a throwaway F2 capacity in each candidate region,
  keep the first region that is accepted, and delete the probe immediately (it lives seconds).

  A region that was already verified is reused without probing. A region that is set but not
  verified (e.g. a manual override) is probed first, so a valid override always wins.
#>
$ErrorActionPreference = 'Continue'

function Get-AzdValue([string]$Name) {
    $v = azd env get-value $Name 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $v) { return '' }
    return "$v".Trim()
}

if ((Get-AzdValue 'DEPLOY_ANALYTICS') -eq 'false') { return }

$existing = Get-AzdValue 'FABRIC_CAPACITY_LOCATION'
$verified = Get-AzdValue 'FABRIC_CAPACITY_LOCATION_VERIFIED'
if ($existing -and $existing -eq $verified) {
    Write-Host "Fabric capacity region: $existing (verified)"
    return
}

$envName  = Get-AzdValue 'AZURE_ENV_NAME'
$appLoc   = Get-AzdValue 'AZURE_LOCATION'
$subId    = Get-AzdValue 'AZURE_SUBSCRIPTION_ID'
$rgName   = Get-AzdValue 'AZURE_RESOURCE_GROUP'
if (-not $rgName) { $rgName = "rg-$envName" }
$admin    = Get-AzdValue 'OWNER_EMAIL'
if (-not $admin) { $admin = (az account show --query user.name -o tsv 2>$null) }

$candidates = @($existing, $appLoc,
    'westus', 'eastus', 'eastus2', 'centralus', 'northcentralus', 'southcentralus', 'westus2', 'westus3',
    'westcentralus', 'canadacentral', 'brazilsouth', 'northeurope', 'westeurope', 'uksouth', 'francecentral',
    'germanywestcentral', 'swedencentral', 'switzerlandnorth', 'norwayeast', 'australiaeast', 'southeastasia',
    'eastasia', 'japaneast', 'koreacentral', 'centralindia', 'southafricanorth', 'uaenorth') |
    Where-Object { $_ } | Select-Object -Unique

Write-Host ""
Write-Host "Analytics is enabled -> detecting a region where this tenant can create a Fabric capacity..."

az provider register --namespace Microsoft.Fabric --wait 2>$null | Out-Null
if ((az group show -n $rgName --query properties.provisioningState -o tsv 2>$null) -eq 'Deleting') {
    Write-Host "  Resource group $rgName is still being deleted; waiting for that to finish..."
    az group wait -n $rgName --deleted --timeout 1800 2>$null
}
if ((az group exists -n $rgName 2>$null) -ne 'true') {
    az group create -n $rgName -l $appLoc --tags "azd-env-name=$envName" -o none 2>$null
}

$base = "https://management.azure.com/subscriptions/$subId/resourceGroups/$rgName/providers/Microsoft.Fabric/capacities"
$bodyFile = New-TemporaryFile
$found = ''
foreach ($region in $candidates) {
    $probe = "fabprobe$(Get-Random -Minimum 10000 -Maximum 99999)"
    @{ location = $region; sku = @{ name = 'F2'; tier = 'Fabric' }
       properties = @{ administration = @{ members = @($admin) } } } |
        ConvertTo-Json -Depth 5 | Set-Content -Path $bodyFile -Encoding utf8
    $out = az rest --method put --url "$base/${probe}?api-version=2023-11-01" --body "@$bodyFile" 2>&1
    if ($LASTEXITCODE -eq 0) {
        az resource delete -g $rgName -n $probe --resource-type Microsoft.Fabric/capacities 2>$null
        $found = $region
        break
    }
    $reason = ("$out" -split "`n" | Where-Object { $_ -match '\S' } | Select-Object -First 1)
    if ("$out" -notmatch 'location|region|LocationNotAvailable|NoRegisteredProviderFound') {
        # Not a placement rejection (e.g. RG being deleted, auth, quota) -- probing more regions won't help.
        Write-Warning "Fabric region detection stopped: $reason"
        break
    }
    Write-Host "  $region -> not allowed ($reason)"
}
Remove-Item $bodyFile -ErrorAction SilentlyContinue

if ($found) {
    if ($existing -and $existing -ne $found) {
        Write-Host "  Saved region '$existing' is not usable on this tenant; replacing it."
    }
    azd env set FABRIC_CAPACITY_LOCATION $found
    azd env set FABRIC_CAPACITY_LOCATION_VERIFIED $found
    Write-Host "Fabric capacity region set to: $found"
} else {
    Write-Warning ("No candidate region accepted a Fabric capacity. Set one manually with " +
        "'azd env set FABRIC_CAPACITY_LOCATION <region>' (Fabric portal -> ? -> About -> 'Your data is stored in'), " +
        "or skip analytics with 'azd env set DEPLOY_ANALYTICS false'.")
}
