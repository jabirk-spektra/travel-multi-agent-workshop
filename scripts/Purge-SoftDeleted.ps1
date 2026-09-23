<#
.SYNOPSIS
  Purges soft-deleted Foundry (Cognitive Services) accounts left behind by a previous deployment of
  this azd environment. Called from the azd preprovision hook.

.DESCRIPTION
  Deleting the resource group (azd down without --purge, or the portal) only soft-deletes the Foundry
  account. Redeploying with the same name then fails validation with FlagMustBeSetForRestore. Only
  accounts that were deleted from THIS environment's resource group are purged.
#>
$ErrorActionPreference = 'Continue'

function Get-AzdValue([string]$Name) {
    $v = azd env get-value $Name 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $v) { return '' }
    return "$v".Trim()
}

$rgName = Get-AzdValue 'AZURE_RESOURCE_GROUP'
if (-not $rgName) { $rgName = "rg-$(Get-AzdValue 'AZURE_ENV_NAME')" }

$deleted = az cognitiveservices account list-deleted --query "[].{id:id,name:name,location:location}" -o json 2>$null |
    ConvertFrom-Json
foreach ($acct in @($deleted | Where-Object { $_.id -match "/resourceGroups/$([regex]::Escape($rgName))/" })) {
    Write-Host "Purging soft-deleted Foundry account $($acct.name) ($($acct.location)) from a previous deployment..."
    az cognitiveservices account purge -g $rgName -n $acct.name -l $acct.location 2>&1 | Out-Host
}
