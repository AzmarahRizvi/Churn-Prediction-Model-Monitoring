param(
    [Parameter(Mandatory = $true)][string]$ProjectId,
    [string]$Region = "us-central1",
    [Parameter(Mandatory = $true)][string]$EndpointId,
    [Parameter(Mandatory = $true)][string]$ModelId
)

$ErrorActionPreference = "Stop"
$deployedModelIds = gcloud ai endpoints describe $EndpointId --region=$Region --project=$ProjectId --format="value(deployedModels.id)"
if ($LASTEXITCODE -ne 0) { throw "Could not read deployed models from endpoint $EndpointId" }
foreach ($deployedModelId in $deployedModelIds) {
    if (-not [string]::IsNullOrWhiteSpace($deployedModelId)) {
        gcloud ai endpoints undeploy-model $EndpointId --region=$Region --deployed-model-id=$deployedModelId --project=$ProjectId
        if ($LASTEXITCODE -ne 0) { throw "Could not undeploy model $deployedModelId" }
    }
}
gcloud ai endpoints delete $EndpointId --region=$Region --quiet --project=$ProjectId
if ($LASTEXITCODE -ne 0) { throw "Could not delete endpoint $EndpointId" }
gcloud ai models delete $ModelId --region=$Region --quiet --project=$ProjectId
if ($LASTEXITCODE -ne 0) { throw "Could not delete model $ModelId" }
Write-Host "Endpoint and model removed. Delete the demo Artifact Registry repository and Cloud Storage objects separately if no longer needed."

