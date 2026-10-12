param(
    [Parameter(Mandatory = $true)][string]$ProjectId,
    [string]$Region = "us-central1",
    [Parameter(Mandatory = $true)][string]$BucketName,
    [string]$Repository = "churn-models",
    [string]$ImageName = "churn-predictor",
    [string]$ModelDisplayName = "telco-churn-xgboost",
    [string]$EndpointDisplayName = "telco-churn-endpoint"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Artifacts = Join-Path $RepoRoot "artifacts"
$ModelArtifact = Join-Path $Artifacts "model\model.joblib"

if (-not (Test-Path $ModelArtifact)) {
    throw "Model artifact not found at $ModelArtifact. Run python -m churn_monitoring.train first."
}

$ImageUri = "$Region-docker.pkg.dev/$ProjectId/$Repository/$ImageName`:latest"
$ModelArtifactUri = "gs://$BucketName/churn/model"

gcloud config set project $ProjectId
if ($LASTEXITCODE -ne 0) { throw "Could not select Google Cloud project $ProjectId" }
gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com storage.googleapis.com --project $ProjectId
if ($LASTEXITCODE -ne 0) { throw "Could not enable required Google Cloud APIs" }

$repoExists = $true
gcloud artifacts repositories describe $Repository --location $Region --project $ProjectId *> $null
if ($LASTEXITCODE -ne 0) { $repoExists = $false }
if (-not $repoExists) {
    gcloud artifacts repositories create $Repository --repository-format=docker --location=$Region --project=$ProjectId
    if ($LASTEXITCODE -ne 0) { throw "Could not create Artifact Registry repository $Repository" }
}

$bucketExists = $true
gcloud storage buckets describe "gs://$BucketName" *> $null
if ($LASTEXITCODE -ne 0) { $bucketExists = $false }
if (-not $bucketExists) {
    gcloud storage buckets create "gs://$BucketName" --location=$Region --uniform-bucket-level-access
    if ($LASTEXITCODE -ne 0) { throw "Could not create Cloud Storage bucket $BucketName" }
}

gcloud auth configure-docker "$Region-docker.pkg.dev" --quiet
if ($LASTEXITCODE -ne 0) { throw "Could not configure Docker authentication for Artifact Registry" }
docker build --tag $ImageUri $RepoRoot
if ($LASTEXITCODE -ne 0) { throw "Docker image build failed" }
docker push $ImageUri
if ($LASTEXITCODE -ne 0) { throw "Docker image push failed" }
gcloud storage cp $ModelArtifact "$ModelArtifactUri/model.joblib"
if ($LASTEXITCODE -ne 0) { throw "Could not upload model artifact to Cloud Storage" }

$modelResource = gcloud ai models upload `
    --region=$Region `
    --display-name=$ModelDisplayName `
    --artifact-uri=$ModelArtifactUri `
    --container-image-uri=$ImageUri `
    --container-predict-route=/predict `
    --container-health-route=/health `
    --container-ports=8080 `
    --project=$ProjectId `
    --format="value(name)"
if ($LASTEXITCODE -ne 0) { throw "Vertex AI model upload failed" }
$modelId = ($modelResource.Trim() -split "/")[-1]

$endpointResource = gcloud ai endpoints create `
    --region=$Region `
    --display-name=$EndpointDisplayName `
    --project=$ProjectId `
    --format="value(name)"
if ($LASTEXITCODE -ne 0) { throw "Vertex AI endpoint creation failed" }
$endpointId = ($endpointResource.Trim() -split "/")[-1]

gcloud ai endpoints deploy-model $endpointId `
    --region=$Region `
    --model=$modelId `
    --display-name=$ModelDisplayName `
    --machine-type=e2-standard-2 `
    --min-replica-count=1 `
    --max-replica-count=1 `
    --project=$ProjectId
if ($LASTEXITCODE -ne 0) { throw "Vertex AI model deployment failed" }

Write-Host "Model deployed."
Write-Host "Project:  $ProjectId"
Write-Host "Region:   $Region"
Write-Host "Endpoint: $endpointId"
Write-Host "Model:    $modelId"
Write-Host "Image:    $ImageUri"
Write-Host "Next:     .\scripts\configure_monitoring.ps1 -ProjectId $ProjectId -Region $Region -EndpointId $endpointId -AlertEmail you@example.com"

