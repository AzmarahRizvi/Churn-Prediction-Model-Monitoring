param(
    [Parameter(Mandatory = $true)][string]$ProjectId,
    [string]$Region = "us-central1",
    [Parameter(Mandatory = $true)][string]$EndpointId,
    [Parameter(Mandatory = $true)][string]$AlertEmail,
    [string]$JobDisplayName = "telco-churn-input-drift",
    [double]$SamplingRate = 1.0,
    [double]$Threshold = 0.1
)

$ErrorActionPreference = "Stop"
if ($SamplingRate -le 0 -or $SamplingRate -gt 1) { throw "SamplingRate must be in (0, 1]." }
if ($Threshold -le 0 -or $Threshold -gt 1) { throw "Threshold must be in (0, 1]." }

$culture = [System.Globalization.CultureInfo]::InvariantCulture
$thresholdText = $Threshold.ToString($culture)
$samplingText = $SamplingRate.ToString($culture)
$thresholds = "tenure=$thresholdText,MonthlyCharges=$thresholdText,TotalCharges=$thresholdText,Contract=$thresholdText,InternetService=$thresholdText,PaymentMethod=$thresholdText"
$sampleRequest = Join-Path $PSScriptRoot "..\examples\predict_request.json"

$monitoringResource = gcloud ai model-monitoring-jobs create `
    --project=$ProjectId `
    --region=$Region `
    --display-name=$JobDisplayName `
    --emails=$AlertEmail `
    --endpoint=$EndpointId `
    --prediction-sampling-rate=$samplingText `
    --monitoring-frequency=1 `
    --feature-thresholds=$thresholds `
    --sample-predict-request=$sampleRequest `
    --anomaly-cloud-logging `
    --format="value(name)"
if ($LASTEXITCODE -ne 0) { throw "Vertex AI Model Monitoring job creation failed" }
$monitoringId = ($monitoringResource.Trim() -split "/")[-1]

Write-Host "Created an hourly Vertex AI input-drift monitoring job for endpoint $EndpointId."
Write-Host "Monitoring job ID: $monitoringId"
Write-Host "Alerts will be sent to $AlertEmail and anomalies will be written to Cloud Logging."

