# Deploy MLflow server to Azure Container Apps
# Run this AFTER CI builds & pushes the MLflow image to ACR
# Requires: az login (you're already logged in)
# Prerequisites: az containerapp env, Azure Files share

param(
    [Parameter(Mandatory = $false)]
    [string]$ResourceGroup = "mlops-rg",

    [Parameter(Mandatory = $false)]
    [string]$AcrName = "waterpotabilityacr",

    [Parameter(Mandatory = $false)]
    [string]$ContainerAppName = "mlflow-server",

    [Parameter(Mandatory = $false)]
    [string]$Environment = "water-potability-env",

    [Parameter(Mandatory = $false)]
    [string]$Location = "southeastasia"
)

$loginServer = "${AcrName}.azurecr.io"
$imageTag = "${loginServer}/mlflow-server:latest"

Write-Host "=== Deploying MLflow server ===" -ForegroundColor Cyan
Write-Host "Image: $imageTag"
Write-Host ""

# Create Azure Files share for persistence (skip if exists)
$shareName = "mlflow-data"
$storageAccount = az storage account list --resource-group $ResourceGroup --query "[0].name" -o tsv

Write-Host "Creating Azure Files share '$shareName'..."
az storage share create --name $shareName `
    --account-name $storageAccount `
    --quota 1 2>$null | Out-Null

$storageKey = az storage account keys list --account-name $storageAccount `
    --resource-group $ResourceGroup --query "[0].value" -o tsv

Write-Host "Creating or updating Container App '$ContainerAppName'..."
az containerapp create --name $ContainerAppName `
    --resource-group $ResourceGroup `
    --environment $Environment `
    --image $imageTag `
    --target-port 5001 `
    --ingress external `
    --min-replicas 1 `
    --max-replicas 1 `
    --secrets "storage-key=$storageKey" `
    --cpu 0.25 --memory 0.5Gi `
    --env-vars "AZURE_STORAGE_ACCOUNT=$storageAccount" `
    --mount-path "/mlflow" `
    --azure-file-volume "account-name=$storageAccount,account-key-ref=storage-key,share-name=$shareName,mount-path=/mlflow" 2>$null

if ($?) {
    Write-Host "Updating existing Container App..." -ForegroundColor Yellow
    az containerapp update --name $ContainerAppName `
        --resource-group $ResourceGroup `
        --image $imageTag | Out-Null
}

Start-Sleep -Seconds 10

$fqdn = az containerapp show --name $ContainerAppName `
    --resource-group $ResourceGroup `
    --query properties.configuration.ingress.fqdn -o tsv

Write-Host ""
Write-Host "=== MLflow server deployed! ===" -ForegroundColor Green
Write-Host "URL: https://${fqdn}"
Write-Host ""
Write-Host "Add this URL as a GitHub variable:" -ForegroundColor Cyan
Write-Host "  Name: MLFLOW_TRACKING_URI"
Write-Host "  Value: https://${fqdn}"
