<#
.SYNOPSIS
    Automated Deployment & Launch Script for DUHA-Net 100-Epoch AWS Training.

.DESCRIPTION
    Syncs the local DUHA-Net repository and Potsdam dataset to your EC2 instance,
    verifies the remote environment, and launches training inside a detached tmux session.

.EXAMPLE
    .\deploy_to_aws.ps1 -PublicIp "204.236.252.159" -KeyPath "..\CMCNet-main-btp\cmcnet-key-pair.pem"
#>

param (
    [Parameter(Mandatory = $true, HelpMessage = "Public IP address of the AWS EC2 instance")]
    [string]$PublicIp,

    [Parameter(Mandatory = $false, HelpMessage = "Path to SSH private key (.pem)")]
    [string]$KeyPath = "..\CMCNet-main-btp\cmcnet-key-pair.pem",

    [Parameter(Mandatory = $false, HelpMessage = "Whether to upload the local data/potsdam dataset directory")]
    [switch]$UploadData,

    [Parameter(Mandatory = $false, HelpMessage = "Resume from existing 20-epoch checkpoint")]
    [switch]$Resume
)

$ErrorActionPreference = "Stop"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "   DUHA-NET AWS EC2 100-EPOCH LAUNCHER" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "Target Host : $PublicIp"
Write-Host "SSH Key     : $KeyPath"

if (-not (Test-Path $KeyPath)) {
    Write-Error "Could not find SSH key at '$KeyPath'. Please provide the correct -KeyPath parameter."
    exit 1
}

$SshUser = "ubuntu"
$RemoteDir = "~/duhanet-HAAR-enhanced-segmentation"

Write-Host "`n[Step 1/4] Testing SSH Connectivity..." -ForegroundColor Yellow
$testConn = ssh -i $KeyPath -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$SshUser@$PublicIp" "echo 'SSH Connection OK'"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to connect to EC2 instance at $PublicIp. Please ensure the instance is running and port 22 is open in its Security Group."
    exit 1
}
Write-Host " Connection verified!" -ForegroundColor Green

Write-Host "`n[Step 2/4] Initializing Remote Project Directory..." -ForegroundColor Yellow
ssh -i $KeyPath "$SshUser@$PublicIp" "mkdir -p $RemoteDir/output $RemoteDir/visualization $RemoteDir/data"

Write-Host "`n[Step 3/4] Transferring Code & Scripts to EC2..." -ForegroundColor Yellow
# Transfer Python scripts, bash runners, and requirements
scp -i $KeyPath train.py test.py save_predictions.py verify_pipeline.py run_duhanet_100ep.sh setup_aws_env.sh requirements.txt "$SshUser@$PublicIp`:$RemoteDir/"
scp -i $KeyPath -r model utils visualization "$SshUser@$PublicIp`:$RemoteDir/"

if ($UploadData) {
    Write-Host "`n[Data Upload] Uploading data/potsdam to EC2 (this may take several minutes)..." -ForegroundColor Yellow
    scp -i $KeyPath -r data/potsdam "$SshUser@$PublicIp`:$RemoteDir/data/"
} else {
    Write-Host "`n[Note] Skipping dataset upload (-UploadData flag not provided). Ensure data/potsdam exists on EC2." -ForegroundColor Gray
}

# Make scripts executable on remote
ssh -i $KeyPath "$SshUser@$PublicIp" "chmod +x $RemoteDir/*.sh"

Write-Host "`n[Step 4/4] Launching 100-Epoch Training in Detached tmux Session..." -ForegroundColor Yellow
$ResumeFlag = if ($Resume) { "--resume ./output/DUHANet_potsdam_best.pth" } else { "" }

$LaunchCmd = @"
tmux kill-session -t duhanet 2>/dev/null || true
tmux new-session -d -s duhanet 'cd $RemoteDir && bash run_duhanet_100ep.sh $ResumeFlag'
echo 'tmux session [duhanet] successfully started!'
"@

ssh -i $KeyPath "$SshUser@$PublicIp" "$LaunchCmd"

Write-Host "`n=================================================================" -ForegroundColor Green
Write-Host " Training launched successfully in background tmux session!" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green
Write-Host "Useful commands to monitor training on EC2:"
Write-Host "  1. Attach to live training terminal : ssh -i `"$KeyPath`" $SshUser@$PublicIp -t `"tmux attach -t duhanet`""
Write-Host "  2. Tail logs in real-time          : ssh -i `"$KeyPath`" $SshUser@$PublicIp `"tail -f $RemoteDir/training_100ep.log`""
Write-Host "  3. Check GPU status                : ssh -i `"$KeyPath`" $SshUser@$PublicIp `"nvidia-smi`""
Write-Host "=================================================================" -ForegroundColor Green
