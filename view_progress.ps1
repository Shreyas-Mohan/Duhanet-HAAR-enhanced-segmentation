<#
.SYNOPSIS
    Clean Live Training Monitor for DUHA-Net on AWS EC2.
    Filters out noisy OpenCV GeoTIFF metadata warnings so you only see clean progress and epoch metrics.
#>

param (
    [string]$PublicIp = "184.73.45.73",
    [string]$KeyPath = ".\duhanet-aws.pem"
)

Write-Host "Connecting to EC2 ($PublicIp) to stream live training output..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C at any time to exit the viewer (training continues running on EC2)." -ForegroundColor Gray
Write-Host "=================================================================" -ForegroundColor DarkGray

ssh -i $KeyPath -o StrictHostKeyChecking=no ubuntu@$PublicIp "tail -n 25 -f ~/duhanet-HAAR-enhanced-segmentation/training_100ep.log | grep --line-buffered -v 'TIFF_Warning\|Unknown field\|grfmt_tiff'"
