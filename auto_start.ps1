# Auto-start script - monitors embeddings and launches app when ready
Write-Host "Monitoring embeddings generation..." -ForegroundColor Cyan

$maxWaitMinutes = 150  # 2.5 hours max wait
$startTime = Get-Date

while ($true) {
    Start-Sleep -Seconds 60  # Check every minute
    
    $elapsed = ((Get-Date) - $startTime).TotalMinutes
    if ($elapsed -gt $maxWaitMinutes) {
        Write-Host "Timeout reached. Please check manually." -ForegroundColor Red
        break
    }
    
    # Check if embeddings file exists
    if (Test-Path "new_data\embeddings\embeddings.json") {
        Write-Host "`n✅ Embeddings generation complete!" -ForegroundColor Green
        Write-Host "Starting Chainlit application..." -ForegroundColor Cyan
        
        # Start the Chainlit app
        Set-Location "chainlit-app"
        $env:PYTHONPATH = "E:\SEM_7\childcare-advanced-rag-main\src"
        E:/SEM_7/childcare-advanced-rag-main/.venv/Scripts/python.exe -m chainlit run app.py
        break
    }
    
    # Show progress
    $job = Get-Job -Name "EmbeddingsGen" -ErrorAction SilentlyContinue
    if ($job) {
        $output = Receive-Job -Name "EmbeddingsGen" -Keep 2>$null | Select-Object -Last 5
        Write-Host "`rStatus: Processing... (Elapsed: $([int]$elapsed) min)" -NoNewline
    }
}
