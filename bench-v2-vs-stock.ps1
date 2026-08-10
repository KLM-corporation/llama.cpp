# bench-v2-vs-stock.ps1 - Bench STOCK vs V2 SAFE sur 2080 Ti
param([string]$Model = "models\mistral-7b-instruct-v0.2.Q4_K_M.gguf")
Write-Host "=== BENCH V2 SAFE vs STOCK - RTX 2080 Ti ===" -ForegroundColor Cyan
Write-Host "Modèle: $Model" -ForegroundColor Yellow
if (!(Test-Path $Model)) { Write-Host "Fichier introuvable: $Model" -ForegroundColor Red; exit 1 }
function Bench($Branch, $BuildDir, $Label) {
  Write-Host "`n--- $Label ($Branch) ---" -ForegroundColor Green
  git checkout $Branch 2>&1 | Out-Null
  if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
  cmake -B $BuildDir -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON 2>&1 | Out-Null
  cmake --build $BuildDir --config Release -j 8 2>&1 | Out-Null
  $exe = ".\$BuildDir\bin\Release\llama-bench.exe"
  if (Test-Path $exe) {
    & $exe -m $Model -ngl 99 -p 512 -n 128 -t 6 2>&1 | Tee-Object -FilePath "bench-$Label.txt"
  } else { Write-Host "Binaire introuvable" -ForegroundColor Red }
}
Bench "turing-2080ti-opti" "build-stock" "STOCK"
Bench "turing-v2" "build-v2" "V2-SAFE"
Write-Host "`n=== COMPARAISON ===" -ForegroundColor Cyan
Get-Content bench-STOCK.txt | Select-String "pp|tg|model"
Get-Content bench-V2-SAFE.txt | Select-String "pp|tg|model"
Write-Host "`nFichiers: bench-STOCK.txt , bench-V2-SAFE.txt" -ForegroundColor Gray
git checkout turing-2080ti-opti 2>&1 | Out-Null
