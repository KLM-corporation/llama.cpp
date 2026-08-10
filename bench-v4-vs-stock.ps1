# bench-v4-vs-stock.ps1 - Bench V4 (vrai LUT) vs STOCK
param([string]$Model = "models\Bonsai-27B-Q1_0.gguf")
$ErrorActionPreference = "SilentlyContinue"
$PSNativeCommandUseErrorActionPreference = $false
Write-Host "=== BENCH V4 (vrai LUT) vs STOCK - RTX 2080 Ti ===" -ForegroundColor Cyan
if (!(Test-Path $Model)) { Write-Host "Fichier introuvable: $Model" -ForegroundColor Red; exit 1 }
function Bench($Branch, $BuildDir, $Label) {
  Write-Host "`n--- $Label ($Branch) ---" -ForegroundColor Green
  git checkout $Branch 2>&1 | Out-Null
  if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
  cmake -B $BuildDir -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON 2>&1 | Out-Null
  cmake --build $BuildDir --config Release -j 8 2>&1 | Out-Null
  $exe = ".\$BuildDir\bin\Release\llama-bench.exe"
  if (Test-Path $exe) { & $exe -m $Model -ngl 99 -p 512 -n 128 -t 6 *>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath "bench-$Label.txt" }
}
Bench "turing-2080ti-opti" "build-stock" "STOCK"
Bench "turing-v4" "build-v4" "V4"
Write-Host "`n=== COMPARAISON ===" -ForegroundColor Cyan
Get-Content bench-STOCK.txt | Select-String "pp|tg|model"
Get-Content bench-V4.txt | Select-String "pp|tg|model"
git checkout turing-2080ti-opti 2>&1 | Out-Null
