# bench-v3-vs-stock.ps1 - Bench V3 LUT vs STOCK
param([string]$Model = "models\mistral-7b-instruct-v0.2.Q4_K_M.gguf")
$ErrorActionPreference = "SilentlyContinue"
$PSNativeCommandUseErrorActionPreference = $false
Write-Host "=== BENCH V3 LUT vs STOCK - RTX 2080 Ti ===" -ForegroundColor Cyan
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
Bench "turing-v3-lut" "build-v3" "V3-LUT"
Write-Host "`n=== COMPARAISON ===" -ForegroundColor Cyan
Get-Content bench-STOCK.txt | Select-String "pp|tg|model"
Get-Content bench-V3-LUT.txt | Select-String "pp|tg|model"
git checkout turing-2080ti-opti 2>&1 | Out-Null
