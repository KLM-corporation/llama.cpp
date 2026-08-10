# bench-patched-vs-stock.ps1 - Bench rapide PATCH vs STOCK sur 2080 Ti
# Usage: .\bench-patched-vs-stock.ps1 -Model "models\Bonsai-27B-Q1_0.gguf"
param([string]$Model = "models\mistral-7b-instruct-v0.2.Q4_K_M.gguf")

Write-Host "=== BENCH PATCH vs STOCK - RTX 2080 Ti ===" -ForegroundColor Cyan
Write-Host "Modèle: $Model" -ForegroundColor Yellow

# Vérifie modèle existe
if (!(Test-Path $Model)) { Write-Host "Fichier introuvable: $Model" -ForegroundColor Red; exit 1 }

# Fonction bench
function Run-Bench($BuildDir, $Label) {
  $exe = Join-Path $BuildDir "bin\Release\llama-bench.exe"
  if (!(Test-Path $exe)) { $exe = Join-Path $BuildDir "bin\Release\llama-bench.exe"; }
  # Alternative: llama-bench est dans build/bin/Release
  if (!(Test-Path $exe)) { $exe = "build\$BuildDir\llama-bench.exe" }
  Write-Host "`n--- $Label ($BuildDir) ---" -ForegroundColor Green
  $cmd = ".\$BuildDir\bin\Release\llama-bench.exe -m $Model -ngl 99 -p 512 -n 128 -t 6"
  Write-Host $cmd -ForegroundColor Gray
  # Si exe pas trouvé, essaie build-patched / build-stock
  if (Test-Path ".\$BuildDir\bin\Release\llama-bench.exe") {
    & ".\$BuildDir\bin\Release\llama-bench.exe" -m $Model -ngl 99 -p 512 -n 128 -t 6 2>&1 | Tee-Object -FilePath "bench-$Label.txt"
  } else {
    Write-Host "Binaire non trouvé: $BuildDir\bin\Release\llama-bench.exe" -ForegroundColor Red
  }
}

# 1. Build STOCK (actuel, sans patch cassé)
Write-Host "`n[1/2] Build STOCK (turing-2080ti-opti clean)..." -ForegroundColor Cyan
git checkout turing-2080ti-opti
cmake -B build-stock -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON 2>&1 | Out-Null
cmake --build build-stock --config Release -j 8 2>&1 | Out-Null
Run-Bench "build-stock" "STOCK"

# 2. Build PATCHED (ancien patch I=64 - va planter mais on bench quand même le pp)
Write-Host "`n[2/2] Build PATCHED (bench-patched)..." -ForegroundColor Cyan
git checkout bench-patched
cmake -B build-patched -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON 2>&1 | Out-Null
cmake --build build-patched --config Release -j 8 2>&1 | Out-Null
Run-Bench "build-patched" "PATCHED"

# 3. Compare
Write-Host "`n=== RESULTATS ===" -ForegroundColor Cyan
if (Test-Path "bench-STOCK.txt") { Write-Host "`n--- STOCK ---" -ForegroundColor Yellow; Get-Content bench-STOCK.txt | Select-String -Pattern "pp|tg|branch|commit" }
if (Test-Path "bench-PATCHED.txt") { Write-Host "`n--- PATCHED ---" -ForegroundColor Yellow; Get-Content bench-PATCHED.txt | Select-String -Pattern "pp|tg|branch|commit" }

Write-Host "`nFichiers: bench-STOCK.txt , bench-PATCHED.txt" -ForegroundColor Gray
Write-Host "Pour revenir au clean: git checkout turing-2080ti-opti" -ForegroundColor Gray
