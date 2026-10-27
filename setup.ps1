# ============================================================================
#  TruthLens — one-time backend setup  (Windows PowerShell)
#
#  Run from the project root:
#      powershell -ExecutionPolicy Bypass -File .\setup.ps1
#
#  Installs every Python dependency, pre-fetches NLTK data, and warms the
#  HuggingFace model cache so the first analysis request doesn't time out.
# ============================================================================

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

function Step($n, $msg) { Write-Host "`n[$n] $msg" -ForegroundColor Cyan }
function Ok($msg)        { Write-Host "    OK  $msg" -ForegroundColor Green }
function Warn($msg)      { Write-Host "    !   $msg" -ForegroundColor Yellow }

Write-Host "======================================================" -ForegroundColor White
Write-Host " TruthLens backend setup" -ForegroundColor White
Write-Host "======================================================" -ForegroundColor White

# ---------------------------------------------------------------- 1. Python
Step 1 "Locating Python"
$py = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $py) { throw "Python not found on PATH. Install Python 3.10-3.12 and retry." }
$pyver = & python --version
Ok $pyver

# ------------------------------------------------------------------ 2. venv
Step 2 "Preparing virtual environment"
$venvPy = Join-Path $root "venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Warn "venv missing or broken — creating a fresh one"
    if (Test-Path (Join-Path $root "venv")) {
        Remove-Item -Recurse -Force (Join-Path $root "venv")
    }
    & python -m venv venv
}
Ok "venv ready at .\venv"

& $venvPy -m pip install --upgrade pip setuptools wheel --quiet
Ok "pip upgraded"

# ------------------------------------------------------------------ 3. torch
# Installed separately from the CPU index — the default PyPI wheel pulls
# ~2.5GB of CUDA libraries most laptops never use.
Step 3 "Installing PyTorch (CPU build)"
$torchInstalled = & $venvPy -c "import torch; print(torch.__version__)" 2>$null
if ($LASTEXITCODE -eq 0) {
    Ok "torch already present ($torchInstalled)"
} else {
    & $venvPy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    Ok "torch installed (CPU)"
}

# ------------------------------------------------------------- 4. everything else
Step 4 "Installing remaining requirements (this takes a few minutes)"
& $venvPy -m pip install -r requirements.txt
Ok "requirements installed"

# ---------------------------------------------------------------- 5. NLTK data
Step 5 "Pre-fetching NLTK tokenizer data"
& $venvPy -c @"
import nltk
for pkg in ('punkt', 'punkt_tab'):
    try:
        nltk.download(pkg, quiet=True)
    except Exception as e:
        print('  warn:', pkg, e)
print('nltk ready')
"@
Ok "punkt + punkt_tab cached"

# --------------------------------------------------------- 6. warm model cache
Step 6 "Warming HuggingFace model cache (downloads ~500MB once)"
& $venvPy -c @"
import warnings; warnings.filterwarnings('ignore')
try:
    from transformers import pipeline
    pipeline('ner', model='dslim/bert-base-NER', aggregation_strategy='simple')
    print('  NER model cached')
except Exception as e:
    print('  warn: NER cache failed ->', e)
try:
    from sentence_transformers import CrossEncoder
    CrossEncoder('cross-encoder/nli-deberta-v3-small')
    print('  NLI model cached')
except Exception as e:
    print('  warn: NLI cache failed ->', e)
try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    mid = 'prithivMLmods/Deep-Fake-Detector-v2-Model'
    AutoImageProcessor.from_pretrained(mid)
    AutoModelForImageClassification.from_pretrained(mid)
    print('  deepfake classifier cached (~340MB)')
except Exception as e:
    print('  warn: deepfake classifier cache failed ->', e)
"@
Ok "model cache warmed"

# ------------------------------------------------------------- 7. verify imports
Step 7 "Verifying the application imports cleanly"
& $venvPy -c "from src.api.main import app; print('  FastAPI app OK —', len(app.routes), 'routes')"
if ($LASTEXITCODE -ne 0) { throw "Application failed to import. Scroll up for the traceback." }
Ok "backend imports cleanly"

# ------------------------------------------------------------------ 8. Neo4j
Step 8 "Checking Neo4j connectivity"
& $venvPy -c @"
try:
    from src.graph_module.graph_query import query_entity_facts
    facts = query_entity_facts('Barack Obama')
    if facts:
        print(f'  Neo4j OK — {len(facts)} fact(s) for the test entity')
    else:
        print('  warn: connected but no facts returned. Run:  .\\venv\\Scripts\\python.exe seed_kg.py')
except Exception as e:
    print('  warn: Neo4j unreachable ->', e)
"@

# ------------------------------------------------------------------- 9. frontend
Step 9 "Installing frontend dependencies"
if (Get-Command npm -ErrorAction SilentlyContinue) {
    Push-Location (Join-Path $root "frontend-app")
    npm install
    Pop-Location
    Ok "npm packages installed"
} else {
    Warn "npm not found — install Node.js 18+ to run the React frontend"
}

Write-Host "`n======================================================" -ForegroundColor Green
Write-Host " Setup complete." -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green
Write-Host "`nStart everything with:  " -NoNewline
Write-Host ".\run.ps1" -ForegroundColor Cyan
Write-Host ""
