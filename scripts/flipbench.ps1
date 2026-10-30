<#
.SYNOPSIS
  Compare UI/present frame timing (and engine throughput) between a base commit
  and the current working tree, using the `flipbench` feature build.

.DESCRIPTION
  Builds `yosh --features flipbench` for -Base (in a temporary git worktree) and
  for the working tree (uncommitted changes included), each into its own target
  dir under target/flipbench-* so the normal target/release binary is untouched.
  Then runs them alternately — a warm-up pass each, then -Rounds measured passes —
  on every -Book (flip through it at full speed) and, with -LibraryRoot, on the
  library grid. Each run is portable and starts from the same fixed settings
  (fit window, single page, no page-turn animation), in its own temp dir.

  Also runs `examples/seek_bench` (the engine alone: read, decode, HQ resize and
  GPU upload) on each book for both builds.

  -Base must already contain the flipbench feature.

.EXAMPLE
  scripts/flipbench.ps1 -Base v0.6.2 -Book "D:\manga\Vol 01.cbz" -LibraryRoot "D:\manga"
#>
param(
    [Parameter(Mandatory)] [string] $Base,
    [string[]] $Book = @(),
    [string] $LibraryRoot = "",
    [int] $Rounds = 2,
    [int] $TimeoutSec = 300
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path "$PSScriptRoot\..").Path
$work = Join-Path ([IO.Path]::GetTempPath()) "yosh-flipbench"
if (-not $Book -and -not $LibraryRoot) { throw "give at least one -Book or a -LibraryRoot" }

function Build([string] $src, [string] $name) {
    $env:CARGO_TARGET_DIR = Join-Path $repo "target\flipbench-$name"
    Push-Location $src
    try {
        Write-Host "building $name ..."
        cargo build --release -p yosh --features flipbench --quiet
        if ($LASTEXITCODE) { throw "build failed: $name" }
        cargo build --release -p yosh-engine --example seek_bench --quiet
        if ($LASTEXITCODE) { throw "seek_bench build failed: $name" }
    } finally {
        Pop-Location
        Remove-Item Env:CARGO_TARGET_DIR
    }
    $rel = Join-Path $repo "target\flipbench-$name\release"
    @{ yosh = "$rel\yosh.exe"; seek = "$rel\examples\seek_bench.exe" }
}

# Fixed, reproducible settings for every run (forward slashes: JSON-safe paths).
$settings = @{
    resume_on_startup = $false; help_seen = $true; theme = "dark"; fit = 0
    layout_spread = $false; scroll = $false; page_transition_enabled = $false
    seekbar_enabled = $true; direction_rtl = $true
}
if ($LibraryRoot) { $settings.library_root = $LibraryRoot.Replace('\', '/') }
$settingsJson = $settings | ConvertTo-Json -Compress

function RunOnce([string] $name, [string] $exe, [string] $book) {
    $dir = Join-Path $work "run-$name"
    New-Item -ItemType Directory -Force $dir | Out-Null
    Copy-Item $exe "$dir\yosh.exe" -Force
    # WriteAllText: BOM-less UTF-8 on every PowerShell (a BOM breaks the JSON parse).
    [IO.File]::WriteAllText("$dir\yosh-state.json", $settingsJson)
    $out = "$dir\report.txt"
    Remove-Item $out -ErrorAction SilentlyContinue
    $argv = @("--flipbench", "`"$out`"")
    if ($book) { $argv += "`"$book`"" }
    $p = Start-Process "$dir\yosh.exe" -ArgumentList $argv -PassThru
    if (-not $p.WaitForExit($TimeoutSec * 1000)) { $p.Kill(); throw "$name timed out" }
    if (-not (Test-Path $out)) { throw "$name wrote no report" }
    $r = @{}
    foreach ($line in Get-Content $out) {
        $k, $v = $line -split ' ', 2
        $r[$k] = $v
    }
    $r
}

# --- build both ---------------------------------------------------------------
New-Item -ItemType Directory -Force $work | Out-Null
$baseTree = Join-Path $work "base"
if (Test-Path $baseTree) { git -C $repo worktree remove --force $baseTree 2>$null }
git -C $repo worktree add --quiet --detach $baseTree $Base
if ($LASTEXITCODE) { throw "cannot check out $Base" }
try {
    if (-not (Select-String -Quiet -Pattern '^flipbench\s*=' "$baseTree\crates\yosh\Cargo.toml")) {
        throw "$Base predates the flipbench feature; pick a newer base"
    }
    $bins = @{ base = Build $baseTree "base"; current = Build $repo "current" }
} finally {
    git -C $repo worktree remove --force $baseTree
}

# --- run ----------------------------------------------------------------------
$cases = @($Book | ForEach-Object { @{ label = [IO.Path]::GetFileNameWithoutExtension($_); book = $_ } })
if ($LibraryRoot) { $cases += @{ label = "library"; book = "" } }
$results = @{}
foreach ($c in $cases) {
    Write-Host "warm-up: $($c.label)"
    foreach ($v in "base", "current") { RunOnce $v $bins[$v].yosh $c.book | Out-Null }
    foreach ($round in 1..$Rounds) {
        foreach ($v in "base", "current") {
            Write-Host "  round $round $v"
            $results["$($c.label)|$v|$round"] = RunOnce $v $bins[$v].yosh $c.book
        }
    }
}

# --- report -------------------------------------------------------------------
# Each cell lists the per-round values. Stats are `median p95 p99 max` in ms.
$metrics = @(
    @{ key = "pages_per_sec"; col = 0; label = "pages/sec shown" },
    @{ key = "ui_ms"; col = 0; label = "ui (egui) med ms" },
    @{ key = "ui_ms"; col = 1; label = "ui (egui) p95 ms" },
    @{ key = "post_ms"; col = 0; label = "post (submit/present) med" },
    @{ key = "pre_ms"; col = 0; label = "pre (reader/quads) med" },
    @{ key = "acquire_ms"; col = 0; label = "acquire (vsync wait) med" },
    @{ key = "total_ms"; col = 2; label = "frame total p99 ms" },
    @{ key = "interval_ms"; col = 0; label = "frame interval med ms" },
    @{ key = "interval_ms"; col = 2; label = "frame interval p99 ms" }
)
foreach ($c in $cases) {
    Write-Host ""
    Write-Host "== $($c.label)"
    "{0,-28} {1,-22} {2,-22}" -f "", "base ($Base)", "current" | Write-Host
    foreach ($m in $metrics) {
        $cells = foreach ($v in "base", "current") {
            $vals = foreach ($round in 1..$Rounds) {
                $r = $results["$($c.label)|$v|$round"]
                if ($r.ContainsKey($m.key)) { ($r[$m.key] -split ' ')[$m.col] } else { "-" }
            }
            $vals -join " / "
        }
        if ($cells[0] -notmatch '^[- /]*$') {
            "{0,-28} {1,-22} {2,-22}" -f $m.label, $cells[0], $cells[1] | Write-Host
        }
    }
}

# Engine alone, for the same books.
foreach ($c in $cases | Where-Object { $_.book }) {
    Write-Host ""
    Write-Host "== engine (seek_bench): $($c.label)"
    foreach ($v in "base", "current") {
        $o = & $bins[$v].seek $c.book 1440 5 | Select-Object -Skip 1
        Write-Host ("{0,-8} {1} | {2}" -f $v, $o[0], $o[1])
    }
}
