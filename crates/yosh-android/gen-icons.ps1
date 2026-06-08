# Regenerate the Android launcher icons under res/ from the desktop logo.
# Mirrors crates/yosh/assets/icons/generate.sh: a one-shot ImageMagick (v7,
# `magick`) step run by hand; the PNGs it emits are committed, and build-apk.ps1
# only *packages* them (it never calls this script). Re-run after the logo changes.
#
#   ./gen-icons.ps1
#
# Produces, per density:
#   ic_launcher_foreground.png  adaptive-icon foreground: the logo padded into the
#                               inner ~66% of a 108dp canvas, transparent elsewhere,
#                               so the launcher mask never clips the ears / hair /
#                               book corners. Background is a flat colour (colors.xml),
#                               so it needs no PNG.
#   ic_launcher.png             legacy full-tile fallback (API 24-25): the logo on the
#                               #672168 purple with a small margin.
$ErrorActionPreference = "Stop"

# Square 256x256 layer of the multi-res desktop .ico (yosh.png is 256x255, non-square).
$src = Join-Path $PSScriptRoot "..\yosh\assets\yosh.ico[0]"
$purple = "#672168"   # sampled from the book in the logo
$res = Join-Path $PSScriptRoot "res"

# density -> { canvas = adaptive fg size (108dp * scale), tile = legacy icon size }
$densities = @(
    @{ name = "mdpi";    canvas = 108; tile = 48  },
    @{ name = "hdpi";    canvas = 162; tile = 72  },
    @{ name = "xhdpi";   canvas = 216; tile = 96  },
    @{ name = "xxhdpi";  canvas = 324; tile = 144 },
    @{ name = "xxxhdpi"; canvas = 432; tile = 192 }
)

foreach ($d in $densities) {
    $dir = Join-Path $res "mipmap-$($d.name)"
    New-Item -ItemType Directory -Force $dir | Out-Null

    # Adaptive foreground: logo at ~66% of the canvas, centred, transparent border.
    $fg = [math]::Round($d.canvas * 0.66)
    & magick $src -resize "${fg}x${fg}" -background none -gravity center `
        -extent "$($d.canvas)x$($d.canvas)" (Join-Path $dir "ic_launcher_foreground.png")
    if ($LASTEXITCODE) { throw "magick foreground failed ($($d.name))" }

    # Legacy full tile: logo at ~86% on solid purple. -background must be set before
    # -extent (so the added border is purple, not the default white) and -alpha remove
    # then composites the monkey's own transparent halo onto the same purple.
    $lg = [math]::Round($d.tile * 0.86)
    & magick $src -resize "${lg}x${lg}" -background $purple -gravity center `
        -extent "$($d.tile)x$($d.tile)" -alpha remove -alpha off (Join-Path $dir "ic_launcher.png")
    if ($LASTEXITCODE) { throw "magick legacy failed ($($d.name))" }

    Write-Host "mipmap-$($d.name): foreground $($d.canvas)px, legacy $($d.tile)px"
}
Write-Host "done -> $res"
