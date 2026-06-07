# Build + sign a universal (arm64-v8a + x86_64) debug APK for yosh-android, with
# no Gradle: cargo-ndk produces the .so per ABI, then aapt2/zipalign/apksigner
# package + sign it. Pass -Run to also install + launch on a connected device.
#
# Prereqs (one-time): the Android NDK, SDK build-tools/platform-tools/platform,
# a JDK, the rustup android targets, and cargo-ndk. See README.md. Adjust the
# paths below to your install if they differ.
param([switch]$Run, [string]$Profile = "debug")

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# --- toolchain locations -----------------------------------------------------
$JDK = "$env:LOCALAPPDATA\Android\tools\jdk-17.0.19+10"
$SDK = "$env:LOCALAPPDATA\Android\Sdk"
$NDK = "$SDK\ndk\android-ndk-r27c"
$BT  = "$SDK\build-tools\34.0.0"
$JAR = "$SDK\platforms\android-34\android.jar"
$env:ANDROID_NDK_HOME = $NDK
$env:JAVA_HOME = $JDK

$proj = $PSScriptRoot
$root = Split-Path (Split-Path $proj)          # workspace root
$out  = "$proj\out"
$abis = @(@{abi='arm64-v8a'; ndk='arm64-v8a';  triple='aarch64-linux-android'},
          @{abi='x86_64';    ndk='x86_64';     triple='x86_64-linux-android'})

# --- 1. cross-compile the .so for each ABI -----------------------------------
$profileFlag = if ($Profile -eq "release") { "--release" } else { "" }
foreach ($a in $abis) {
    Write-Host "Building $($a.triple) ($Profile)..."
    & cargo ndk --target $a.ndk --platform 24 build --package yosh-android $profileFlag
    if ($LASTEXITCODE) { throw "cargo ndk failed for $($a.triple)" }
}

# --- 2. stage stripped libs --------------------------------------------------
Remove-Item $out -Recurse -Force -ErrorAction SilentlyContinue
$strip = "$NDK\toolchains\llvm\prebuilt\windows-x86_64\bin\llvm-strip.exe"
foreach ($a in $abis) {
    $d = "$out\stage\lib\$($a.abi)"
    New-Item -ItemType Directory -Force $d | Out-Null
    Copy-Item "$root\target\$($a.triple)\$Profile\libyosh_android.so" "$d\libyosh_android.so"
    & $strip "$d\libyosh_android.so"
}

# --- 3. package + sign -------------------------------------------------------
& "$BT\aapt2.exe" link -o "$out\base.apk" -I $JAR --manifest "$proj\AndroidManifest.xml" `
    --min-sdk-version 24 --target-sdk-version 34
& "$JDK\bin\jar.exe" uf "$out\base.apk" -C "$out\stage" lib
& "$BT\zipalign.exe" -f 4 "$out\base.apk" "$out\aligned.apk"
if (-not (Test-Path "$out\debug.keystore")) {
    & "$JDK\bin\keytool.exe" -genkeypair -keystore "$out\debug.keystore" -alias ad -keyalg RSA `
        -keysize 2048 -validity 10000 -storepass android -keypass android -dname "CN=Android Debug" | Out-Null
}
& "$BT\apksigner.bat" sign --ks "$out\debug.keystore" --ks-pass pass:android --key-pass pass:android `
    --out "$out\yosh.apk" "$out\aligned.apk"
Write-Host "APK: $out\yosh.apk ($([math]::Round((Get-Item "$out\yosh.apk").Length/1MB,1)) MB)"

# --- 4. optionally install + launch ------------------------------------------
if ($Run) {
    $adb = "$SDK\platform-tools\adb.exe"
    & $adb install -r "$out\yosh.apk"
    & $adb shell am start -n com.thedatabase.yosh/android.app.NativeActivity
}
