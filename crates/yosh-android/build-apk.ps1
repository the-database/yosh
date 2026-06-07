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
foreach ($a in $abis) {
    Write-Host "Building $($a.triple) ($Profile)..."
    $cargoArgs = @('ndk', '--target', $a.ndk, '--platform', '24', 'build', '--package', 'yosh-android')
    if ($Profile -eq 'release') { $cargoArgs += '--release' }
    & cargo @cargoArgs
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

# --- 2b. compile + dex the Java SAF bridge -> stage/classes.dex --------------
$classes = "$out\classes"
New-Item -ItemType Directory -Force $classes | Out-Null
$javaFiles = @((Get-ChildItem -Recurse "$proj\java" -Filter *.java).FullName)
& "$JDK\bin\javac.exe" -nowarn -source 8 -target 8 -classpath $JAR -d $classes $javaFiles
if ($LASTEXITCODE) { throw "javac failed" }
$classFiles = @((Get-ChildItem -Recurse $classes -Filter *.class).FullName)
& "$BT\d8.bat" --min-api 24 --lib $JAR --output "$out\stage" $classFiles
if ($LASTEXITCODE) { throw "d8 failed" }

# --- 3. package + sign -------------------------------------------------------
& "$BT\aapt2.exe" link -o "$out\base.apk" -I $JAR --manifest "$proj\AndroidManifest.xml" `
    --min-sdk-version 24 --target-sdk-version 34
& "$JDK\bin\jar.exe" uf "$out\base.apk" -C "$out\stage" lib
& "$JDK\bin\jar.exe" uf "$out\base.apk" -C "$out\stage" classes.dex
& "$BT\zipalign.exe" -f 4 "$out\base.apk" "$out\aligned.apk"
if (-not (Test-Path "$proj\debug.keystore")) {
    & "$JDK\bin\keytool.exe" -genkeypair -keystore "$proj\debug.keystore" -alias ad -keyalg RSA `
        -keysize 2048 -validity 10000 -storepass android -keypass android -dname "CN=Android Debug" | Out-Null
}
& "$BT\apksigner.bat" sign --ks "$proj\debug.keystore" --ks-pass pass:android --key-pass pass:android `
    --out "$out\yosh.apk" "$out\aligned.apk"
Write-Host "APK: $out\yosh.apk ($([math]::Round((Get-Item "$out\yosh.apk").Length/1MB,1)) MB)"

# --- 4. optionally install + launch ------------------------------------------
if ($Run) {
    $adb = "$SDK\platform-tools\adb.exe"
    & $adb install -r "$out\yosh.apk"
    & $adb shell am start -n com.thedatabase.yosh/.YoshActivity
}
