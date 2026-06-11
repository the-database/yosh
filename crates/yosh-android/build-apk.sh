#!/usr/bin/env bash
# Linux counterpart of build-apk.ps1: build + sign a universal (arm64-v8a +
# x86_64) APK for yosh-android with no Gradle. cargo-ndk produces the .so per ABI,
# then aapt2/zipalign/apksigner package + sign it. This is the path CI uses
# (ubuntu-latest) — and the only host where RAR/CBR cross-compiles, so the `rar`
# feature is enabled here (with -DUNIX_TIME_NS so UnRAR uses utimensat, not the
# Bionic-absent lutimes). Also usable locally on Linux/WSL.
#
# Driven entirely by env vars (so CI can pass a release keystore + version, and a
# local run can just call it bare):
#   PROFILE        release | debug            (default: release)
#   FEATURES       cargo features for the .so (default: rar)
#   VERSION_NAME   APK versionName            (default: 0.0.0)
#   VERSION_CODE   APK versionCode (integer)  (default: 1)
#   KEYSTORE       signing keystore path. If unset, a throwaway debug keystore is
#                  generated (local convenience) — unsuitable for distribution.
#   KEYSTORE_PASS / KEY_PASS / KEY_ALIAS   keystore credentials (when KEYSTORE set)
# Toolchain is auto-detected from the Android SDK env the runner provides
# (ANDROID_SDK_ROOT/ANDROID_HOME, ANDROID_NDK_LATEST_HOME, JAVA_HOME); override by
# exporting those if your local layout differs.
set -euo pipefail

PROFILE="${PROFILE:-release}"
FEATURES="${FEATURES:-rar}"
VERSION_NAME="${VERSION_NAME:-0.0.0}"
VERSION_CODE="${VERSION_CODE:-1}"

proj="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$proj/../.." && pwd)"
out="$proj/out"

# --- toolchain locations -----------------------------------------------------
SDK="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"
[ -n "$SDK" ] || { echo "ANDROID_SDK_ROOT or ANDROID_HOME must be set"; exit 1; }
JDK="${JAVA_HOME:?JAVA_HOME must be set}"

# Newest installed build-tools (d8/aapt2/zipalign/apksigner live here).
BT="$(ls -d "$SDK"/build-tools/*/ 2>/dev/null | sort -V | tail -1)"; BT="${BT%/}"
[ -n "$BT" ] || { echo "no build-tools found under $SDK/build-tools"; exit 1; }

# NDK: prefer the runner's pinned latest, else newest under $SDK/ndk.
NDK="${ANDROID_NDK_LATEST_HOME:-${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}}"
if [ -z "$NDK" ]; then NDK="$(ls -d "$SDK"/ndk/*/ 2>/dev/null | sort -V | tail -1)"; NDK="${NDK%/}"; fi
[ -n "$NDK" ] || { echo "no NDK found (set ANDROID_NDK_LATEST_HOME or install one under $SDK/ndk)"; exit 1; }
export ANDROID_NDK_HOME="$NDK"

# Platform android.jar to compile/link against: prefer android-34 (our targetSdk),
# else the newest available.
if [ -f "$SDK/platforms/android-34/android.jar" ]; then
  JAR="$SDK/platforms/android-34/android.jar"
else
  JAR="$(ls "$SDK"/platforms/android-*/android.jar 2>/dev/null | sort -V | tail -1)"
fi
[ -n "$JAR" ] || { echo "no platform android.jar found under $SDK/platforms"; exit 1; }

STRIP="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip"

echo "SDK=$SDK"
echo "NDK=$NDK"
echo "build-tools=$BT"
echo "platform jar=$JAR"
echo "profile=$PROFILE  features=$FEATURES  versionName=$VERSION_NAME  versionCode=$VERSION_CODE"

abis=("arm64-v8a:aarch64-linux-android" "x86_64:x86_64-linux-android")

# RAR's UnRAR C++ needs UNIX_TIME_NS so it uses utimensat (Bionic has no lutimes).
# Harmless when the rar feature is off. Applies to whichever ABIs we build.
export CXXFLAGS_aarch64_linux_android="${CXXFLAGS_aarch64_linux_android:-} -DUNIX_TIME_NS"
export CXXFLAGS_x86_64_linux_android="${CXXFLAGS_x86_64_linux_android:-} -DUNIX_TIME_NS"

# --- 1. cross-compile the .so for each ABI -----------------------------------
relflag=()
[ "$PROFILE" = "release" ] && relflag=(--release)
featflag=()
[ -n "$FEATURES" ] && featflag=(--features "$FEATURES")
for entry in "${abis[@]}"; do
  ndk_abi="${entry%%:*}"
  triple="${entry##*:}"
  echo "Building $triple ($PROFILE)..."
  ( cd "$root" && cargo ndk --target "$ndk_abi" --platform 24 build --package yosh-android "${relflag[@]}" "${featflag[@]}" )
done

# --- 2. stage stripped libs --------------------------------------------------
rm -rf "$out"
for entry in "${abis[@]}"; do
  ndk_abi="${entry%%:*}"
  triple="${entry##*:}"
  d="$out/stage/lib/$ndk_abi"
  mkdir -p "$d"
  cp "$root/target/$triple/$PROFILE/libyosh_android.so" "$d/libyosh_android.so"
  "$STRIP" "$d/libyosh_android.so"
done

# --- 2b. compile + dex the Java SAF bridge -> stage/classes.dex --------------
classes="$out/classes"
mkdir -p "$classes"
mapfile -t javaFiles < <(find "$proj/java" -name '*.java')
"$JDK/bin/javac" -encoding UTF-8 -nowarn -source 8 -target 8 -classpath "$JAR" -d "$classes" "${javaFiles[@]}"
mapfile -t classFiles < <(find "$classes" -name '*.class')
"$BT/d8" --min-api 24 --lib "$JAR" --output "$out/stage" "${classFiles[@]}"

# --- 3. package + sign -------------------------------------------------------
# aapt2 can't link a raw res dir: compile it to a .flat zip first, then link that.
# --version-code/--version-name inject the version the manifest omits, so released
# APKs are update-installable (Android rejects an update with a non-increasing
# versionCode).
"$BT/aapt2" compile --dir "$proj/res" -o "$out/res.zip"
"$BT/aapt2" link -o "$out/base.apk" -I "$JAR" --manifest "$proj/AndroidManifest.xml" \
  "$out/res.zip" --min-sdk-version 24 --target-sdk-version 34 \
  --version-code "$VERSION_CODE" --version-name "$VERSION_NAME"
"$JDK/bin/jar" uf "$out/base.apk" -C "$out/stage" lib
"$JDK/bin/jar" uf "$out/base.apk" -C "$out/stage" classes.dex
"$BT/zipalign" -f 4 "$out/base.apk" "$out/aligned.apk"

# Signing: a real keystore from env, else a throwaway debug key (local only).
if [ -z "${KEYSTORE:-}" ]; then
  echo "No KEYSTORE set — generating a throwaway debug keystore (not for distribution)."
  KEYSTORE="$proj/debug.keystore"; KEYSTORE_PASS="android"; KEY_PASS="android"; KEY_ALIAS="ad"
  if [ ! -f "$KEYSTORE" ]; then
    "$JDK/bin/keytool" -genkeypair -keystore "$KEYSTORE" -alias "$KEY_ALIAS" -keyalg RSA \
      -keysize 2048 -validity 10000 -storepass "$KEYSTORE_PASS" -keypass "$KEY_PASS" -dname "CN=Android Debug"
  fi
fi
"$BT/apksigner" sign --ks "$KEYSTORE" --ks-pass "pass:${KEYSTORE_PASS}" --key-pass "pass:${KEY_PASS}" \
  --ks-key-alias "${KEY_ALIAS}" --out "$out/yosh.apk" "$out/aligned.apk"

size=$(awk "BEGIN{printf \"%.1f\", $(stat -c%s "$out/yosh.apk")/1048576}")
echo "APK: $out/yosh.apk (${size} MB)"
