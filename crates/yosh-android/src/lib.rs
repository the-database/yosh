//! Android shell for yosh: the platform glue under the shared mobile shell
//! ([`yosh_mobile`]).
//!
//! `android_main` builds a [`Shell`] over [`AndroidPlatform`] and runs it on
//! winit's Android event loop. `AndroidPlatform` is the shell's view of the OS: JNI
//! into the `YoshActivity` Java bridge (document picker, all-files access, window
//! flags, theme, locale, volume), the app's private data dir, and the kernel's
//! `/proc` + sysfs files for the device tier.
//!
//! The whole crate is Android-only; the desktop shell is `crates/yosh`.
#![cfg(target_os = "android")]

use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use jni::objects::{JObject, JString, JValue};
use jni::{jni_sig, jni_str, JavaVM};
use winit::event_loop::EventLoop;
use winit::keyboard::KeyCode;
use winit::platform::android::activity::AndroidApp;
use winit::platform::android::EventLoopBuilderExtAndroid;

use yosh_engine::source::{PageSource, ZipSource};
use yosh_mobile::{PickedJob, Platform, Shell};

/// Entry point android-activity calls on the native-activity thread.
#[unsafe(no_mangle)]
fn android_main(app: AndroidApp) {
    android_logger::init_once(
        android_logger::Config::default().with_max_level(log::LevelFilter::Info),
    );
    log::info!("yosh-android starting");
    // winit allows exactly one EventLoop per process. android-activity re-invokes
    // android_main when the activity is recreated in a still-cached process (e.g.
    // Back destroys the activity, then the user reopens) — building a second
    // EventLoop there panics. Defensively bail on re-entry so Android relaunches us
    // fresh instead. (The process::exit at the end normally prevents re-entry from
    // happening at all; this guards the timing edge where a reopen races the exit.)
    static STARTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if STARTED.swap(true, std::sync::atomic::Ordering::SeqCst) {
        log::warn!("android_main re-entered in a live process; exiting for a clean restart");
        std::process::exit(0);
    }
    let platform: Rc<dyn Platform> = Rc::new(AndroidPlatform { app: app.clone() });
    let mut shell = Shell::new(platform);
    // Volume keys are delivered to us (winit's default hands them straight to the
    // OS): with the "Volume keys turn pages" option on they flip while reading.
    // Every other press is passed back to the system via `adjust_volume`.
    let event_loop = EventLoop::builder()
        .with_android_app(app)
        .handle_volume_keys()
        .build()
        .expect("build event loop");
    if let Err(e) = event_loop.run_app(&mut shell) {
        log::error!("event loop exited with error: {e}");
    }
    // run_app returns only when the activity is destroyed (not on background — that
    // delivers Suspended and keeps running). Exit the process so the next launch is a
    // fresh process with a fresh EventLoop, instead of leaving a cached process that
    // would re-enter android_main and hit the one-EventLoop-per-process limit.
    log::info!("yosh-android event loop ended; exiting process");
    std::process::exit(0);
}

// --- AndroidPlatform ---------------------------------------------------------

/// [`Platform`] over the `YoshActivity` Java bridge and the kernel's `/proc` and
/// sysfs files.
struct AndroidPlatform {
    /// The vm + activity pointers JNI calls need.
    app: AndroidApp,
}

impl Platform for AndroidPlatform {
    fn data_dir(&self) -> Option<PathBuf> {
        self.app.internal_data_path()
    }
    fn default_library_root(&self) -> PathBuf {
        PathBuf::from("/storage/emulated/0")
    }
    fn has_library_access(&self) -> bool {
        has_all_files(&self.app)
    }
    fn request_library_access(&self) {
        request_all_files(&self.app)
    }
    fn launch_picker(&self) {
        launch_picker(&self.app)
    }
    fn take_picked(&self) -> Option<String> {
        take_picked_uri(&self.app)
    }
    /// The JNI call that hands us the descriptor stays on this thread (it wants
    /// the activity), but the descriptor is just an integer, so the job owns it.
    fn open_picked(&self, uri: &str) -> Option<PickedJob> {
        let fd = open_fd(&self.app, uri);
        if fd < 0 {
            log::warn!("openFd failed for {uri}");
            return None;
        }
        Some(Box::new(move || {
            // We own the fd (Java detachFd'd it); reading to end + drop closes it.
            let mut file = unsafe { <std::fs::File as std::os::fd::FromRawFd>::from_raw_fd(fd) };
            let mut bytes = Vec::new();
            std::io::Read::read_to_end(&mut file, &mut bytes)
                .map_err(|e| format!("read picked fd: {e}"))?;
            drop(file);
            let src = ZipSource::from_bytes(bytes).map_err(|e| format!("from_bytes: {e}"))?;
            Ok(Arc::new(src) as Arc<dyn PageSource>)
        }))
    }
    fn system_dark(&self) -> bool {
        system_dark(&self.app)
    }
    fn locale_tag(&self) -> String {
        locale_tag(&self.app)
    }
    fn set_keep_screen_on(&self, on: bool) {
        set_keep_screen_on(&self.app, on)
    }
    fn set_immersive(&self, on: bool) {
        set_immersive(&self.app, on)
    }
    fn top_inset_px(&self) -> i32 {
        status_bar_height(&self.app)
    }
    fn mem_mb(&self) -> (u64, u64) {
        let info = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        let total = meminfo_mb(&info, "MemTotal:").unwrap_or(4096);
        // Older kernels lack MemAvailable; fall back to the total-only estimate.
        let avail = meminfo_mb(&info, "MemAvailable:").unwrap_or(total / 2);
        (total, avail)
    }
    fn max_cpu_khz(&self) -> Option<u64> {
        max_cpu_khz()
    }
    fn cjk_font_bytes(&self) -> Option<Vec<u8>> {
        std::fs::read("/system/fonts/NotoSansCJK-Regular.ttc").ok()
    }
    fn unused_key(&self, code: KeyCode) {
        if let Some(adj) = volume_adjust(code) {
            adjust_volume(&self.app, adj);
        }
    }
}

// --- JNI bridge to YoshActivity ---------------------------------------------

/// Launch the SAF document picker via `YoshActivity.openDocument()`.
fn launch_picker(app: &AndroidApp) {
    match with_env(app, |env, activity| {
        env.call_method(activity, jni_str!("openDocument"), jni_sig!("()V"), &[])?;
        Ok(())
    }) {
        Ok(()) => log::info!("launched document picker"),
        Err(e) => log::error!("launch picker failed: {e}"),
    }
}

/// True if the app holds all-files access (can browse the library by path).
fn has_all_files(app: &AndroidApp) -> bool {
    with_env(app, |env, activity| {
        env.call_method(activity, jni_str!("hasAllFiles"), jni_sig!("()Z"), &[])?.z()
    })
    .unwrap_or(false)
}

/// True if the OS is in night (dark) mode — resolves the `System` theme. winit
/// doesn't surface the system theme on Android, so read the uiMode night flag via
/// JNI (available since API 29). Defaults to dark on failure (the historical look).
fn system_dark(app: &AndroidApp) -> bool {
    with_env(app, |env, activity| {
        env.call_method(activity, jni_str!("isSystemDark"), jni_sig!("()Z"), &[])?.z()
    })
    .unwrap_or(true)
}

/// Open Settings so the user can grant all-files access.
fn request_all_files(app: &AndroidApp) {
    let _ = with_env(app, |env, activity| {
        env.call_method(activity, jni_str!("requestAllFiles"), jni_sig!("()V"), &[])?;
        Ok(())
    });
}

/// Take the picked content:// URI (string) if one has arrived, else None.
fn take_picked_uri(app: &AndroidApp) -> Option<String> {
    with_env(app, |env, activity| {
        let res = env.call_method(activity, jni_str!("takePickedUri"), jni_sig!("()Ljava/lang/String;"), &[])?;
        let obj = res.l()?;
        if obj.is_null() {
            return Ok(None);
        }
        Ok(Some(env.cast_local::<JString>(obj)?.try_to_string(env)?))
    })
    .ok()
    .flatten()
}

/// Open a content:// URI to an owned file descriptor (-1 on failure).
fn open_fd(app: &AndroidApp, uri: &str) -> i32 {
    with_env(app, |env, activity| {
        let juri = env.new_string(uri)?;
        let res = env.call_method(
            activity,
            jni_str!("openFd"),
            jni_sig!("(Ljava/lang/String;)I"),
            &[JValue::Object(&juri)],
        )?;
        res.i()
    })
    .unwrap_or(-1)
}

/// Hold (true) / release (false) the KEEP_SCREEN_ON window flag, via the
/// `YoshActivity` JNI bridge. Deliberately NOT `AndroidApp::set_window_flags`:
/// that wrapper holds android-activity's glue mutex across
/// `ANativeActivity_setWindowFlags`, and calling it from the event-loop thread
/// mid-dispatch deadlocks against the UI thread's lifecycle callbacks on some
/// ROMs (froze the whole app on ZUI / Android 16 — the Java main thread futex-
/// waited in `NativeActivity.onPause` while our loop never returned from the
/// flags call). The bridge just `runOnUiThread`s the flag change, fire-and-forget.
fn set_keep_screen_on(app: &AndroidApp, on: bool) {
    let _ = with_env(app, |env, activity| {
        env.call_method(activity, jni_str!("setKeepScreenOn"), jni_sig!("(Z)V"), &[JValue::Bool(on)])?;
        Ok(())
    });
}

/// `AudioManager.ADJUST_*` for a volume key: raise, lower, toggle mute.
fn volume_adjust(code: KeyCode) -> Option<i32> {
    match code {
        KeyCode::AudioVolumeUp => Some(1),
        KeyCode::AudioVolumeDown => Some(-1),
        KeyCode::AudioVolumeMute => Some(101),
        _ => None,
    }
}

/// Do what the system would have done with a volume key we received but didn't
/// use for paging (`handle_volume_keys` routes every press to us): adjust the
/// active stream and show the volume panel.
fn adjust_volume(app: &AndroidApp, adj: i32) {
    let _ = with_env(app, |env, activity| {
        env.call_method(activity, jni_str!("adjustVolume"), jni_sig!("(I)V"), &[JValue::Int(adj)])?;
        Ok(())
    });
}

/// The UI locale's BCP-47 tag (per-app language if set, else the system's) — ""
/// if the bridge fails, which resolves to English.
fn locale_tag(app: &AndroidApp) -> String {
    with_env(app, |env, activity| {
        let s = env.call_method(activity, jni_str!("localeTag"), jni_sig!("()Ljava/lang/String;"), &[])?.l()?;
        env.cast_local::<JString>(s)?.try_to_string(env)
    })
    .unwrap_or_default()
}

/// Hide (true) / show (false) the Android status + navigation bars. The window
/// stays edge-to-edge; chrome is inset by `status_bar_height` while bars show.
fn set_immersive(app: &AndroidApp, immersive: bool) {
    let _ = with_env(app, |env, activity| {
        env.call_method(
            activity,
            jni_str!("setImmersive"),
            jni_sig!("(Z)V"),
            &[JValue::Bool(immersive)],
        )?;
        Ok(())
    });
}

/// Status-bar height in px (0 if unknown) — a platform constant used to inset the
/// library / empty-state chrome so the bars don't cover it.
fn status_bar_height(app: &AndroidApp) -> i32 {
    with_env(app, |env, activity| {
        env.call_method(activity, jni_str!("statusBarHeight"), jni_sig!("()I"), &[])?.i()
    })
    .unwrap_or(0)
}

/// Run a closure with an attached JNI `Env` and the `YoshActivity` instance.
///
/// jni 0.22's `attach_current_thread` attaches permanently (so every call after the
/// first is cheap), pushes a fresh local-reference frame around the closure, and
/// catches any Java exception the closure leaves pending as
/// `Error::CaughtJavaException` — the clear-after the 0.21 version did by hand.
fn with_env<T>(
    app: &AndroidApp,
    f: impl FnOnce(&mut jni::Env, &JObject) -> jni::errors::Result<T>,
) -> jni::errors::Result<T> {
    // SAFETY: android-activity hands us the process's live JavaVM pointer.
    let vm = unsafe { JavaVM::from_raw(app.vm_as_ptr().cast()) };
    vm.attach_current_thread(|env| {
        // SAFETY: the activity pointer is a global ref owned by the glue; `JObject`
        // has no `Drop`, so wrapping it here never deletes it.
        let activity = unsafe { JObject::from_raw(env, app.activity_as_ptr().cast()) };
        f(env, &activity)
    })
}

/// Read one `/proc/meminfo` field in MB (the values are in kB).
fn meminfo_mb(text: &str, field: &str) -> Option<u64> {
    text.lines()
        .find_map(|l| l.strip_prefix(field))?
        .split_whitespace()
        .next()?
        .parse::<u64>()
        .ok()
        .map(|kb| kb / 1024)
}

/// Highest CPU clock (kHz) the SoC advertises, across all cores. This is the
/// big.LITTLE discriminator the tier heuristic needs: all-little budget SoCs top
/// out around 1.8–2.0 GHz while a flagship prime core is 2.8 GHz+, and core *count*
/// can't tell them apart (both are 8). `None` when sysfs isn't readable (some
/// devices restrict it) — the caller then tiers on RAM alone.
fn max_cpu_khz() -> Option<u64> {
    let dir = std::fs::read_dir("/sys/devices/system/cpu").ok()?;
    let mut best: Option<u64> = None;
    for e in dir.flatten() {
        let p = e.path().join("cpufreq/cpuinfo_max_freq");
        if let Ok(s) = std::fs::read_to_string(&p)
            && let Ok(khz) = s.trim().parse::<u64>()
        {
            best = Some(best.map_or(khz, |b: u64| b.max(khz)));
        }
    }
    best
}
