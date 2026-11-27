//! The OS seam: what the shell asks of the platform crate.

use super::*;

/// The slow half of opening a picked document: read and parse it into a page
/// source. Built on the UI thread by [`Platform::open_picked`], run on the open
/// worker.
pub type PickedJob = Box<dyn FnOnce() -> Result<Arc<dyn PageSource>, String> + Send>;

/// Everything the mobile shell asks of the OS. Android implements it over JNI to
/// `YoshActivity` plus `/proc` and sysfs probes (`yosh_android::AndroidPlatform`).
///
/// Every method is called on the event-loop thread, so an implementation may hold
/// thread-bound OS objects; only the [`PickedJob`] it returns crosses threads.
pub trait Platform: 'static {
    /// The app-private directory the shell's state files live in.
    fn data_dir(&self) -> Option<PathBuf>;
    /// The library folder used until the user picks one (and the test for "no
    /// library configured yet").
    fn default_library_root(&self) -> PathBuf;
    /// Whether the app may browse the library folder by path.
    fn has_library_access(&self) -> bool;
    /// Ask the user for that access (Android: the all-files settings screen).
    fn request_library_access(&self);
    /// Show the system document picker; its result is collected by `take_picked`.
    fn launch_picker(&self);
    /// The identity key of a document the picker has returned, if one has.
    fn take_picked(&self) -> Option<String>;
    /// Resolve a picked document's key on this thread and return the job that
    /// reads it. `None` = it couldn't be opened (already logged).
    fn open_picked(&self, key: &str) -> Option<PickedJob>;
    /// The OS is in dark mode (resolves the `System` theme).
    fn system_dark(&self) -> bool;
    /// The UI locale's BCP-47 tag ("" if unknown, which resolves to English).
    fn locale_tag(&self) -> String;
    /// Hold (true) / release (false) the screen awake.
    fn set_keep_screen_on(&self, on: bool);
    /// Hide (true) / show (false) the system bars.
    fn set_immersive(&self, on: bool);
    /// How far chrome must sit below the top edge while the bars show, in px.
    fn top_inset_px(&self) -> i32;
    /// `(total, available)` device RAM in MB.
    fn mem_mb(&self) -> (u64, u64);
    /// Highest CPU clock the device advertises, in kHz, if readable.
    fn max_cpu_khz(&self) -> Option<u64>;
    /// A CJK font collection (Noto Sans CJK face order) for the UI's fallback.
    fn cjk_font_bytes(&self) -> Option<Vec<u8>>;
    /// A key press the shell didn't use (Android: a volume key goes back to the
    /// system volume).
    fn unused_key(&self, code: KeyCode);
}
