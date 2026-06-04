//! Self-update against the public GitHub Releases API. `check()` runs on a
//! background thread at startup; if a newer release exists the UI offers to
//! download the new binary and replace the running exe in place — no admin, and
//! it works for both the per-user install and the portable build (the portable
//! marker + settings sit next to the exe and are untouched).

use std::time::Duration;

const REPO: &str = "the-database/yosh";

/// The release asset to download for this platform (the bare executable).
#[cfg(windows)]
const ASSET: &str = "yosh-windows-x64.exe";
#[cfg(not(windows))]
const ASSET: &str = "yosh-linux-x64";

const UA: &str = concat!("yosh/", env!("CARGO_PKG_VERSION"));

#[derive(Clone)]
pub struct Update {
    pub version: String,      // e.g. "0.1.19"
    pub download_url: String, // browser_download_url of ASSET
}

/// Query the latest release; `Some` if it is newer than this build. Blocking —
/// call on a background thread. `None` on any error or when already current.
pub fn check() -> Option<Update> {
    let url = format!("https://api.github.com/repos/{REPO}/releases/latest");
    let body = ureq::get(&url)
        .set("User-Agent", UA)
        .set("Accept", "application/vnd.github+json")
        .timeout(Duration::from_secs(10))
        .call()
        .ok()?
        .into_string()
        .ok()?;
    let json: serde_json::Value = serde_json::from_str(&body).ok()?;
    let latest = json["tag_name"].as_str()?.trim_start_matches('v').to_string();
    if !newer(&latest, env!("CARGO_PKG_VERSION")) {
        return None;
    }
    let download_url = json["assets"]
        .as_array()?
        .iter()
        .find(|a| a["name"].as_str() == Some(ASSET))?["browser_download_url"]
        .as_str()?
        .to_string();
    Some(Update { version: latest, download_url })
}

/// Download the new binary and replace the running exe in place. Blocking — call
/// on a background thread. On success the caller relaunches the (new) exe.
pub fn apply(update: &Update) -> Result<(), String> {
    let mut reader = ureq::get(&update.download_url)
        .set("User-Agent", UA)
        .timeout(Duration::from_secs(300))
        .call()
        .map_err(|e| e.to_string())?
        .into_reader();
    let tmp = std::env::temp_dir().join("yosh-update.bin");
    {
        let mut f = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
        std::io::copy(&mut reader, &mut f).map_err(|e| e.to_string())?;
    }
    // Guard against replacing the live exe with a non-executable body: a 200 OK
    // carrying an HTML/XML error page (CDN hiccup, expired signature served as 200)
    // would otherwise be installed verbatim and brick the install. Require the
    // platform executable magic before handing the file to self_replace.
    if !is_executable(&tmp) {
        let _ = std::fs::remove_file(&tmp);
        return Err("downloaded update was not a valid executable".into());
    }
    self_replace::self_replace(&tmp).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&tmp);
    Ok(())
}

/// Cheap sanity check that a downloaded file is actually a native executable:
/// `MZ` (PE) on Windows, the ELF magic elsewhere. Closes the wrong-content brick path.
fn is_executable(path: &std::path::Path) -> bool {
    use std::io::Read;
    let mut head = [0u8; 4];
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    if f.read_exact(&mut head).is_err() {
        return false;
    }
    #[cfg(windows)]
    {
        head[..2] == [0x4D, 0x5A] // 'MZ'
    }
    #[cfg(not(windows))]
    {
        head == [0x7F, b'E', b'L', b'F']
    }
}

/// `a > b` for dotted numeric versions (e.g. "0.1.19" > "0.1.18"). Each component
/// is read up to its first non-digit, so a semver pre-release suffix degrades
/// sanely ("0.1.20-rc1" → [0,1,20]) instead of collapsing that field to 0 and
/// corrupting the comparison.
fn newer(a: &str, b: &str) -> bool {
    fn parts(s: &str) -> Vec<u64> {
        s.split('.')
            .map(|p| {
                let digits: String = p.chars().take_while(|c| c.is_ascii_digit()).collect();
                digits.parse().unwrap_or(0)
            })
            .collect()
    }
    parts(a) > parts(b)
}
