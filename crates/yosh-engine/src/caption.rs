//! Library caption text: turn volume filenames into short, *distinguishing* tile
//! captions.
//!
//! Volume names in a series almost always repeat the series name
//! (`Fate kaleid liner PRISMA ILLYA v03 (2016) (Digital).cbz`), so a caption that
//! truncates from the end shows the same prefix on every tile. The tiles already
//! sit under a series header, so the shared head is redundant: strip the archive
//! extension and the prefix common to the whole series, leaving the part that
//! actually differs (`v03 (2016) (Digital)`).

/// `name` minus a known comic-archive extension, compared case-insensitively:
/// `cbz`, `zip`, `cbr`, `rar`, `7z`, `cb7`. Folder volumes and unknown suffixes
/// pass through untouched, so a folder named `Vol. 1` keeps its `.1`.
pub fn display_stem(name: &str) -> &str {
    match name.rsplit_once('.') {
        Some((stem, ext))
            if ["cbz", "zip", "cbr", "rar", "7z", "cb7"]
                .iter()
                .any(|k| ext.eq_ignore_ascii_case(k)) =>
        {
            stem
        }
        _ => name,
    }
}

/// Separators a common prefix may be cut at, so a caption never starts mid-word.
fn is_sep(c: char) -> bool {
    matches!(c, ' ' | '-' | '_')
}

/// Per-volume display captions for one series row, in the same order as `names`:
/// extension-stripped; with two or more volumes, also stripped of the prefix
/// common to all the stems — but only back to the last separator within that
/// prefix, so a caption never starts mid-word. Never empty: a volume left with
/// nothing falls back to its extension-stripped name.
pub fn series_captions(names: &[&str]) -> Vec<String> {
    let stems: Vec<&str> = names.iter().map(|n| display_stem(n)).collect();
    if stems.len() < 2 {
        return stems.iter().map(|s| (*s).to_string()).collect();
    }
    // Longest prefix shared by every stem, measured in whole chars so the cut can
    // never land inside a multi-byte sequence.
    let mut common = stems[0].len();
    for s in &stems[1..] {
        let mut n = 0;
        for ((i, a), b) in stems[0].char_indices().zip(s.chars()) {
            if a != b {
                break;
            }
            n = i + a.len_utf8();
        }
        common = common.min(n);
        if common == 0 {
            break;
        }
    }
    // Back off to the last separator in the shared head (and past the run it ends,
    // since every separator in the head moves the cut). No separator at all means
    // the names are one word — strip nothing rather than chop a word in half.
    let mut cut = 0;
    for (i, c) in stems[0][..common].char_indices() {
        if is_sep(c) {
            cut = i + c.len_utf8();
        }
    }
    stems
        .iter()
        .map(|s| {
            let rest = s
                .get(cut..)
                .unwrap_or("")
                .trim_start_matches(|c: char| c.is_whitespace() || is_sep(c));
            if rest.is_empty() {
                (*s).to_string()
            } else {
                rest.to_string()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stem_strips_only_archive_extensions() {
        assert_eq!(display_stem("v01.cbz"), "v01");
        assert_eq!(display_stem("v01.CBZ"), "v01");
        assert_eq!(display_stem("book.cb7"), "book");
        assert_eq!(display_stem("book.7z"), "book");
        assert_eq!(display_stem("book.RaR"), "book");
        // A folder volume keeps its dotted name — no naive `file_stem`.
        assert_eq!(display_stem("Vol. 1"), "Vol. 1");
        assert_eq!(display_stem("notes.txt"), "notes.txt");
        assert_eq!(display_stem("no extension here"), "no extension here");
    }

    #[test]
    fn single_volume_keeps_its_whole_name() {
        assert_eq!(
            series_captions(&["Some Series v01 (2016).cbz"]),
            vec!["Some Series v01 (2016)"]
        );
    }

    #[test]
    fn strips_series_prefix_shared_by_every_volume() {
        let names = [
            "Fate kaleid liner PRISMA ILLYA v01 (2016) (Digital) (LuCaZ).cbz",
            "Fate kaleid liner PRISMA ILLYA v02 (2016) (Digital) (LuCaZ).cbz",
            "Fate kaleid liner PRISMA ILLYA v03 (2016) (Digital) (LuCaZ).cbz",
        ];
        assert_eq!(
            series_captions(&names),
            vec![
                "v01 (2016) (Digital) (LuCaZ)",
                "v02 (2016) (Digital) (LuCaZ)",
                "v03 (2016) (Digital) (LuCaZ)",
            ]
        );
    }

    #[test]
    fn cut_lands_on_a_word_boundary() {
        // The shared head is "Series v0" — cutting there would leave "1"/"2".
        assert_eq!(
            series_captions(&["Series v01.cbz", "Series v02.cbz"]),
            vec!["v01", "v02"]
        );
    }

    #[test]
    fn dashes_and_underscores_are_separators() {
        assert_eq!(
            series_captions(&["Series - v01.cbz", "Series - v02.cbz"]),
            vec!["v01", "v02"]
        );
        assert_eq!(
            series_captions(&["Series_v01.cbz", "Series_v02.cbz"]),
            vec!["v01", "v02"]
        );
        // A run of separators is consumed whole.
        assert_eq!(
            series_captions(&["Series -_ v01.cbz", "Series -_ v02.cbz"]),
            vec!["v01", "v02"]
        );
    }

    #[test]
    fn one_word_names_are_left_alone() {
        assert_eq!(
            series_captions(&["vol1.cbz", "vol2.cbz"]),
            vec!["vol1", "vol2"]
        );
    }

    #[test]
    fn captions_are_never_empty() {
        // The same volume as a folder and as an archive.
        assert_eq!(series_captions(&["X", "X.cbz"]), vec!["X", "X"]);
        // One name a strict prefix of another: the shorter keeps its full stem.
        assert_eq!(
            series_captions(&["Series", "Series v02.cbz"]),
            vec!["Series", "Series v02"]
        );
        // The remainder after the cut is empty for the first volume.
        assert_eq!(
            series_captions(&["Series - ", "Series - v02.cbz"]),
            vec!["Series - ", "v02"]
        );
    }

    #[test]
    fn multibyte_names_are_split_on_char_boundaries() {
        assert_eq!(
            series_captions(&["ワンピース 第01巻.cbz", "ワンピース 第02巻.cbz"]),
            vec!["第01巻", "第02巻"]
        );
        // The stems diverge *inside* a multi-byte char (第 and 秒 share a lead byte).
        assert_eq!(series_captions(&["巻 第x", "巻 秒x"]), vec!["第x", "秒x"]);
    }

    #[test]
    fn degenerate_input_does_not_panic() {
        assert_eq!(series_captions(&[]), Vec::<String>::new());
        assert_eq!(series_captions(&["", ""]), vec!["", ""]);
        assert_eq!(
            series_captions(&["", "Series v01.cbz"]),
            vec!["", "Series v01"]
        );
        // A stem no longer than the shared head of the others.
        assert_eq!(
            series_captions(&["Series v01 extra", "Series v01", "Series v01 more"]),
            vec!["v01 extra", "v01", "v01 more"]
        );
    }
}
