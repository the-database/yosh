#!/usr/bin/env bash
# yosh file-type icons v7: BandiView-style — a near-full-bleed white "document"
# card, a format-colour image panel with a faint yosh-monkey watermark, and a bold
# (Arial Black) dark label. One design for every format; auto-resized to
# 256/48/32/16. The colour + bold label carry small sizes.
set -e
cd "$(dirname "$0")"
BLACK="C:/Windows/Fonts/ariblk.ttf"
LOGO="../yosh.png"
[ -f "$LOGO" ] || LOGO="C:/Users/jsoos/Documents/programming/yosh-rust/crates/yosh/assets/yosh.png"
rm -f *.ico big_*.png

icon() { # label colour out
  magick "$LOGO" -colorspace gray -fill white -colorize 58% -resize 150x150 _mk.png
  magick -background none -fill "#262626" -font "$BLACK" -pointsize 240 \
    label:"$1" -trim +repage -resize x54 -resize "212x>" _lbl.png
  magick -size 256x256 xc:none \
    -fill "#cfcfcf" -draw "roundrectangle 14,10 242,246 16,16" \
    -fill white   -draw "roundrectangle 18,14 238,242 13,13" \
    -fill "$2"    -draw "roundrectangle 30,26 226,178 10,10" \
    _card.png
  magick _card.png \
    \( _mk.png -channel A -evaluate multiply 0.55 +channel \) -gravity NorthWest -geometry +53+20 -composite \
    _lbl.png -gravity South -geometry +0+20 -composite "$3"
}

build() { icon "$2" "$3" "big_$1.png"; magick "big_$1.png" -define icon:auto-resize=256,48,32,16 "$1.ico"; }

# Comic archives: the same card with a dark book spine on the left (BandiView-style).
book_icon() { # label colour out
  magick "$LOGO" -colorspace gray -fill white -colorize 58% -resize 130x130 _mk.png
  magick -background none -fill "#262626" -font "$BLACK" -pointsize 240 \
    label:"$1" -trim +repage -resize x48 -resize "172x>" _lbl.png
  magick -size 256x256 xc:none \
    -fill "#cfcfcf" -draw "roundrectangle 14,10 242,246 16,16" \
    -fill white     -draw "roundrectangle 18,14 238,242 13,13" \
    -fill "$2"      -draw "roundrectangle 58,26 228,180 9,9" \
    -fill "#383838" -draw "roundrectangle 18,14 52,242 13,13" \
    -fill "#383838" -draw "rectangle 40,14 52,242" \
    _bcard.png
  magick _bcard.png \
    \( _mk.png -channel A -evaluate multiply 0.55 +channel \) -gravity NorthWest -geometry +78+24 -composite \
    _lbl.png -gravity South -geometry +17+22 -composite "$3"
}
build_book() { book_icon "$2" "$3" "big_$1.png"; magick "big_$1.png" -define icon:auto-resize=256,48,32,16 "$1.ico"; }

# existing formats (redesigned)
build png  PNG  "#3DA35D"
build jpg  JPG  "#2D7DD2"
build gif  GIF  "#E84A8A"
build bmp  BMP  "#16A39B"
build webp WEBP "#18B5C4"
build avif AVIF "#E8743B"
build jxl  JXL  "#8E5BD8"
build psd  PSD  "#2B4B7E"
build_book cbz  CBZ  "#3F51B5"
build_book cbr  CBR  "#D23F3F"
build_book cb7  CB7  "#E0A11B"
# new formats
build tif  TIF  "#5D4037"
build tga  TGA  "#00897B"
build dds  DDS  "#6A1B9A"
build exr  EXR  "#00838F"
build qoi  QOI  "#AD1457"
build hdr  HDR  "#EF6C00"

magick montage \
  big_png.png big_jpg.png big_gif.png big_bmp.png big_webp.png big_avif.png \
  big_jxl.png big_psd.png big_cbz.png big_cbr.png big_cb7.png big_tif.png \
  big_tga.png big_dds.png big_exr.png big_qoi.png big_hdr.png \
  -tile 6x3 -geometry 124x124+5+5 -background "#2b2b2b" review_hero.png

# real-32px Details mock for a few
mkrow() { magick -size 470x40 xc:"#fbfbfb" "$1.ico[2]" -gravity NorthWest -geometry +12+4 -composite -font "C:/Windows/Fonts/segoeui.ttf" -pointsize 17 -fill "#1a1a1a" -gravity West -annotate +54+0 "$2" "_row_$1.png"; }
mkrow png "cover.png"; mkrow tif "scan.tif"; mkrow tga "sprite.tga"; mkrow dds "texture.dds"; mkrow exr "render.exr"; mkrow cbz "Volume 01.cbz"
magick _row_png.png _row_tif.png _row_tga.png _row_dds.png _row_exr.png _row_cbz.png -append detail_1x.png
rm -f _*.png
echo done
