#!/usr/bin/env bash
# yosh file-type icons v5: the format label takes ~half the icon (the only way it
# stays readable at the ~32px Details-view size), with a smaller motif on top:
# images = photo frame + faded monkey; archives = book. One design, auto-resized.
set -e
cd "$(dirname "$0")"
FONT="C:/Windows/Fonts/arialbd.ttf"
LOGO="../yosh.png"
rm -f *.ico big_*.png

# Trim to ink, then size to a UNIFORM cap height (x96); only over-wide 4-char
# labels shrink to fit the band width (198), so sizes look consistent.
label() { magick -background none -fill white -font "$FONT" -pointsize 240 label:"$1" -trim +repage -resize x96 -resize "198x>" _lbl.png; }

img_big() { # label color out
  magick "$LOGO" -colorspace gray -fill "$2" -colorize 68% -resize 92x92 _mk.png
  label "$1"
  magick -size 256x256 xc:none \
    -fill "$2" -draw "roundrectangle 30,10 226,118 14,14" \
    -fill white -draw "roundrectangle 38,18 218,110 7,7" \
    -fill "$2" -draw "roundrectangle 26,126 230,250 16,16" \
    _base.png
  magick _base.png \
    \( _mk.png -channel A -evaluate multiply 0.55 +channel \) -gravity NorthWest -geometry +82+17 -composite \
    _lbl.png -gravity South -geometry +0+15 -composite "$3"
}

book_big() { # label color spine out
  magick "$LOGO" -colorspace gray -fill white -colorize 55% -resize 66x66 _mkl.png
  label "$1"
  magick -size 256x256 xc:none \
    -fill "#d7d7d7" -draw "roundrectangle 110,14 178,112 6,6" \
    -fill "#bdbdbd" -draw "roundrectangle 104,10 172,108 6,6" \
    -fill "$2" -draw "roundrectangle 84,8 160,114 8,8" \
    -fill "$3" -draw "roundrectangle 84,8 104,114 8,8" \
    _bk.png
  magick _bk.png \
    \( _mkl.png -channel A -evaluate multiply 0.4 +channel \) -gravity NorthWest -geometry +98+28 -composite \
    -fill "$2" -draw "roundrectangle 26,126 230,250 16,16" \
    _lbl.png -gravity South -geometry +0+15 -composite "$4"
}

build() { local f="$1"; if [ "$4" = "--img" ]; then img_big "$2" "$3" "big_$f.png"; else book_big "$2" "$3" "$4" "big_$f.png"; fi; magick "big_$f.png" -define icon:auto-resize=256,48,32,16 "$f.ico"; }

build png  PNG  "#3DA35D" --img
build jpg  JPG  "#2D7DD2" --img
build gif  GIF  "#E84A8A" --img
build bmp  BMP  "#16A39B" --img
build webp WEBP "#18B5C4" --img
build avif AVIF "#E8743B" --img
build jxl  JXL  "#8E5BD8" --img
build psd  PSD  "#2B4B7E" --img
build cbz  CBZ  "#3F51B5" "#2C397F"
build cbr  CBR  "#D23F3F" "#8F2A2A"
build cb7  CB7  "#E0A11B" "#9C6E0F"

magick montage big_png.png big_jpg.png big_gif.png big_bmp.png big_webp.png big_avif.png big_jxl.png big_psd.png big_cbz.png big_cbr.png big_cb7.png \
  -tile 6x2 -geometry 128x128+6+6 -background "#202020" review_hero.png
# real-size Details mock at 32px
mkrow() { magick -size 470x40 xc:"#fbfbfb" "$1.ico[2]" -gravity NorthWest -geometry +12+4 -composite -font "C:/Windows/Fonts/segoeui.ttf" -pointsize 17 -fill "#1a1a1a" -gravity West -annotate +54+0 "$2" "_row_$1.png"; }
mkrow cbz "One Piece v01.cbz"; mkrow png "cover.png"; mkrow webp "art.webp"; mkrow avif "scan.avif"; mkrow cbr "Berserk.cbr"; mkrow jxl "page.jxl"
magick _row_cbz.png _row_png.png _row_webp.png _row_avif.png _row_cbr.png _row_jxl.png -append detail_1x.png
magick png.ico[2] webp.ico[2] avif.ico[2] cbz.ico[2] cbr.ico[2] cb7.ico[2] -background "#fafafa" +append _s.png
magick _s.png -filter point -resize 500% small_zoom.png
rm -f _*.png
echo done
