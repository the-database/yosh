; yosh — Windows installer (Inno Setup). Per-user, no admin required.
; Build:  ISCC.exe /DAppVersion=0.1.17 yosh.iss   (version defaults to 0.0.0 if omitted)
; Paths are relative to this script's folder (crates/yosh/installer).

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#define AppName "yosh"
#define AppExe "yosh.exe"
#define AppPublisher "the-database"
#define AppUrl "https://github.com/the-database/yosh"

[Setup]
AppId={{6E9B4F2C-1A7D-4B3E-9C82-5D0A1F6E3B4D}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppUrl}
AppSupportURL={#AppUrl}
VersionInfoVersion={#AppVersion}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=Output
OutputBaseFilename=yosh-setup-x64
SetupIconFile=..\assets\yosh.ico
WizardImageFile=..\assets\yosh-wizard.bmp
WizardSmallImageFile=..\assets\yosh-wizard-small.bmp
UninstallDisplayIcon={app}\{#AppExe}
WizardStyle=modern
Compression=lzma2
SolidCompression=yes
ChangesAssociations=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "assoc_comics"; Description: "&Comic archives  (.cbz, .cbr, .cb7)"; GroupDescription: "Open these file types with yosh:"
Name: "assoc_images"; Description: "&Images  (.png, .jpg, .webp, .gif, .bmp, .avif, .jxl, .tif, .tga, .dds, .exr, .qoi, .hdr)"; GroupDescription: "Open these file types with yosh:"; Flags: unchecked
Name: "context_menu"; Description: "Add ""View with yosh"" when right-clicking comics, archives, images, and folders"; GroupDescription: "Right-click menu:"

[Files]
Source: "..\..\..\target\release\{#AppExe}"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\assets\yosh.ico"; DestDir: "{app}"; Flags: ignoreversion
; Per-file-type icons shown in Explorer (one .ico per format, referenced by the
; per-format ProgIDs below).
Source: "..\assets\icons\*.ico"; DestDir: "{app}\icons"; Flags: ignoreversion

; AppUserModelID must match the ID the app sets at runtime
; (SetCurrentProcessExplicitAppUserModelID) so the taskbar resolves the running
; window to this shortcut — enables pinning and a stable taskbar icon.
[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"; AppUserModelID: "the-database.yosh"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon; AppUserModelID: "the-database.yosh"

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch yosh"; Flags: nowait postinstall skipifsilent

[Registry]
; ---- Per-format ProgIDs (one per file type, each carrying its own DefaultIcon
;      so Explorer shows a distinct icon per format). .jpg/.jpeg share yosh.jpg. ----
Root: HKCU; Subkey: "Software\Classes\yosh.cbz"; ValueType: string; ValueName: ""; ValueData: "CBZ comic"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\yosh.cbz\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\icons\cbz.ico"
Root: HKCU; Subkey: "Software\Classes\yosh.cbz\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""
Root: HKCU; Subkey: "Software\Classes\yosh.cbr"; ValueType: string; ValueName: ""; ValueData: "CBR comic"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\yosh.cbr\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\icons\cbr.ico"
Root: HKCU; Subkey: "Software\Classes\yosh.cbr\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""
Root: HKCU; Subkey: "Software\Classes\yosh.cb7"; ValueType: string; ValueName: ""; ValueData: "CB7 comic"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\yosh.cb7\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\icons\cb7.ico"
Root: HKCU; Subkey: "Software\Classes\yosh.cb7\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""
Root: HKCU; Subkey: "Software\Classes\yosh.png"; ValueType: string; ValueName: ""; ValueData: "PNG image"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\yosh.png\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\icons\png.ico"
Root: HKCU; Subkey: "Software\Classes\yosh.png\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""
Root: HKCU; Subkey: "Software\Classes\yosh.jpg"; ValueType: string; ValueName: ""; ValueData: "JPEG image"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\yosh.jpg\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\icons\jpg.ico"
Root: HKCU; Subkey: "Software\Classes\yosh.jpg\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""
Root: HKCU; Subkey: "Software\Classes\yosh.gif"; ValueType: string; ValueName: ""; ValueData: "GIF image"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\yosh.gif\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\icons\gif.ico"
Root: HKCU; Subkey: "Software\Classes\yosh.gif\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""
Root: HKCU; Subkey: "Software\Classes\yosh.bmp"; ValueType: string; ValueName: ""; ValueData: "BMP image"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\yosh.bmp\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\icons\bmp.ico"
Root: HKCU; Subkey: "Software\Classes\yosh.bmp\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""
Root: HKCU; Subkey: "Software\Classes\yosh.webp"; ValueType: string; ValueName: ""; ValueData: "WebP image"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\yosh.webp\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\icons\webp.ico"
Root: HKCU; Subkey: "Software\Classes\yosh.webp\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""
Root: HKCU; Subkey: "Software\Classes\yosh.avif"; ValueType: string; ValueName: ""; ValueData: "AVIF image"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\yosh.avif\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\icons\avif.ico"
Root: HKCU; Subkey: "Software\Classes\yosh.avif\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""
Root: HKCU; Subkey: "Software\Classes\yosh.jxl"; ValueType: string; ValueName: ""; ValueData: "JPEG XL image"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\yosh.jxl\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\icons\jxl.ico"
Root: HKCU; Subkey: "Software\Classes\yosh.jxl\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""
; PSD ProgID + icon exist, but .psd is only added to "Open with" (additive) below
; — never set as the default — so Photoshop keeps it. The icon shows only if the
; user makes yosh the .psd handler themselves.
Root: HKCU; Subkey: "Software\Classes\yosh.psd"; ValueType: string; ValueName: ""; ValueData: "PSD image"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\yosh.psd\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\icons\psd.ico"
Root: HKCU; Subkey: "Software\Classes\yosh.psd\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""
Root: HKCU; Subkey: "Software\Classes\yosh.tif"; ValueType: string; ValueName: ""; ValueData: "TIFF image"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\yosh.tif\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\icons\tif.ico"
Root: HKCU; Subkey: "Software\Classes\yosh.tif\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""
Root: HKCU; Subkey: "Software\Classes\yosh.tga"; ValueType: string; ValueName: ""; ValueData: "TGA image"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\yosh.tga\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\icons\tga.ico"
Root: HKCU; Subkey: "Software\Classes\yosh.tga\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""
Root: HKCU; Subkey: "Software\Classes\yosh.dds"; ValueType: string; ValueName: ""; ValueData: "DDS texture"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\yosh.dds\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\icons\dds.ico"
Root: HKCU; Subkey: "Software\Classes\yosh.dds\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""
Root: HKCU; Subkey: "Software\Classes\yosh.exr"; ValueType: string; ValueName: ""; ValueData: "OpenEXR image"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\yosh.exr\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\icons\exr.ico"
Root: HKCU; Subkey: "Software\Classes\yosh.exr\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""
Root: HKCU; Subkey: "Software\Classes\yosh.qoi"; ValueType: string; ValueName: ""; ValueData: "QOI image"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\yosh.qoi\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\icons\qoi.ico"
Root: HKCU; Subkey: "Software\Classes\yosh.qoi\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""
Root: HKCU; Subkey: "Software\Classes\yosh.hdr"; ValueType: string; ValueName: ""; ValueData: "Radiance HDR image"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\yosh.hdr\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\icons\hdr.ico"
Root: HKCU; Subkey: "Software\Classes\yosh.hdr\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""

; ---- Comic extensions: make yosh the handler (these are normally unassociated) ----
Root: HKCU; Subkey: "Software\Classes\.cbz"; ValueType: string; ValueName: ""; ValueData: "yosh.cbz"; Tasks: assoc_comics; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.cbz\OpenWithProgids"; ValueType: string; ValueName: "yosh.cbz"; ValueData: ""; Tasks: assoc_comics; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.cbr"; ValueType: string; ValueName: ""; ValueData: "yosh.cbr"; Tasks: assoc_comics; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.cbr\OpenWithProgids"; ValueType: string; ValueName: "yosh.cbr"; ValueData: ""; Tasks: assoc_comics; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.cb7"; ValueType: string; ValueName: ""; ValueData: "yosh.cb7"; Tasks: assoc_comics; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.cb7\OpenWithProgids"; ValueType: string; ValueName: "yosh.cb7"; ValueData: ""; Tasks: assoc_comics; Flags: uninsdeletevalue

; ---- Image extensions: make yosh the handler + add to "Open with" ----
; (For an image type another app already owns, Windows keeps that default until
;  the user confirms via "Open with > Always" or Settings > Default apps.)
Root: HKCU; Subkey: "Software\Classes\.png"; ValueType: string; ValueName: ""; ValueData: "yosh.png"; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.png\OpenWithProgids"; ValueType: string; ValueName: "yosh.png"; ValueData: ""; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.jpg"; ValueType: string; ValueName: ""; ValueData: "yosh.jpg"; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.jpg\OpenWithProgids"; ValueType: string; ValueName: "yosh.jpg"; ValueData: ""; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.jpeg"; ValueType: string; ValueName: ""; ValueData: "yosh.jpg"; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.jpeg\OpenWithProgids"; ValueType: string; ValueName: "yosh.jpg"; ValueData: ""; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.webp"; ValueType: string; ValueName: ""; ValueData: "yosh.webp"; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.webp\OpenWithProgids"; ValueType: string; ValueName: "yosh.webp"; ValueData: ""; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.gif"; ValueType: string; ValueName: ""; ValueData: "yosh.gif"; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.gif\OpenWithProgids"; ValueType: string; ValueName: "yosh.gif"; ValueData: ""; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.bmp"; ValueType: string; ValueName: ""; ValueData: "yosh.bmp"; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.bmp\OpenWithProgids"; ValueType: string; ValueName: "yosh.bmp"; ValueData: ""; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.avif"; ValueType: string; ValueName: ""; ValueData: "yosh.avif"; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.avif\OpenWithProgids"; ValueType: string; ValueName: "yosh.avif"; ValueData: ""; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.jxl"; ValueType: string; ValueName: ""; ValueData: "yosh.jxl"; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.jxl\OpenWithProgids"; ValueType: string; ValueName: "yosh.jxl"; ValueData: ""; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.tif"; ValueType: string; ValueName: ""; ValueData: "yosh.tif"; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.tif\OpenWithProgids"; ValueType: string; ValueName: "yosh.tif"; ValueData: ""; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.tiff"; ValueType: string; ValueName: ""; ValueData: "yosh.tif"; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.tiff\OpenWithProgids"; ValueType: string; ValueName: "yosh.tif"; ValueData: ""; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.tga"; ValueType: string; ValueName: ""; ValueData: "yosh.tga"; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.tga\OpenWithProgids"; ValueType: string; ValueName: "yosh.tga"; ValueData: ""; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.dds"; ValueType: string; ValueName: ""; ValueData: "yosh.dds"; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.dds\OpenWithProgids"; ValueType: string; ValueName: "yosh.dds"; ValueData: ""; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.exr"; ValueType: string; ValueName: ""; ValueData: "yosh.exr"; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.exr\OpenWithProgids"; ValueType: string; ValueName: "yosh.exr"; ValueData: ""; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.qoi"; ValueType: string; ValueName: ""; ValueData: "yosh.qoi"; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.qoi\OpenWithProgids"; ValueType: string; ValueName: "yosh.qoi"; ValueData: ""; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.hdr"; ValueType: string; ValueName: ""; ValueData: "yosh.hdr"; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.hdr\OpenWithProgids"; ValueType: string; ValueName: "yosh.hdr"; ValueData: ""; Tasks: assoc_images; Flags: uninsdeletevalue
; PSD — additive only (adds yosh to "Open with", never the default). Photoshop
; keeps the default; the yosh.psd icon only appears if the user opts yosh in.
Root: HKCU; Subkey: "Software\Classes\.psd\OpenWithProgids"; ValueType: string; ValueName: "yosh.psd"; ValueData: ""; Flags: uninsdeletevalue

; ---- App registration so yosh shows in "Open with" and Settings > Default apps ----
Root: HKCU; Subkey: "Software\Classes\Applications\{#AppExe}"; ValueType: string; ValueName: "FriendlyAppName"; ValueData: "{#AppName}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\Applications\{#AppExe}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#AppExe},0"
Root: HKCU; Subkey: "Software\Classes\Applications\{#AppExe}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities"; ValueType: string; ValueName: "ApplicationName"; ValueData: "{#AppName}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities"; ValueType: string; ValueName: "ApplicationDescription"; ValueData: "High-throughput manga / comic / image reader"
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".cbz"; ValueData: "yosh.cbz"; Tasks: assoc_comics
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".cbr"; ValueData: "yosh.cbr"; Tasks: assoc_comics
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".cb7"; ValueData: "yosh.cb7"; Tasks: assoc_comics
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".png"; ValueData: "yosh.png"; Tasks: assoc_images
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".jpg"; ValueData: "yosh.jpg"; Tasks: assoc_images
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".jpeg"; ValueData: "yosh.jpg"; Tasks: assoc_images
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".webp"; ValueData: "yosh.webp"; Tasks: assoc_images
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".gif"; ValueData: "yosh.gif"; Tasks: assoc_images
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".bmp"; ValueData: "yosh.bmp"; Tasks: assoc_images
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".avif"; ValueData: "yosh.avif"; Tasks: assoc_images
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".jxl"; ValueData: "yosh.jxl"; Tasks: assoc_images
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".tif"; ValueData: "yosh.tif"; Tasks: assoc_images
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".tiff"; ValueData: "yosh.tif"; Tasks: assoc_images
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".tga"; ValueData: "yosh.tga"; Tasks: assoc_images
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".dds"; ValueData: "yosh.dds"; Tasks: assoc_images
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".exr"; ValueData: "yosh.exr"; Tasks: assoc_images
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".qoi"; ValueData: "yosh.qoi"; Tasks: assoc_images
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".hdr"; ValueData: "yosh.hdr"; Tasks: assoc_images
Root: HKCU; Subkey: "Software\RegisteredApplications"; ValueType: string; ValueName: "{#AppName}"; ValueData: "Software\{#AppName}\Capabilities"; Flags: uninsdeletevalue

; ---- "View with yosh" right-click verb (Tasks: context_menu) ----
; Non-destructive: adds a menu entry without changing any default handler, via
; SystemFileAssociations (attaches to the file type, not its ProgID) — so it
; covers .zip even though yosh isn't its default app. Classic registry verb:
; shows in the Win10 menu and the Win11 "Show more options" submenu. (The new
; Win11 top-level menu needs a signed IExplorerCommand package — see notes.)
; Each verb key carries uninsdeletekey, which removes its \command child too.
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.zip\shell\yosh.view"; ValueType: string; ValueName: ""; ValueData: "View with yosh"; Tasks: context_menu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.zip\shell\yosh.view"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExe},0"; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.zip\shell\yosh.view\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.7z\shell\yosh.view"; ValueType: string; ValueName: ""; ValueData: "View with yosh"; Tasks: context_menu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.7z\shell\yosh.view"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExe},0"; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.7z\shell\yosh.view\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.rar\shell\yosh.view"; ValueType: string; ValueName: ""; ValueData: "View with yosh"; Tasks: context_menu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.rar\shell\yosh.view"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExe},0"; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.rar\shell\yosh.view\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.cbz\shell\yosh.view"; ValueType: string; ValueName: ""; ValueData: "View with yosh"; Tasks: context_menu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.cbz\shell\yosh.view"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExe},0"; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.cbz\shell\yosh.view\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.cbr\shell\yosh.view"; ValueType: string; ValueName: ""; ValueData: "View with yosh"; Tasks: context_menu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.cbr\shell\yosh.view"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExe},0"; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.cbr\shell\yosh.view\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.cb7\shell\yosh.view"; ValueType: string; ValueName: ""; ValueData: "View with yosh"; Tasks: context_menu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.cb7\shell\yosh.view"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExe},0"; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.cb7\shell\yosh.view\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.png\shell\yosh.view"; ValueType: string; ValueName: ""; ValueData: "View with yosh"; Tasks: context_menu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.png\shell\yosh.view"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExe},0"; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.png\shell\yosh.view\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jpg\shell\yosh.view"; ValueType: string; ValueName: ""; ValueData: "View with yosh"; Tasks: context_menu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jpg\shell\yosh.view"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExe},0"; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jpg\shell\yosh.view\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jpeg\shell\yosh.view"; ValueType: string; ValueName: ""; ValueData: "View with yosh"; Tasks: context_menu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jpeg\shell\yosh.view"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExe},0"; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jpeg\shell\yosh.view\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.webp\shell\yosh.view"; ValueType: string; ValueName: ""; ValueData: "View with yosh"; Tasks: context_menu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.webp\shell\yosh.view"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExe},0"; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.webp\shell\yosh.view\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.gif\shell\yosh.view"; ValueType: string; ValueName: ""; ValueData: "View with yosh"; Tasks: context_menu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.gif\shell\yosh.view"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExe},0"; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.gif\shell\yosh.view\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.bmp\shell\yosh.view"; ValueType: string; ValueName: ""; ValueData: "View with yosh"; Tasks: context_menu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.bmp\shell\yosh.view"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExe},0"; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.bmp\shell\yosh.view\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.avif\shell\yosh.view"; ValueType: string; ValueName: ""; ValueData: "View with yosh"; Tasks: context_menu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.avif\shell\yosh.view"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExe},0"; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.avif\shell\yosh.view\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jxl\shell\yosh.view"; ValueType: string; ValueName: ""; ValueData: "View with yosh"; Tasks: context_menu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jxl\shell\yosh.view"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExe},0"; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jxl\shell\yosh.view\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.ico\shell\yosh.view"; ValueType: string; ValueName: ""; ValueData: "View with yosh"; Tasks: context_menu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.ico\shell\yosh.view"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExe},0"; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.ico\shell\yosh.view\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.tif\shell\yosh.view"; ValueType: string; ValueName: ""; ValueData: "View with yosh"; Tasks: context_menu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.tif\shell\yosh.view"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExe},0"; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.tif\shell\yosh.view\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.tiff\shell\yosh.view"; ValueType: string; ValueName: ""; ValueData: "View with yosh"; Tasks: context_menu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.tiff\shell\yosh.view"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExe},0"; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.tiff\shell\yosh.view\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.tga\shell\yosh.view"; ValueType: string; ValueName: ""; ValueData: "View with yosh"; Tasks: context_menu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.tga\shell\yosh.view"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExe},0"; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.tga\shell\yosh.view\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.dds\shell\yosh.view"; ValueType: string; ValueName: ""; ValueData: "View with yosh"; Tasks: context_menu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.dds\shell\yosh.view"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExe},0"; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.dds\shell\yosh.view\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.exr\shell\yosh.view"; ValueType: string; ValueName: ""; ValueData: "View with yosh"; Tasks: context_menu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.exr\shell\yosh.view"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExe},0"; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.exr\shell\yosh.view\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.qoi\shell\yosh.view"; ValueType: string; ValueName: ""; ValueData: "View with yosh"; Tasks: context_menu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.qoi\shell\yosh.view"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExe},0"; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.qoi\shell\yosh.view\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.hdr\shell\yosh.view"; ValueType: string; ValueName: ""; ValueData: "View with yosh"; Tasks: context_menu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.hdr\shell\yosh.view"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExe},0"; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.hdr\shell\yosh.view\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: context_menu
; PSD — "View with yosh" right-click only. Deliberately NOT a default association
; (Photoshop keeps that); yosh just offers to open the flattened composite.
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.psd\shell\yosh.view"; ValueType: string; ValueName: ""; ValueData: "View with yosh"; Tasks: context_menu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.psd\shell\yosh.view"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExe},0"; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.psd\shell\yosh.view\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: context_menu
; Folders (right-click a folder of images)
Root: HKCU; Subkey: "Software\Classes\Directory\shell\yosh.view"; ValueType: string; ValueName: ""; ValueData: "View with yosh"; Tasks: context_menu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\Directory\shell\yosh.view"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExe},0"; Tasks: context_menu
Root: HKCU; Subkey: "Software\Classes\Directory\shell\yosh.view\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: context_menu
