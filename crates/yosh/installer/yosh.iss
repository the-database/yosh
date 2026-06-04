; yosh — Windows installer (Inno Setup). Per-user, no admin required.
; Build:  ISCC.exe /DAppVersion=0.1.17 yosh.iss   (version defaults to 0.0.0 if omitted)
; Paths are relative to this script's folder (crates/yosh/installer).

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#define AppName "yosh"
#define AppExe "yosh.exe"
#define AppPublisher "the-database"
#define AppUrl "https://github.com/the-database/yosh-rust"

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
Name: "assoc_images"; Description: "&Images  (.png, .jpg, .jpeg, .webp, .gif, .bmp, .avif)"; GroupDescription: "Open these file types with yosh:"; Flags: unchecked

[Files]
Source: "..\..\..\target\release\{#AppExe}"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\assets\yosh.ico"; DestDir: "{app}"; Flags: ignoreversion

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
; ---- ProgIDs (the handlers yosh.exe registers) ----
Root: HKCU; Subkey: "Software\Classes\yosh.comic"; ValueType: string; ValueName: ""; ValueData: "Comic archive"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\yosh.comic\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#AppExe},0"
Root: HKCU; Subkey: "Software\Classes\yosh.comic\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""
Root: HKCU; Subkey: "Software\Classes\yosh.image"; ValueType: string; ValueName: ""; ValueData: "Image"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\yosh.image\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#AppExe},0"
Root: HKCU; Subkey: "Software\Classes\yosh.image\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""

; ---- Comic extensions: make yosh the handler (these are normally unassociated) ----
Root: HKCU; Subkey: "Software\Classes\.cbz"; ValueType: string; ValueName: ""; ValueData: "yosh.comic"; Tasks: assoc_comics; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.cbz\OpenWithProgids"; ValueType: string; ValueName: "yosh.comic"; ValueData: ""; Tasks: assoc_comics; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.cbr"; ValueType: string; ValueName: ""; ValueData: "yosh.comic"; Tasks: assoc_comics; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.cbr\OpenWithProgids"; ValueType: string; ValueName: "yosh.comic"; ValueData: ""; Tasks: assoc_comics; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.cb7"; ValueType: string; ValueName: ""; ValueData: "yosh.comic"; Tasks: assoc_comics; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.cb7\OpenWithProgids"; ValueType: string; ValueName: "yosh.comic"; ValueData: ""; Tasks: assoc_comics; Flags: uninsdeletevalue

; ---- Image extensions: make yosh the handler + add to "Open with" ----
; (For an image type another app already owns, Windows keeps that default until
;  the user confirms via "Open with > Always" or Settings > Default apps.)
Root: HKCU; Subkey: "Software\Classes\.png"; ValueType: string; ValueName: ""; ValueData: "yosh.image"; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.png\OpenWithProgids"; ValueType: string; ValueName: "yosh.image"; ValueData: ""; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.jpg"; ValueType: string; ValueName: ""; ValueData: "yosh.image"; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.jpg\OpenWithProgids"; ValueType: string; ValueName: "yosh.image"; ValueData: ""; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.jpeg"; ValueType: string; ValueName: ""; ValueData: "yosh.image"; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.jpeg\OpenWithProgids"; ValueType: string; ValueName: "yosh.image"; ValueData: ""; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.webp"; ValueType: string; ValueName: ""; ValueData: "yosh.image"; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.webp\OpenWithProgids"; ValueType: string; ValueName: "yosh.image"; ValueData: ""; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.gif"; ValueType: string; ValueName: ""; ValueData: "yosh.image"; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.gif\OpenWithProgids"; ValueType: string; ValueName: "yosh.image"; ValueData: ""; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.bmp"; ValueType: string; ValueName: ""; ValueData: "yosh.image"; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.bmp\OpenWithProgids"; ValueType: string; ValueName: "yosh.image"; ValueData: ""; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.avif"; ValueType: string; ValueName: ""; ValueData: "yosh.image"; Tasks: assoc_images; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.avif\OpenWithProgids"; ValueType: string; ValueName: "yosh.image"; ValueData: ""; Tasks: assoc_images; Flags: uninsdeletevalue

; ---- App registration so yosh shows in "Open with" and Settings > Default apps ----
Root: HKCU; Subkey: "Software\Classes\Applications\{#AppExe}"; ValueType: string; ValueName: "FriendlyAppName"; ValueData: "{#AppName}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\Applications\{#AppExe}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#AppExe},0"
Root: HKCU; Subkey: "Software\Classes\Applications\{#AppExe}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities"; ValueType: string; ValueName: "ApplicationName"; ValueData: "{#AppName}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities"; ValueType: string; ValueName: "ApplicationDescription"; ValueData: "High-throughput manga / comic / image reader"
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".cbz"; ValueData: "yosh.comic"; Tasks: assoc_comics
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".cbr"; ValueData: "yosh.comic"; Tasks: assoc_comics
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".cb7"; ValueData: "yosh.comic"; Tasks: assoc_comics
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".png"; ValueData: "yosh.image"; Tasks: assoc_images
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".jpg"; ValueData: "yosh.image"; Tasks: assoc_images
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".jpeg"; ValueData: "yosh.image"; Tasks: assoc_images
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".webp"; ValueData: "yosh.image"; Tasks: assoc_images
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".gif"; ValueData: "yosh.image"; Tasks: assoc_images
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".bmp"; ValueData: "yosh.image"; Tasks: assoc_images
Root: HKCU; Subkey: "Software\{#AppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".avif"; ValueData: "yosh.image"; Tasks: assoc_images
Root: HKCU; Subkey: "Software\RegisteredApplications"; ValueType: string; ValueName: "{#AppName}"; ValueData: "Software\{#AppName}\Capabilities"; Flags: uninsdeletevalue
