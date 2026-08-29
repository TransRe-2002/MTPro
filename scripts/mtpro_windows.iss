; MTPro Windows 安装包 — Inno Setup 脚本
;
; 输入: outputs/mtpro-windows-x86_64/  (Nuitka --standalone 产出的完整目录)
; 输出: outputs/MTPro-Setup-<version>.exe
;
; 用法:
;   iscc /DMyAppVersion=1.2.3 scripts\mtpro_windows.iss
; version 缺省为 1.0.0,可由 build_nuitka.ps1 -Installer 或 CI 自动注入。

#ifndef MyAppName
  #define MyAppName "MTPro"
#endif
#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif
#ifndef MyAppPublisher
  #define MyAppPublisher "TransRe"
#endif
#ifndef MyAppExeName
  #define MyAppExeName "MTPro.exe"
#endif

[Setup]
AppId={{B7C3F5E2-4A9D-4C6B-9E8F-1A2B3C4D5E6F}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\outputs
OutputBaseFilename=MTPro-Setup-{#MyAppVersion}
; 有应用图标后,可取消下面两行让安装包/快捷方式带图(需 python/src/mtpro.ico):
; SetupIconFile=..\python\src\mtpro.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExeName}
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加图标:"; Flags: unchecked

[Files]
; 相对路径均相对本 .iss 所在目录 (scripts/)
Source: "..\outputs\mtpro-windows-x86_64\*"; DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs; \
    Excludes: "bin\linux"

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "启动 {#MyAppName}"; \
    Flags: nowait postinstall skipifsilent