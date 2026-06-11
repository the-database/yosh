package com.thedatabase.yosh;

import android.Manifest;
import android.app.NativeActivity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.os.ParcelFileDescriptor;
import android.provider.Settings;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.view.WindowInsets;
import android.view.WindowInsetsController;

// A thin NativeActivity subclass that exists only to bridge the Storage Access
// Framework, which the NDK can't reach: a bare NativeActivity receives no
// onActivityResult, so the document-picker result has nowhere to land. This
// captures it into a static the Rust side polls, plus helpers to launch the
// picker and turn the chosen content:// URI into a file descriptor. The native
// library still loads via the android.app.lib_name manifest meta-data (inherited
// from NativeActivity), so android_main runs exactly as before.
public class YoshActivity extends NativeActivity {
    private static final int PICK_REQUEST = 1001;
    private static final int READ_REQUEST = 1002;

    /** Set by onActivityResult; read + cleared by the native side via takePickedUri. */
    public static volatile String pickedUri = null;

    /** Read + clear the picked URI (null if none). An *instance* method so native
     *  can call it on the activity object — a static call needs JNI FindClass,
     *  which from the native thread can't see app (dex) classes. */
    public String takePickedUri() {
        String u = pickedUri;
        pickedUri = null;
        return u;
    }

    /** True if the app can read the shared storage to browse the library by path.
     *  Android 11+ uses all-files access (MANAGE_EXTERNAL_STORAGE); Android 10 and
     *  below use the legacy storage model gated on runtime READ_EXTERNAL_STORAGE
     *  (which, unlike pre-30 before, we must actually check — the blanket `true`
     *  let the UI think it had access while every std::fs read silently failed). */
    public boolean hasAllFiles() {
        if (Build.VERSION.SDK_INT >= 30) {
            return Environment.isExternalStorageManager();
        }
        return checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE)
                == PackageManager.PERMISSION_GRANTED;
    }

    /** Ask the user for storage access. Called from native. Android 11+ opens the
     *  all-files Settings page; Android 10 and below show the runtime READ permission
     *  dialog. After granting, the native side re-polls hasAllFiles() on resume. */
    public void requestAllFiles() {
        if (Build.VERSION.SDK_INT >= 30) {
            runOnUiThread(() -> {
                try {
                    Intent intent =
                        new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION);
                    intent.setData(Uri.parse("package:" + getPackageName()));
                    startActivity(intent);
                } catch (Exception e) {
                    startActivity(new Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION));
                }
            });
        } else {
            runOnUiThread(() -> requestPermissions(
                new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, READ_REQUEST));
        }
    }

    /** Launch the SAF document picker. Called from native via JNI. */
    public void openDocument() {
        // startActivityForResult must run on the UI thread.
        runOnUiThread(() -> {
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            // Comic archives (and a catch-all so odd MIME types still show).
            intent.setType("*/*");
            startActivityForResult(intent, PICK_REQUEST);
        });
    }

    /** Hide/show the system bars (status + navigation). Called from native via JNI.
     *  immersive=true → hide with swipe-to-reveal (sticky); false → show. The window
     *  is always edge-to-edge (full-bleed): NativeActivity takes the window surface
     *  directly, so it ignores fitSystemWindows and the surface spans the whole
     *  screen regardless — the native side insets its own chrome (see
     *  statusBarHeight) when the bars are shown. */
    public void setImmersive(boolean immersive) {
        runOnUiThread(() -> {
            Window window = getWindow();
            if (Build.VERSION.SDK_INT >= 30) {
                window.setDecorFitsSystemWindows(false);
                WindowInsetsController c = window.getInsetsController();
                if (c != null) {
                    int bars = WindowInsets.Type.statusBars()
                             | WindowInsets.Type.navigationBars();
                    if (immersive) {
                        c.setSystemBarsBehavior(
                            WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE);
                        c.hide(bars);
                    } else {
                        c.show(bars);
                    }
                }
            } else {
                // Pre-API-30: deprecated visibility flags.
                View decor = window.getDecorView();
                int layout = View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                           | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                           | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN;
                if (immersive) {
                    decor.setSystemUiVisibility(layout
                        | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_FULLSCREEN
                        | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY);
                } else {
                    decor.setSystemUiVisibility(layout);
                }
            }
        });
    }

    /** Status-bar height in px (0 if unknown). A constant from the platform
     *  resource, so it doesn't depend on the bars' current (async) visibility — the
     *  native side pads its top chrome by this while the bars are shown. */
    public int statusBarHeight() {
        int id = getResources().getIdentifier("status_bar_height", "dimen", "android");
        return id > 0 ? getResources().getDimensionPixelSize(id) : 0;
    }

    /** Open a previously-picked content:// URI as an owned file descriptor (-1 on
     *  failure). The caller (Rust) owns the fd and must close it. */
    public int openFd(String uriStr) {
        try {
            ParcelFileDescriptor pfd =
                getContentResolver().openFileDescriptor(Uri.parse(uriStr), "r");
            // detachFd() transfers ownership out of the PFD to the caller.
            return pfd.detachFd();
        } catch (Exception e) {
            return -1;
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        Log.i("yosh_java", "onActivityResult req=" + requestCode + " res=" + resultCode
                + " hasData=" + (data != null));
        if (requestCode == PICK_REQUEST && resultCode == RESULT_OK && data != null) {
            Uri uri = data.getData();
            if (uri != null) {
                pickedUri = uri.toString();
                Log.i("yosh_java", "pickedUri=" + pickedUri);
            }
        }
    }
}
