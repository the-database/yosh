package com.thedatabase.yosh;

import android.app.NativeActivity;
import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.os.ParcelFileDescriptor;
import android.provider.Settings;
import android.util.Log;

// A thin NativeActivity subclass that exists only to bridge the Storage Access
// Framework, which the NDK can't reach: a bare NativeActivity receives no
// onActivityResult, so the document-picker result has nowhere to land. This
// captures it into a static the Rust side polls, plus helpers to launch the
// picker and turn the chosen content:// URI into a file descriptor. The native
// library still loads via the android.app.lib_name manifest meta-data (inherited
// from NativeActivity), so android_main runs exactly as before.
public class YoshActivity extends NativeActivity {
    private static final int PICK_REQUEST = 1001;

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

    /** True if the app has all-files access (so it can browse the library by path). */
    public boolean hasAllFiles() {
        if (Build.VERSION.SDK_INT >= 30) {
            return Environment.isExternalStorageManager();
        }
        return true; // pre-Android-11 uses the legacy storage model
    }

    /** Open Settings so the user can grant all-files access. Called from native. */
    public void requestAllFiles() {
        runOnUiThread(() -> {
            try {
                Intent intent = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION);
                intent.setData(Uri.parse("package:" + getPackageName()));
                startActivity(intent);
            } catch (Exception e) {
                startActivity(new Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION));
            }
        });
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
