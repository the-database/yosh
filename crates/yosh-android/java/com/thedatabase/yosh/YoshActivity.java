package com.thedatabase.yosh;

import android.app.NativeActivity;
import android.content.Intent;
import android.net.Uri;
import android.os.ParcelFileDescriptor;

// A thin NativeActivity subclass that exists only to bridge the Storage Access
// Framework, which the NDK can't reach: a bare NativeActivity receives no
// onActivityResult, so the document-picker result has nowhere to land. This
// captures it into a static the Rust side polls, plus helpers to launch the
// picker and turn the chosen content:// URI into a file descriptor. The native
// library still loads via the android.app.lib_name manifest meta-data (inherited
// from NativeActivity), so android_main runs exactly as before.
public class YoshActivity extends NativeActivity {
    private static final int PICK_REQUEST = 1001;

    /** Set by onActivityResult; polled + cleared by the native side. */
    public static volatile String pickedUri = null;

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
        if (requestCode == PICK_REQUEST && resultCode == RESULT_OK && data != null) {
            Uri uri = data.getData();
            if (uri != null) {
                pickedUri = uri.toString();
            }
        }
    }
}
