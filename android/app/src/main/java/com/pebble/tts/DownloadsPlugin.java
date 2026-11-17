package com.pebble.tts;

import android.Manifest;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Base64;

import com.getcapacitor.JSObject;
import com.getcapacitor.PermissionState;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;
import com.getcapacitor.annotation.Permission;
import com.getcapacitor.annotation.PermissionCallback;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;

/**
 * Writes a Base64 payload straight into the device's public Downloads folder.
 *
 * <p>Android 10 (API 29) and newer: MediaStore.Downloads with RELATIVE_PATH
 * "Download". This needs no storage permission and the entry is published to
 * the media database, so Files/Downloads shows it immediately.
 *
 * <p>Android 9 (API 28) and older: the public Downloads directory through the
 * file system, which requires WRITE_EXTERNAL_STORAGE (declared with
 * maxSdkVersion="28" in the manifest and requested at runtime).
 *
 * <p>Registered manually from MainActivity; no npm package involved.
 */
@CapacitorPlugin(
    name = "PebbleDownloads",
    permissions = { @Permission(alias = "storage", strings = { Manifest.permission.WRITE_EXTERNAL_STORAGE }) }
)
public class DownloadsPlugin extends Plugin {

    private static final String MIME_WAV = "audio/wav";

    @PluginMethod
    public void saveFile(PluginCall call) {
        String fileName = call.getString("fileName");
        String data = call.getString("data");

        if (fileName == null || fileName.trim().isEmpty()) {
            call.reject("A fileName is required.");
            return;
        }
        if (data == null || data.isEmpty()) {
            call.reject("Audio data is required.");
            return;
        }

        // Public Downloads is only reachable through the file system before
        // Android 10, and that needs WRITE_EXTERNAL_STORAGE.
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q && getPermissionState("storage") != PermissionState.GRANTED) {
            requestPermissionForAlias("storage", call, "storagePermissionCallback");
            return;
        }

        writeToDownloads(call, fileName, data);
    }

    @PermissionCallback
    private void storagePermissionCallback(PluginCall call) {
        if (getPermissionState("storage") == PermissionState.GRANTED) {
            writeToDownloads(call, call.getString("fileName"), call.getString("data"));
        } else {
            call.reject("Storage permission is required to save into Downloads on this Android version.");
        }
    }

    private void writeToDownloads(PluginCall call, String fileName, String data) {
        byte[] bytes;
        try {
            bytes = Base64.decode(data, Base64.DEFAULT);
        } catch (IllegalArgumentException ex) {
            call.reject("The audio data is not valid Base64.", ex);
            return;
        }

        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                saveWithMediaStore(call, fileName, bytes);
            } else {
                saveWithFileSystem(call, fileName, bytes);
            }
        } catch (Exception ex) {
            call.reject("Could not save into Downloads: " + ex.getMessage(), ex);
        }
    }

    /** API 29+: publish through MediaStore so Files/Downloads sees it at once. */
    private void saveWithMediaStore(PluginCall call, String fileName, byte[] bytes) throws Exception {
        ContentResolver resolver = getContext().getContentResolver();

        ContentValues values = new ContentValues();
        values.put(MediaStore.MediaColumns.DISPLAY_NAME, fileName);
        values.put(MediaStore.MediaColumns.MIME_TYPE, MIME_WAV);
        values.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS);
        values.put(MediaStore.MediaColumns.IS_PENDING, 1);

        Uri item = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values);
        if (item == null) {
            throw new Exception("MediaStore refused the new Downloads entry.");
        }

        try (OutputStream out = resolver.openOutputStream(item)) {
            if (out == null) {
                throw new Exception("Could not open the Downloads entry for writing.");
            }
            out.write(bytes);
            out.flush();
        } catch (Exception ex) {
            // Never leave a half-written, invisible entry behind.
            resolver.delete(item, null, null);
            throw ex;
        }

        ContentValues published = new ContentValues();
        published.put(MediaStore.MediaColumns.IS_PENDING, 0);
        resolver.update(item, published, null, null);

        resolve(call, fileName, item.toString());
    }

    /** API 24-28: direct file write into /storage/emulated/0/Download. */
    private void saveWithFileSystem(PluginCall call, String fileName, byte[] bytes) throws Exception {
        File dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
        if (dir == null || (!dir.exists() && !dir.mkdirs())) {
            throw new Exception("The Downloads folder is unavailable.");
        }

        File out = new File(dir, fileName);
        try (FileOutputStream fos = new FileOutputStream(out)) {
            fos.write(bytes);
            fos.flush();
        }

        resolve(call, fileName, Uri.fromFile(out).toString());
    }

    private void resolve(PluginCall call, String fileName, String uri) {
        JSObject result = new JSObject();
        result.put("fileName", fileName);
        result.put("uri", uri);
        result.put("location", Environment.DIRECTORY_DOWNLOADS);
        call.resolve(result);
    }
}
