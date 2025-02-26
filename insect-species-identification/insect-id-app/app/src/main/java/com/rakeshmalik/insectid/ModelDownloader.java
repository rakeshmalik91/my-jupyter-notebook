package com.rakeshmalik.insectid;

import static com.rakeshmalik.insectid.Constants.*;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.widget.TextView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Consumer;

import okhttp3.*;

public class ModelDownloader {

    private final OkHttpClient client = new OkHttpClient();
    private final Context context;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private final SharedPreferences prefs;
    private final TextView outputText;
    private final MetadataManager metadataManager;

    public ModelDownloader(Context context, TextView outputText, MetadataManager metadataManager) {
        this.context = context;
        this.outputText = outputText;
        this.prefs = context.getSharedPreferences(PREF, Context.MODE_PRIVATE);
        this.metadataManager = metadataManager;
    }

    public boolean isModelAlreadyDownloaded(ModelType modelType) {
        String classesFileName = String.format(CLASSES_FILE_NAME_FMT, modelType.modelName);
        String classDetailsFileName = String.format(CLASS_DETAILS_FILE_NAME_FMT, modelType.modelName);
        String modelFileName = String.format(MODEL_FILE_NAME_FMT, modelType.modelName);

        int currentVersion = prefs.getInt(modelVersionPrefName(modelType), 0);
        int latestVersion = metadataManager.getMetadata(modelType).optInt(FIELD_VERSION, 0);
        Log.d(LOG_TAG, String.format("Model type: %s, current version: %d, latest version: %d", modelType.modelName, currentVersion, latestVersion));

        return isFileAlreadyDownloaded(classesFileName)
                && isFileAlreadyDownloaded(classDetailsFileName)
                && isFileAlreadyDownloaded(modelFileName)
                && currentVersion == latestVersion;
    }

    public void downloadModel(ModelType modelType, Runnable onSuccess, Runnable onFailure) {
        try {
            if (isModelAlreadyDownloaded(modelType)) {
                Log.d(LOG_TAG, "Model " + modelType.modelName + " already downloaded.");
                return;
            }

            Log.d(LOG_TAG, "Going to download " + modelType.modelName + " model");

            String classesFileName = String.format(CLASSES_FILE_NAME_FMT, modelType.modelName);
            String classDetailsFileName = String.format(CLASS_DETAILS_FILE_NAME_FMT, modelType.modelName);
            String modelFileName = String.format(MODEL_FILE_NAME_FMT, modelType.modelName);

            String classesFileUrl = metadataManager.getMetadata(modelType).optString(FIELD_CLASSES_URL, null);
            String classDetailsFileUrl = metadataManager.getMetadata(modelType).optString(FIELD_CLASS_DETAILS_URL, null);
            String modelFileUrl = metadataManager.getMetadata(modelType).optString(FIELD_MODEL_URL, null);

            downloadFile(classesFileName, classesFileUrl, null, onFailure, "class list", modelType);
            downloadFile(classDetailsFileName, classDetailsFileUrl, null, onFailure, "class details", modelType);
            downloadFile(modelFileName, modelFileUrl, onSuccess, onFailure, modelType.displayName.toLowerCase() + " model", modelType);
        } catch(Exception ex) {
            if(onFailure != null) {
                onFailure.run();
            }
        }
    }

    private void downloadFile(String fileName, String fileUrl, Runnable onSuccess, Runnable onFailure, String fileType, ModelType modelType) {
        Log.d(LOG_TAG, "Downloading " + fileType + " " + fileName + " from " + fileUrl + "...");
        client.newCall(new Request.Builder().url(fileUrl).build()).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                Log.e(LOG_TAG, "Download " + fileType + " failed: " + e.getMessage());
                mainHandler.post(() -> outputText.setText("Download " + fileType + " failed!"));
                if(onFailure != null) {
                    onFailure.run();
                }
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                if (!response.isSuccessful()) {
                    Log.e(LOG_TAG, "Server error: " + response.code());
                    mainHandler.post(() -> outputText.setText("Download " + fileType + " failed!"));
                    return;
                }
                File cacheDir = context.getCacheDir();
                File file = new File(cacheDir, fileName);
                try(InputStream inputStream = response.body().byteStream();
                    FileOutputStream outputStream = new FileOutputStream(file); ) {
                    byte[] buffer = new byte[4096];
                    long totalBytes = response.body().contentLength();
                    long downloadedBytes = 0;
                    int bytesRead;
                    while ((bytesRead = inputStream.read(buffer)) != -1) {
                        outputStream.write(buffer, 0, bytesRead);
                        downloadedBytes += bytesRead;
                        int progress = (int) ((downloadedBytes * 100) / totalBytes);
                        String msg = String.format("Downloading %s...\n%d%% (%d/%d MB)", fileType, progress, downloadedBytes/1024/1024, totalBytes/1024/1024);
                        mainHandler.post(() -> outputText.setText(msg));
                    }
                    Log.d(LOG_TAG, "File downloaded successfully: " + file.getAbsolutePath());
                    mainHandler.post(() -> outputText.setText("Downloaded " + fileType + " successfully"));
                    if(onSuccess != null) {
                        onSuccess.run();
                    }
                    prefs.edit().putBoolean(fileDownloadedPrefName(fileName), true).apply();
                    if(fileType.contains("model")) {
                        int version = metadataManager.getMetadata(modelType).optInt(FIELD_VERSION, 0);
                        prefs.edit().putInt(modelVersionPrefName(modelType), version).apply();
                    }
                }
            }
        });
    }

    private boolean isFileAlreadyDownloaded(String fileName) {
        File file = new File(context.getCacheDir(), fileName);
        return file.exists() && prefs.getBoolean(fileDownloadedPrefName(fileName), false);
    }

    private String fileDownloadedPrefName(String fileName) {
        return PREF_FILE_DOWNLOADED + "::" + fileName;
    }

    private String modelVersionPrefName(ModelType modelType) {
        return PREF_MODEL_VERSION + "::" + modelType.modelName;
    }

}
