package com.rakeshmalik.insectid;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.widget.TextView;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Scanner;
import java.util.function.Consumer;
import java.util.zip.CRC32C;

import okhttp3.*;

public class ModelDownloader {

    private final OkHttpClient client = new OkHttpClient();
    private final Context context;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private final SharedPreferences prefs;
    private final TextView outputText;

    public ModelDownloader(Context context, TextView outputText) {
        this.context = context;
        this.outputText = outputText;
        this.prefs = context.getSharedPreferences(Constants.PREF, Context.MODE_PRIVATE);
    }

    public boolean isModelAlreadyDownloaded(ModelType modelType) {
        String classesFileName = String.format(Constants.CLASSES_FILE_NAME_FMT, modelType.modelName);
        String classDetailsFileName = String.format(Constants.CLASS_DETAILS_FILE_NAME_FMT, modelType.modelName);
        String modelFileName = String.format(Constants.MODEL_FILE_NAME_FMT, modelType.modelName);

        String classesFileUrl = String.format(Constants.CLASSES_URL_FMT, modelType.modelName);
        String classDetailsFileUrl = String.format(Constants.CLASS_DETAILS_URL_FMT, modelType.modelName);
        String modelFileUrl = String.format(Constants.MODEL_URL_FMT, modelType.modelName);

        return isFileAlreadyDownloaded(classesFileName, classesFileUrl)
                && isFileAlreadyDownloaded(classDetailsFileName, classDetailsFileUrl)
                && isFileAlreadyDownloaded(modelFileName, modelFileUrl);
    }

    public void downloadModel(ModelType modelType, Consumer<Context> callback) {
        if (isModelAlreadyDownloaded(modelType)) {
            Log.d(Constants.LOG_TAG, "Model already downloaded.");
            return;
        }

        String classesFileName = String.format(Constants.CLASSES_FILE_NAME_FMT, modelType.modelName);
        String classDetailsFileName = String.format(Constants.CLASS_DETAILS_FILE_NAME_FMT, modelType.modelName);
        String modelFileName = String.format(Constants.MODEL_FILE_NAME_FMT, modelType.modelName);

        String classesFileUrl = String.format(Constants.CLASSES_URL_FMT, modelType.modelName);
        String classDetailsFileUrl = String.format(Constants.CLASS_DETAILS_URL_FMT, modelType.modelName);
        String modelFileUrl = String.format(Constants.MODEL_URL_FMT, modelType.modelName);

        downloadFile(classesFileName, classesFileUrl, null, "classes");
        downloadFile(classDetailsFileName, classDetailsFileUrl, null, "metadata");
        downloadFile(modelFileName, modelFileUrl, callback, modelType.displayName.toLowerCase() + " model");
    }

    private void downloadFile(String fileName, String fileUrl, Consumer<Context> callback, String fileType) {
        Log.d(Constants.LOG_TAG, "Downloading " + fileType + " " + fileName + " from " + fileUrl + "...");
        client.newCall(new Request.Builder().url(fileUrl).build()).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                Log.e(Constants.LOG_TAG, "Download " + fileType + " failed: " + e.getMessage());
                mainHandler.post(() -> outputText.setText("Download " + fileType + " failed!"));
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                if (!response.isSuccessful()) {
                    Log.e(Constants.LOG_TAG, "Server error: " + response.code());
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
                        mainHandler.post(() -> outputText.setText("Downloading " + fileType + " (" + progress + "%) ..."));
                    }
                    Log.d(Constants.LOG_TAG, "File downloaded successfully: " + file.getAbsolutePath());
                    mainHandler.post(() -> outputText.setText("Downloaded " + fileType + " successfully"));
                    if(callback != null) {
                        callback.accept(context);
                    }
                    prefs.edit().putBoolean(fileDownloadedPrefName(fileName), true).apply();
                }
            }
        });
    }

    private boolean isFileAlreadyDownloaded(String fileName, String fileUrl) {
        File file = new File(context.getCacheDir(), fileName);
        if(file.exists() && prefs.getBoolean(fileDownloadedPrefName(fileName), false)) {
            String localCRC = getLocalFileCRC32C(file);
            String firebaseCRC = getFirebaseFileCRC32C(fileUrl);
            Log.d(Constants.LOG_TAG, "Firebase CRC: " + firebaseCRC + ", Local CRC: " + localCRC);
            return localCRC == null || localCRC.equals(firebaseCRC);
        }
        return false;
    }

    public String getLocalFileCRC32C(File file) {
        try (FileInputStream fis = new FileInputStream(file);
             FileChannel channel = fis.getChannel()) {
            CRC32C crc32c = null;
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                crc32c = new CRC32C();
            } else {
                return null;
            }
            ByteBuffer buffer = ByteBuffer.allocate(8192);
            while (channel.read(buffer) > 0) {
                buffer.flip();
                crc32c.update(buffer);
                buffer.clear();
            }
            return String.valueOf(crc32c.getValue());
        } catch (Exception ex) {
            Log.e(Constants.LOG_TAG, "Exception during calculating CRC32C for " + file.getPath(), ex);
            return "-1";
        }
    }

    private String getFirebaseFileCRC32C(String fileUrl) {
        try {
            String urlString = fileUrl.split("\\?")[0] + Constants.FIREBASE_METADATA_PARAMS;
            URL url = new URL(urlString);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            int responseCode = connection.getResponseCode();
            if (responseCode == HttpURLConnection.HTTP_OK) {
                Scanner scanner = new Scanner(connection.getInputStream());
                StringBuilder response = new StringBuilder();
                while (scanner.hasNext()) {
                    response.append(scanner.nextLine());
                }
                scanner.close();
                JSONObject jsonResponse = new JSONObject(response.toString());
                return jsonResponse.optString("crc32c", "-1");
            } else {
                Log.e(Constants.LOG_TAG, "Failed to fetch CRC32C from firebase storage for " + fileUrl);
            }
            connection.disconnect();
        } catch (IOException | JSONException ex) {
            Log.e(Constants.LOG_TAG, "Exception during fetching CRC32C from firebase storage for " + fileUrl, ex);
        }
        return "-1";
    }

    private String fileDownloadedPrefName(String fileName) {
        return Constants.PREF_FILE_DOWNLOADED + "::" + fileName;
    }

}
