package com.rakeshmalik.insectid;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.widget.ProgressBar;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import okhttp3.*;

public class ModelDownloadManager {

    private final OkHttpClient client = new OkHttpClient();
    private final Context context;
    private final ProgressBar progressBar;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    public ModelDownloadManager(Context context, ProgressBar progressBar) {
        this.context = context;
        this.progressBar = progressBar;
    }

    public boolean isFileAlreadyDownloaded(String fileName) {
        File file = new File(context.getCacheDir(), fileName);
        return file.exists();
    }

    public void downloadFile(ModelType modelType) {
        String fileUrl = modelType.modelUrl;
        String fileName = String.format(Constants.MODEL_FILE_NAME_FORMAT, modelType.modelName);
        if (isFileAlreadyDownloaded(fileName)) {
            Log.d(Constants.LOG_TAG, "File already downloaded.");
            return;
        }

        Request request = new Request.Builder().url(fileUrl).build();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                Log.e(Constants.LOG_TAG, "Download failed: " + e.getMessage());
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                if (!response.isSuccessful()) {
                    Log.e(Constants.LOG_TAG, "Server error: " + response.code());
                    return;
                }

                // Get the cache directory
                File cacheDir = context.getCacheDir();
                File file = new File(cacheDir, fileName);

                // Write the file
                InputStream inputStream = response.body().byteStream();
                FileOutputStream outputStream = new FileOutputStream(file);
                byte[] buffer = new byte[4096];
                long totalBytes = response.body().contentLength();
                long downloadedBytes = 0;
                int bytesRead;

                // Update progress in UI
                mainHandler.post(() -> progressBar.setMax((int) totalBytes));

                while ((bytesRead = inputStream.read(buffer)) != -1) {
                    outputStream.write(buffer, 0, bytesRead);
                    downloadedBytes += bytesRead;
                    int progress = (int) ((downloadedBytes * 100) / totalBytes);
                    mainHandler.post(() -> progressBar.setProgress(progress));
                }

                outputStream.close();
                inputStream.close();

                Log.d(Constants.LOG_TAG, "File downloaded successfully: " + file.getAbsolutePath());
                mainHandler.post(() -> progressBar.setProgress(100)); // Ensure it's set to 100
            }
        });
    }
}
