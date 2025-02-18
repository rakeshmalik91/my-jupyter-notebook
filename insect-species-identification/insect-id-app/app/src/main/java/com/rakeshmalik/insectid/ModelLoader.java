package com.rakeshmalik.insectid;

import android.content.Context;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileInputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import android.content.res.AssetFileDescriptor;
import android.util.Log;

import com.google.gson.Gson;

public class ModelLoader {

    public static String assetFilePath(Context context, String assetName) {
        try {
            InputStream is = context.getAssets().open(assetName);
            File tempFile = File.createTempFile("model", "tmp", context.getCacheDir());
            tempFile.deleteOnExit();
            FileOutputStream outputStream = new FileOutputStream(tempFile);
            byte[] buffer = new byte[4 * 1024];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
            outputStream.close();
            return tempFile.getAbsolutePath();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static final Map<String, List<String>> classLabelsCache = new HashMap<>();

    public static List<String> loadClassLabels(Context context, String assetName) {
        if(classLabelsCache.containsKey(assetName)) {
            return classLabelsCache.get(assetName);
        }
        List<String> classLabels = null;
        try {
            InputStream is = context.getAssets().open(assetName);
            byte[] buffer = new byte[is.available()];
            is.read(buffer);
            is.close();
            String json = new String(buffer, StandardCharsets.UTF_8);
            Gson gson = new Gson();
            classLabels = gson.fromJson(json, List.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        classLabelsCache.put(assetName, classLabels);
        return classLabels;
    }

    private static final Map<String, Map<String, Map<String, String>>> classDetailsCache = new HashMap<>();

    public static Map<String, Map<String, String>> loadClassDetails(Context context, String assetName) {
        if(classDetailsCache.containsKey(assetName)) {
            return classDetailsCache.get(assetName);
        }
        Map<String, Map<String, String>> classDetails = Map.of();
        try {
            InputStream is = context.getAssets().open(assetName);
            byte[] buffer = new byte[is.available()];
            is.read(buffer);
            is.close();
            String json = new String(buffer, StandardCharsets.UTF_8);
            Gson gson = new Gson();
            classDetails = gson.fromJson(json, Map.class);
        } catch (IOException ex) {
            Log.d("loadClassDetails", "Exception loading class", ex);
            return classDetails;
        }
        classDetailsCache.put(assetName, classDetails);
        return classDetails;
    }
}
