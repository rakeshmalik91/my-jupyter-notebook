package com.rakeshmalik.insectid;

import android.content.Context;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileInputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import android.util.Log;

import com.google.gson.Gson;

public class ModelLoader {

    public static String loadModel(Context context, String fileName) {
        File file = new File(context.getCacheDir(), fileName);
        try(InputStream is = new FileInputStream(file)) {
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
        } catch (IOException ex) {
            Log.d(Constants.LOG_TAG, "Exception loading model", ex);
            throw new RuntimeException(ex);
        }
    }

    private static final Map<String, List<String>> classLabelsCache = new HashMap<>();

    public static List<String> loadClassLabels(Context context, String fileName) {
        if(classLabelsCache.containsKey(fileName)) {
            return classLabelsCache.get(fileName);
        }
        List<String> classLabels = null;
        File file = new File(context.getCacheDir(), fileName);
        try(InputStream is = new FileInputStream(file)) {
            byte[] buffer = new byte[is.available()];
            is.read(buffer);
            is.close();
            String json = new String(buffer, StandardCharsets.UTF_8);
            Gson gson = new Gson();
            classLabels = gson.fromJson(json, List.class);
        } catch (IOException ex) {
            Log.d(Constants.LOG_TAG, "Exception loading class labels", ex);
            throw new RuntimeException(ex);
        }
        classLabelsCache.put(fileName, classLabels);
        return classLabels;
    }

    private static final Map<String, Map<String, Map<String, String>>> classDetailsCache = new HashMap<>();

    public static Map<String, Map<String, String>> loadClassDetails(Context context, String fileName) {
        if(classDetailsCache.containsKey(fileName)) {
            return classDetailsCache.get(fileName);
        }
        Map<String, Map<String, String>> classDetails = Map.of();
        File file = new File(context.getCacheDir(), fileName);
        try(InputStream is = new FileInputStream(file)) {
            byte[] buffer = new byte[is.available()];
            is.read(buffer);
            is.close();
            String json = new String(buffer, StandardCharsets.UTF_8);
            Gson gson = new Gson();
            classDetails = gson.fromJson(json, Map.class);
        } catch (IOException ex) {
            Log.d(Constants.LOG_TAG, "Exception loading class details", ex);
            return classDetails;
        }
        classDetailsCache.put(fileName, classDetails);
        return classDetails;
    }
}
