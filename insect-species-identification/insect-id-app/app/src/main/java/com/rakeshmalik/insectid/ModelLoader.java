package com.rakeshmalik.insectid;

import android.content.Context;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileInputStream;
import java.util.List;

import android.content.res.AssetFileDescriptor;

import com.google.gson.Gson;

public class ModelLoader {

    // Function to load model file from assets folder
    // Function to load model file from assets folder
    public static String assetFilePath(Context context, String assetName) {
        try {
            // Open the model file from the assets folder
            InputStream is = context.getAssets().open(assetName);

            // Create a temporary file to store the model
            File tempFile = File.createTempFile("model", "tmp", context.getCacheDir());
            tempFile.deleteOnExit();

            // Write the input stream data to the temporary file
            FileOutputStream outputStream = new FileOutputStream(tempFile);
            byte[] buffer = new byte[4 * 1024];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }

            outputStream.close();
            return tempFile.getAbsolutePath(); // Return the path to the temporary file
        } catch (IOException e) {
            throw new RuntimeException(e); // Handle error
        }
    }

    public static List<String> loadClassLabels(Context context, String assetName) {
        List<String> classLabels = null;
        try {
            // Open the class_labels.json file from the assets folder
            InputStream is = context.getAssets().open(assetName);
            byte[] buffer = new byte[is.available()];
            is.read(buffer);
            is.close();

            // Convert the byte array to a string
            String json = new String(buffer, StandardCharsets.UTF_8);

            // Use Gson to parse the JSON into a List<String>
            Gson gson = new Gson();
            classLabels = gson.fromJson(json, List.class);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return classLabels;
    }
}
