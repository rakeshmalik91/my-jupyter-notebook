package com.rakeshmalik.insectid;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import com.yalantis.ucrop.UCrop;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Formatter;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

import org.pytorch.Tensor;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.torchvision.TensorImageUtils;

public class MainActivity extends AppCompatActivity {

    private ImageView imageView;
    private TextView outputText;
    private Uri photoUri;
    private ExecutorService executorService;
    private Spinner modelTypeSpinner;
    private ModelType selectedModelType;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button buttonPickImage = findViewById(R.id.buttonPickImage);
        imageView = findViewById(R.id.imageView);

        buttonPickImage.setOnClickListener(v -> showImagePickerDialog());

        executorService = Executors.newSingleThreadExecutor();

        outputText = findViewById(R.id.outputText);

        modelTypeSpinner = findViewById(R.id.modelTypeSpinner);
        createModelTypeSpinner();
    }

    private final void createModelTypeSpinner() {
        // Convert enum values to a string array for the Spinner
        ModelType[] modelTypes = ModelType.values();
        String[] modelTypeNames = new String[modelTypes.length];
        for (int i = 0; i < modelTypes.length; i++) {
            modelTypeNames[i] = modelTypes[i].displayName;
        }

        // Create and set adapter for the Spinner
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, modelTypeNames);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        modelTypeSpinner.setAdapter(adapter);

        // Set listener for Spinner item selection
        modelTypeSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                selectedModelType = modelTypes[position];
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                selectedModelType = null;
            }
        });
    }

    // Launcher for picking an image from the gallery
    private final ActivityResultLauncher<Intent> galleryLauncher =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    photoUri = result.getData().getData();
                    if (photoUri != null) {
                        launchImageCrop();
                    }
                }
            });

    // Launcher for taking a photo with the camera
    private final ActivityResultLauncher<Intent> cameraLauncher =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
                if (result.getResultCode() == RESULT_OK && photoUri != null) {
                    launchImageCrop();
                } else {
                    Toast.makeText(this, "Failed to capture image", Toast.LENGTH_SHORT).show();
                }
            });

    // Request camera permission
    private final ActivityResultLauncher<String> cameraPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
                if (isGranted) {
                    openCamera();
                } else {
                    Toast.makeText(this, "Camera permission is required", Toast.LENGTH_SHORT).show();
                }
            });

    // Show dialog to choose Gallery or Camera
    private void showImagePickerDialog() {
        String[] options = {"Gallery", "Camera"};
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Select an Option");
        builder.setItems(options, (dialog, which) -> {
            if (which == 0) {
                openGallery();
            } else {
                checkCameraPermission();
            }
        });
        builder.show();
    }

    // Open the Gallery
    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        galleryLauncher.launch(intent);
    }

    // Check and request Camera permission
    private void checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            openCamera();
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    // Open the Camera
    private void openCamera() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (intent.resolveActivity(getPackageManager()) != null) {
            Log.d("CameraTest", "Camera app found!");
            // Stores the photo file
            File photoFile = createImageFile();
            if (photoFile != null) {
                photoUri = FileProvider.getUriForFile(this, getApplicationContext().getPackageName() + ".provider", photoFile);
                intent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri);
                cameraLauncher.launch(intent);
            } else {
                Toast.makeText(this, "Error creating file", Toast.LENGTH_SHORT).show();
            }
        } else {
            Log.d("CameraTest", "Camera app not found!");
            Toast.makeText(this, "Camera app not found", Toast.LENGTH_SHORT).show();
        }
    }

    // Create a temporary file to store the captured image
    private File createImageFile() {
        try {
            String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
            File storageDir = getExternalFilesDir(null);
            return File.createTempFile("IMG_" + timeStamp, ".jpg", storageDir);
        } catch (IOException ex) {
            Log.e("CreateImageFile", "Exception during image creation", ex);
            return null;
        }
    }

    private void launchImageCrop() {
        imageView.setImageURI(photoUri);
        File croppedFile = new File(getCacheDir(), "cropped.jpg");
        if (croppedFile.exists()) {
            croppedFile.delete();
        }
        Uri croppedUri = Uri.fromFile(new File(getCacheDir(), "cropped.jpg"));
        UCrop.of(photoUri, croppedUri)
                .withAspectRatio(1, 1)
                .withMaxResultSize(300, 300)
                .start(this);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == UCrop.REQUEST_CROP && resultCode == RESULT_OK) {
            photoUri = UCrop.getOutput(data);
            if (photoUri != null) {
                imageView.setImageURI(photoUri);
                executorService.submit(new PredictRunnable());
            }
        }
    }

    public static Integer[] getTopKIndices(float[] array, int k) {
        Integer[] indices = new Integer[array.length];
        for (int i = 0; i < array.length; i++) {
            indices[i] = i;
        }
        Arrays.sort(indices, (i1, i2) -> Double.compare(array[i2], array[i1]));
        return Arrays.copyOfRange(indices, 0, k);
    }

    private String predict() {
        String modelName = String.format(Constants.MODEL_FILE_NAME_FORMAT, selectedModelType.modelName);
        String classListName = String.format(Constants.CLASSES_FILE_NAME_FORMAT, selectedModelType.modelName);

        try {
            String modelPath = ModelLoader.assetFilePath(this, modelName);
            Module model = Module.load(modelPath);

            Log.d("Prediction", "Loading photo: " + photoUri);
            Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), photoUri);
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
            Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap,
                    new float[]{0.485f, 0.456f, 0.406f},
                    new float[]{0.229f, 0.224f, 0.225f});

            Tensor outputTensor = model.forward(IValue.from(inputTensor)).toTensor();
            float[] logitScores = outputTensor.getDataAsFloatArray();
            Log.d("Prediction", "scores: " + Arrays.toString(logitScores));
            float[] softMaxScores = toSoftMax(logitScores.clone());
            Log.d("Prediction", "softMaxScores: " + Arrays.toString(softMaxScores));

            int k = Constants.MAX_PREDICTIONS;
            Integer[] predictedClass = getTopKIndices(softMaxScores, k);
            Log.d("Prediction", "Top " + k + " scores: " + Arrays.stream(predictedClass).map(c -> logitScores[c]).collect(Collectors.toList()));
            Log.d("Prediction", "Top " + k + " softMaxScores: " + Arrays.stream(predictedClass).map(c -> softMaxScores[c]).collect(Collectors.toList()));

            List<String> classLabels = ModelLoader.loadClassLabels(this, classListName);
            List<String> predictions = Arrays.stream(predictedClass)
                    .filter(c -> softMaxScores[c] > Constants.MIN_ACCEPTED_SOFTMAX)
                    .filter(c -> logitScores[c] > Constants.MIN_ACCEPTED_LOGIT)
                    .map(c -> classLabels.get(c) + " - " + String.format("%.2f", softMaxScores[c] * 100) + "%")
                    .collect(Collectors.toList());
            Log.d("Prediction", "Predicted class: " + predictions);
            return predictions.isEmpty() ? getString(R.string.no_match_found) : String.join("\n", predictions);
        } catch(Exception ex) {
            Log.e("Predict", "Exception during prediction", ex);
            Toast.makeText(this, "Exception during prediction", Toast.LENGTH_SHORT).show();
        }
        return null;
    }

    private float[] toSoftMax(float[] scores) {
        float sumExp = 0.0f;
        for (float score : scores) {
            sumExp += (float) Math.exp(score);
        }
        for (int i = 0; i < scores.length; i++) {
            scores[i] = (float) Math.exp(scores[i]) / sumExp;
        }
        return scores;
    }

    class PredictRunnable implements Runnable {
        @Override
        public void run() {
            outputText.setText(R.string.predicting);
            String predictions = predict();
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    outputText.setText(predictions);
                }
            });
        }
    }

}
