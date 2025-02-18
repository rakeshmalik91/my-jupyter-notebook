package com.rakeshmalik.insectid;

import static com.rakeshmalik.insectid.Constants.*;

import android.content.Context;
import android.graphics.Bitmap;
import android.net.Uri;
import android.provider.MediaStore;
import android.util.Log;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Collectors;

public class PredictionManager {

    public static String predict(Context context, ModelType modelType, Uri photoUri) {
        String modelName = String.format(Constants.MODEL_FILE_NAME_FORMAT, modelType.modelName);
        String classListName = String.format(Constants.CLASSES_FILE_NAME_FORMAT, modelType.modelName);
        String classDetailsName = String.format(Constants.CLASS_DETAILS_FILE_NAME_FORMAT, modelType.modelName);

        try {
            String modelPath = ModelLoader.assetFilePath(context, modelName);
            Module model = Module.load(modelPath);
            List<String> classLabels = ModelLoader.loadClassLabels(context, classListName);
            Map<String, Map<String, String>> classDetails = ModelLoader.loadClassDetails(context, classDetailsName);

            Log.d("predict", "Loading photo: " + photoUri);
            Bitmap bitmap = MediaStore.Images.Media.getBitmap(context.getContentResolver(), photoUri);
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
            Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap,
                    new float[]{0.485f, 0.456f, 0.406f},
                    new float[]{0.229f, 0.224f, 0.225f});

            Tensor outputTensor = model.forward(IValue.from(inputTensor)).toTensor();
            float[] logitScores = outputTensor.getDataAsFloatArray();
            Log.d("predict", "scores: " + Arrays.toString(logitScores));
            float[] softMaxScores = toSoftMax(logitScores.clone());
            Log.d("predict", "softMaxScores: " + Arrays.toString(softMaxScores));

            int k = MAX_PREDICTIONS;
            Integer[] predictedClass = getTopKIndices(softMaxScores, k);
            Log.d("predict", "Top " + k + " scores: " + Arrays.stream(predictedClass).map(c -> logitScores[c]).collect(Collectors.toList()));
            Log.d("predict", "Top " + k + " softMaxScores: " + Arrays.stream(predictedClass).map(c -> softMaxScores[c]).collect(Collectors.toList()));

            List<String> predictions = Arrays.stream(predictedClass)
                    .filter(c -> softMaxScores[c] > MIN_ACCEPTED_SOFTMAX.get(modelType))
                    .filter(c -> logitScores[c] > MIN_ACCEPTED_LOGIT.get(modelType))
                    .map(c -> {
                        String className = classLabels.get(c);
                        String value = "<font color='#FF7755'><i>" + className + "</i></font><br>";
                        if(classDetails.containsKey(className) && classDetails.get(className).containsKey(NAME)) {
                            value += "<font color='#FFFFFF'>" + classDetails.get(className).get(NAME) + "</font><br>";
                        }
                        value += String.format(Locale.getDefault(), "<font color='#777777'>~%.2f%% match</font><br><br>", softMaxScores[c] * 100);
                        return value;
                    })
                    .collect(Collectors.toList());
            Log.d("predict", "Predicted class: " + predictions);
            return predictions.isEmpty() ? context.getString(R.string.no_match_found) : String.join("\n", predictions);
        } catch(Exception ex) {
            Log.e("predict", "Exception during prediction", ex);
        }
        return "Failed to predict!!!";
    }

    private static float[] toSoftMax(float[] scores) {
        float sumExp = 0.0f;
        for (float score : scores) {
            sumExp += (float) Math.exp(score);
        }
        for (int i = 0; i < scores.length; i++) {
            scores[i] = (float) Math.exp(scores[i]) / sumExp;
        }
        return scores;
    }

    public static Integer[] getTopKIndices(float[] array, int k) {
        Integer[] indices = new Integer[array.length];
        for (int i = 0; i < array.length; i++) {
            indices[i] = i;
        }
        Arrays.sort(indices, (i1, i2) -> Double.compare(array[i2], array[i1]));
        return Arrays.copyOfRange(indices, 0, k);
    }

}
