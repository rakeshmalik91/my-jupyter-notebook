package com.rakeshmalik.insectid;

import java.util.Map;

public class Constants {

    public static final int MAX_PREDICTIONS = 10;
    public static final Map<ModelType, Float> MIN_ACCEPTED_LOGIT = Map.of(
            ModelType.BUTTERFLY, -50.0f,
            ModelType.MOTH, -50.0f
    );
    public static final Map<ModelType, Float> MIN_ACCEPTED_SOFTMAX = Map.of(
            ModelType.BUTTERFLY, 0.1f,
            ModelType.MOTH, 0.1f
    );
    public static final String MODEL_FILE_NAME_FMT = "m.checkpoint.%s.pt";
    public static final String CLASSES_FILE_NAME_FMT = "classes.%s.json";
    public static final String CLASS_DETAILS_FILE_NAME_FMT = "class_details.%s.json";

    public static final String FIREBASE_FILE_PARAMS = "?alt=media&token=f07a4656-2c48-424f-ba04-18136437bae3";
    public static final String FIREBASE_METADATA_PARAMS = "?alt=json";
    public static final String FIREBASE_BASE_URL_FMT = "https://firebasestorage.googleapis.com/v0/b/telebirding-49623.appspot.com/o/models%%2F";
    public static final String MODEL_URL_FMT = FIREBASE_BASE_URL_FMT + "m.checkpoint.%s.pt" + FIREBASE_FILE_PARAMS;
    public static final String CLASSES_URL_FMT = FIREBASE_BASE_URL_FMT + "classes.%s.json" + FIREBASE_FILE_PARAMS;
    public static final String CLASS_DETAILS_URL_FMT = FIREBASE_BASE_URL_FMT + "class_details.%s.json" + FIREBASE_FILE_PARAMS;

    public static final String NAME = "name";
    public static final String LOG_TAG = "insect-id";
    public static final String PREF = "insect-id";
    public static final String PREF_FILE_DOWNLOADED = "insect-id::file-downloaded";

}
