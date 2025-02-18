package com.rakeshmalik.insectid;

import java.util.Map;

public class Constants {

    public static final int MAX_PREDICTIONS = 10;
    public static final Map<ModelType, Float> MIN_ACCEPTED_LOGIT = Map.of(
            ModelType.BUTTERFLY, -16.0f,
            ModelType.MOTH, -50.0f                                                                  // TODO change when model is trained
    );
    public static final Map<ModelType, Float> MIN_ACCEPTED_SOFTMAX = Map.of(
            ModelType.BUTTERFLY, 0.01f,
            ModelType.MOTH, 0.02f
    );
    public static final String MODEL_FILE_NAME_FORMAT = "m.checkpoint.%s.pt";
    public static final String CLASSES_FILE_NAME_FORMAT = "classes.%s.json";
    public static final String CLASS_DETAILS_FILE_NAME_FORMAT = "class_details.%s.json";

    public static final String NAME = "name";

    public static final String LOG_TAG = "insect-id";

}
