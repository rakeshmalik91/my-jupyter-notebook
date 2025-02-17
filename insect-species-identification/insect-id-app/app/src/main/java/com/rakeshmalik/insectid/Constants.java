package com.rakeshmalik.insectid;

public class Constants {

    public static final int MAX_PREDICTIONS = 10;
    public static final float MIN_ACCEPTED_LOGIT = -15.0f;
    public static final float MIN_ACCEPTED_SOFTMAX = 0.01f;
    public static final String MODEL_FILE_NAME_FORMAT = "m.checkpoint.%s.pt";
    public static final String CLASSES_FILE_NAME_FORMAT = "classes.%s.json";

}
