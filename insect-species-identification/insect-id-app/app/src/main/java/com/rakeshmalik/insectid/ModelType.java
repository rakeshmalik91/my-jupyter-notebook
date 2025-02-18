package com.rakeshmalik.insectid;

public enum ModelType {
    BUTTERFLY("Butterfly", "butterfly"),
    MOTH("Moth", "moth");

    public final String displayName;
    public final String modelName;

    ModelType(String displayName, String modelName) {
        this.displayName = displayName;
        this.modelName = modelName;
    }
}
