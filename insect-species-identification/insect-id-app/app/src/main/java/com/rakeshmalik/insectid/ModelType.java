package com.rakeshmalik.insectid;

public enum ModelType {
    BUTTERFLY("Butterfly", "2025.02.16.butterfly"),
    MOTH("Moth", "2025.02.16.moth");

    public final String displayName;
    public final String modelName;

    ModelType(String displayName, String modelName) {
        this.displayName = displayName;
        this.modelName = modelName;
    }
}
