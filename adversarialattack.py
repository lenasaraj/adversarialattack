import json
import os
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from google.colab.patches import cv2_imshow

# Function to get class indexes from ImageNet
def get_class_idx(label):
    labelPath = "/content/imagenet_class_index.json"
    with open(labelPath) as f:
        imageNetClasses = {labels[1]: int(idx) for (idx, labels) in json.load(f).items()}
    return imageNetClasses.get(label, None)

# Function to preprocess image for ResNet50 model
def preprocess_image_resnet(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    return preprocess_input(image)

# Function to generate targeted adversarial examples
def generate_targeted_adversaries(model, baseImage, classIdx, target, steps=1000, eps=1/255.0, lr=1.5e-2):
    sccLoss = SparseCategoricalCrossentropy()
    optimizer = Adam(learning_rate=lr)
    delta = tf.Variable(tf.zeros_like(baseImage), trainable=True)
    for step in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(delta)
            adversary = preprocess_input(baseImage + delta)
            predictions = model(adversary, training=False)
            originalLoss = -sccLoss(tf.convert_to_tensor([classIdx]), predictions)
            targetLoss = sccLoss(tf.convert_to_tensor([target]), predictions)
            totalLoss = originalLoss + targetLoss
            if step % 20 == 0:
                print(f"step: {step}, loss: {totalLoss.numpy()}...")
            gradients = tape.gradient(totalLoss, delta)
            optimizer.apply_gradients([(gradients, delta)])
            delta.assign_add(tf.clip_by_value(delta, clip_value_min=-eps, clip_value_max=eps))
    return delta

# Function to annotate and display images
def annotate_and_show(image, label, prob, window_name="Image"):
    text = f"{label}: {prob:.2f}%"
    cv2.putText(image, text, (3, 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
    cv2_imshow(image)

# Main process to load, process, and display images
def main():
    args = {
        "input": "/content/goldfish.jpeg",
        "output": "/content/adversarial.jpeg",
        "class_idx": 1,
        "target_class_idx": 2
    }

    # Load and preprocess image
    print("[INFO] loading image...")
    image = cv2.imread(args["input"])
    preprocessedImage = preprocess_image_resnet(image)

    # Load ResNet50 model
    print("[INFO] loading pre-trained ResNet50 model...")
    model = ResNet50(weights="imagenet")

    # Predictions on the original image
    print("[INFO] making predictions...")
    predictions = model.predict(preprocessedImage)
    top_predictions = decode_predictions(predictions, top=3)[0]
    for (i, (imagenetID, label, prob)) in enumerate(top_predictions):
        if i == 0:
            print(f"[INFO] {label} => {get_class_idx(label)}")
        print(f"[INFO] {i + 1}. {label}: {prob * 100:.2f}%")
    annotate_and_show(image, top_predictions[0][1], top_predictions[0][2] * 100)

    # Generate adversarial example
    baseImage = tf.constant(preprocessedImage, dtype=tf.float32)
    print("[INFO] generating perturbation...")
    deltaUpdated = generate_targeted_adversaries(model, baseImage, args["class_idx"], args["target_class_idx"])

    # Save and display adversarial example
    print("[INFO] creating targeted adversarial example...")
    adverImage = (baseImage + deltaUpdated).numpy().squeeze()
    adverImage = np.clip(adverImage, 0, 255).astype("uint8")
    adverImage = cv2.cvtColor(adverImage, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args["output"], adverImage)

    # Inference on adversarial example
    print("[INFO] running inference on the adversarial example...")
    adverPredictions = model.predict(preprocess_input(baseImage + deltaUpdated))
    top_adver_predictions = decode_predictions(adverPredictions,
