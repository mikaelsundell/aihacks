// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025 - present Mikael Sundell
// https://github.com/mikaelsundell/aihacks

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// AI hacks animals vocabulary (used by this example)
//
// Sample:
//   One observed data item from the real world.
//   Example: one photo, one sound clip, one sensor reading.
//
// Feature / Feature vector:
//   A numeric representation of a sample.
//   In this example: a small vector of length NUM_FEATURES.
//   In real systems: can be very large.
//
// Sensor / Input tensor:
//   The structured collection of feature values produced by a sensor.
//   Example (image):
//     width x height x channels
//     e.g. 224 x 224 x 3 (RGB image)
//   Example (audio):
//     time x frequency
//   This example flattens the sensor output into a 1D feature vector.
//
// Class / Label:
//   A discrete category describing what the sample is.
//   Example: "cat", "dog", "car", "person".
//
// Model:
//   A mathematical function with learnable parameters that maps
//   feature vectors → class probabilities.
//
// Training:
//   The process of adjusting model parameters so that the model’s
//   output matches known labels for known samples.
//
// Inference:
//   Using a trained model to predict labels for new, unseen samples.

// how many categories (labels) the model can choose from (e.g., cat/dog/fox/cow)
#define NUM_CLASSES 4

// how many measured input values describe each sample (the “features” from real-world capture, e.g., pixels/sensor values/etc.)
#define NUM_FEATURES 3

typedef struct {
    double x[NUM_FEATURES];
    int label; // 0..NUM_CLASSES-1
} Sample;

static const char* class_name(int k) {
    static const char* names[NUM_CLASSES] = {"cat", "dog", "fox", "cow"};
    return names[k];
}

// simple deterministic RNG for repeatability
// The RNG is only used to introduce small, repeatable randomness so the training data and initial weights
//  are slightly different instead of identical.
// - deterministic = The same input always produces the same output, same results for every run
// - RNG = pseudo-random number generator
// - numerical Recipes
// - multiplier = 1664525
// - increment  = 1013904223
static unsigned int rng_state = 1;
static double rand01(void) {
    rng_state = 1664525u * rng_state + 1013904223u;
    return (rng_state / 4294967296.0); // 0,1, 2^32 max value
}

// uniform noise in [-amp, +amp]
static double noise(double amp) {
    return (rand01() * 2.0 - 1.0) * amp;
}

// softmax with numerical stability
// softmax converts scores into probabilities:
// - all values are between 0 and 1
// - all values sum to 1
// - bigger scores → bigger probabilities
// Numerical stability - Making sure calculations behave sensibly on a computer,
// even when numbers get very large or very small.
// Computers do not work with real numbers like math on paper:
// - they have limited precision
// - they can overflow
// - they can underflow
static void softmax(const double* z, double* p, int n) {
    double maxz = z[0];
    for (int i = 1; i < n; i++) if (z[i] > maxz) maxz = z[i];

    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        p[i] = exp(z[i] - maxz);
        sum += p[i];
    }
    for (int i = 0; i < n; i++) p[i] /= sum;
}

// probabilities
// normalized way to express how strongly the model associates the input numbers 
// with each possible class, and they are used both to train the model and to 
// choose the final prediction.
static int argmax(const double* a, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (a[i] > a[best]) best = i;
    return best;
}

// Generative example (conceptual)
//
// Instead of asking:
//   "What class is this input?"
//
// We ask:
//   "Give me an input that looks like this class"
//
// This samples around the same prototype distribution used
// to create the training data.
//
// This is NOT a full generative model.
// It is an educational inversion of classification.

static void generate_sample_for_class(int class_id, double out_x[NUM_FEATURES]) {
    const double proto[NUM_CLASSES][NUM_FEATURES] = {
        { 0.9,  0.1,  0.2}, // cat
        { 0.1,  0.9,  0.2}, // dog
        { 0.2,  0.2,  0.9}, // fox
        { 0.8,  0.8,  0.1}  // cow
    };

    const double gen_noise = 0.10;

    for (int f = 0; f < NUM_FEATURES; f++) {
        out_x[f] = proto[class_id][f] + noise(gen_noise);
    }
}

int main(void) {
    // prototypes: 3D points for each animal.
    // These are “feature space” positions (not pixels). You can change them 
    // freely.
    const double proto[NUM_CLASSES][NUM_FEATURES] = {
        { 0.9,  0.1,  0.2}, // cat
        { 0.1,  0.9,  0.2}, // dog
        { 0.2,  0.2,  0.9}, // fox
        { 0.8,  0.8,  0.1}  // cow
    };

    // Expose the conceptual world the model operates in
    // These are the original class definitions in feature space
    printf("--- Class prototypes (feature space) ---\n");
    for (int k = 0; k < NUM_CLASSES; k++) {
        printf("%s : [", class_name(k));
        for (int f = 0; f < NUM_FEATURES; f++) {
            printf("%.2f", proto[k][f]);
            if (f + 1 < NUM_FEATURES) printf(" ");
        }
        printf("]\n");
    }
    printf("\n");

    // This code is explicitly constructing a dataset whose samples are associated
    // with known prototype points, with small variations added.
    const int samples_per_class = 80;
    const int N = NUM_CLASSES * samples_per_class;
    Sample* data = (Sample*)calloc((size_t)N, sizeof(Sample));
    if (!data) {
        fprintf(stderr, "Out of memory.\n");
        return 1;
    }

    // one observation from a sensor, converted into numbers, with a known label.
    const double noise_amp = 0.15;
    int idx = 0;
    for (int k = 0; k < NUM_CLASSES; k++) {
        for (int i = 0; i < samples_per_class; i++) {
            for (int f = 0; f < NUM_FEATURES; f++) {
                data[idx].x[f] = proto[k][f] + noise(noise_amp);
            }
            data[idx].label = k;
            idx++;
        }
    }

    // model parameters: W (classes x features) and b (classes)
    // this code defines the parameters the model is allowed to learn,
    // everything else in the program exists to adjust these numbers.
    double W[NUM_CLASSES][NUM_FEATURES];
    double b[NUM_CLASSES];

    // initialize small random weights
    for (int k = 0; k < NUM_CLASSES; k++) {
        b[k] = 0.0;
        for (int f = 0; f < NUM_FEATURES; f++) {
            W[k][f] = (rand01() * 2.0 - 1.0) * 0.05;
        }
    }

    printf("--- Training ---\n");

    // One epoch = one full pass over all training samples.
    // For each epoch:
    // - The model sees every sample
    // - Makes predictions
    // - Measures errors
    // - Adjusts weights slightly
    // Over time:
    // - Errors decrease
    // - Predictions stabilize
    const int epochs = 2000;

    // controls how big a step the model takes when it updates its weights after seeing an error.
    const double lr = 0.2;

    // training loop: softmax regression with cross-entropy loss
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;

        // stochastic gradient descent (one sample at a time)
        for (int i = 0; i < N; i++) {
            double z[NUM_CLASSES];
            for (int k = 0; k < NUM_CLASSES; k++) {
                double s = b[k];
                for (int f = 0; f < NUM_FEATURES; f++) {
                    s += W[k][f] * data[i].x[f];
                }
                z[k] = s;
            }

            double p[NUM_CLASSES];
            softmax(z, p, NUM_CLASSES);

            int y = data[i].label;
            double py = p[y] < 1e-15 ? 1e-15 : p[y];
            total_loss += -log(py);

            double g[NUM_CLASSES];
            for (int k = 0; k < NUM_CLASSES; k++) {
                g[k] = p[k] - (k == y ? 1.0 : 0.0);
            }

            for (int k = 0; k < NUM_CLASSES; k++) {
                b[k] -= lr * g[k];
                for (int f = 0; f < NUM_FEATURES; f++) {
                    W[k][f] -= lr * g[k] * data[i].x[f];
                }
            }
        }

        if (epoch % 200 == 0) {
            printf("Epoch %4d | avg loss %.4f\n", epoch, total_loss / (double)N);
        }
    }

    // Test on prototypes and a few mixed points
    double tests[][NUM_FEATURES] = {
        {0.9, 0.1, 0.2}, // cat proto
        {0.1, 0.9, 0.2}, // dog proto
        {0.2, 0.2, 0.9}, // fox proto
        {0.8, 0.8, 0.1}, // cow proto
        {0.6, 0.4, 0.2}, // between cat/cow-ish
        {0.15,0.65,0.25},// dog-ish
        {0.25,0.25,0.70} // fox-ish
    };

    printf("\n--- Tests ---\n");
    for (size_t ti = 0; ti < sizeof(tests)/sizeof(tests[0]); ti++) {
        double z[NUM_CLASSES];
        for (int k = 0; k < NUM_CLASSES; k++) {
            double s = b[k];
            for (int f = 0; f < NUM_FEATURES; f++) s += W[k][f] * tests[ti][f];
            z[k] = s;
        }
        double p[NUM_CLASSES];
        softmax(z, p, NUM_CLASSES);

        int pred = argmax(p, NUM_CLASSES);

        printf("x = [%.2f %.2f %.2f] -> predicted: %s | probs:",
               tests[ti][0], tests[ti][1], tests[ti][2], class_name(pred));
        for (int k = 0; k < NUM_CLASSES; k++) {
            printf(" %s=%.2f", class_name(k), p[k]);
        }
        printf("\n");
    }

    // Generative demonstration: "give me a cat"
    printf("\n--- Generative example ---\n");
    for (int k = 0; k < NUM_CLASSES; k++) {
        double x[NUM_FEATURES];
        generate_sample_for_class(k, x);

        printf("Generated %s x = [%.2f %.2f %.2f]\n",
               class_name(k), x[0], x[1], x[2]);
    }

    free(data);
    return 0;
}
