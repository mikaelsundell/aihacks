aihacks
==================

[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg?style=flat-square)](LICENSE)

Introduction
------------

This repository contains small, self-contained C/C++ programs that explore how basic machine learning and AI techniques work under the hood.

The focus is on simple, readable examples rather than frameworks or full-featured systems.
Everything is implemented directly in C, so the math and data flow are visible and explicit.

The intent is to make it easier to reason about what common AI building blocks actually do, without needing to learn a large library or toolchain first.

---

Animals example
---------------

The first example is a minimal **animal classification and generation** program written in C.

It covers:

- Feature vectors and labels
- Softmax-based classification
- Cross-entropy loss
- Stochastic gradient descent
- Learned weights and bias
- A simple contrast between **classification** and **generation**

Animals like `cat`, `dog`, `fox`, and `cow` are represented as small 3-dimensional feature vectors.
The model is trained on noisy samples around these reference points, roughly simulating variation you might get from real-world measurements.

After training, the program:

- Classifies known and mixed inputs
- Prints class probabilities
- Shows a basic generative idea:
  *“give me a cat”* → produce a plausible feature vector

The generative part is intentionally small and illustrative.
It’s meant as a conceptual step toward understanding how larger generative models work, not as a complete solution.

---

What this example is (and is not)
---------------------------------

**This example is:**
- Small and self-contained
- Easy to step through in a debugger
- Focused on core ideas
- Free of external dependencies

**This example is not:**
- A machine learning framework
- A deep neural network
- Meant for production use
- Trying to mimic human intelligence

It’s simply a teaching and exploration tool.

---

Build and run
-------------

The animals example is a single C source file.

Compile and run it with a standard compiler:

```bash
mdkir build_cmake
cmake .. && make
./animals
