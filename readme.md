# Similarity Tools

Similarity Tools is a collection of classes for matching and classifying text using fuzzy matching and Sentence-BERT (SBERT) embeddings. It provides utilities for fuzzy matching, SBERT matching, and a hybrid approach that combines both methods.

## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Similarity Tools is designed to facilitate text matching and classification tasks. It offers the following classes:

- `FuzzyMatcher`: A class for performing fuzzy matching based on the Levenshtein distance.
- `SBertMatching`: A class for matching text using SBERT embeddings.
- `Classifier`: A class that combines fuzzy matching and SBERT matching for text classification.

## Usage

To use the Similarity Tools package, follow these steps:

1. Install the package and its dependencies (see the [Installation](#installation) section).
2. Import the necessary classes from the package into your code.
3. Create an instance of the `Classifier` class, specifying the matching method and data file.
4. Call the `predict` method of the `Classifier` class to classify new text.

Here's an example of how to use the `Classifier` class:

```python
from Classifier import Classifier

# Create an instance of the Classifier class
classifier = Classifier(method='hybrid', data_file='Data/base_data.csv', text_col='Text', target_col='Disposition')

# Classify a query string
query = "i dont like"
prediction = classifier.predict(query)

print("Predicted label:", prediction) // NOT_INTRESTED
```


## Dependencies

The Similarity Tools package has the following dependencies:

    fuzzywuzzy
    sentence-transformers
    pandas
    faiss
    numpy

These dependencies will be installed automatically when you install the package using `pip`.


## Credit
Mohammad Ali Abbas (Sr. ML Engineer IdrakAi)
