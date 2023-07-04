# Similarity Tools

Similarity Tools is a collection of classes for matching and classifying text using fuzzy matching and Sentence-BERT (SBERT) embeddings. It provides utilities for fuzzy matching, SBERT matching, and a hybrid approach that combines both methods.

## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

# Similarity Tools

Similarity Tools is a collection of classes for text matching and classification using fuzzy matching and Sentence-BERT (SBERT) embeddings. It provides utilities for fuzzy matching, SBERT matching, and a hybrid approach that combines both methods.

## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Similarity Tools is designed to help with text matching and classification tasks. It offers the following classes:

- `FuzzyMatcher`: A class for performing fuzzy matching based on the Levenshtein distance.
- `SBertMatching`: A class for matching text using SBERT embeddings.
- `Classifier`: A class that combines fuzzy matching and SBERT matching for text classification.
- `SearchEngine`: A class that performs text search using both FuzzyMatcher and SBertMatching.

## Usage

To use the Similarity Tools package, follow these steps:

1. Install the package and its dependencies (see the [Installation](#installation) section).
2. Import the necessary classes from the package into your code.
3. Create an instance of the desired class (`FuzzyMatcher`, `SBertMatching`, `Classifier`, or `SearchEngine`).
4. Use the methods provided by the class to perform text matching, classification, or search.

Here are some examples of how to use the classes:

### Example 1: Using the Classifier class

```python
from Classifier import Classifier

# Create an instance of the Classifier class
classifier = Classifier(method='hybrid', data_file='Data/base_data.csv', text_col='Text', target_col='Disposition',db_path='./Stores/disposition_classifier', store_path='./Stores/disposition_base.csv')

# Classify a query string
query = "This is a sample query."
prediction = classifier.predict(query)

print("Predicted label:", prediction)
```
### Example 1: Using the SearchEngine
```python
from SearchEngine import SearchEngine

# Create an instance of the SearchEngine class
search_engine = SearchEngine(db_path='./Stores/search_db',store_path='./Stores/search_base.csv')

# Search for similar strings
query = "Similarity search query."
list_of_strings = ["First string", "Second string", "Third string"]
results = search_engine.search(query, list_of_strings, num_item=5)

print("Matching results:")
for result in results:
    print(result)

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
