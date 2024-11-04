# Text Similarity Comparison with Synonym & Antonym Adjustment

This project compares two pieces of text to measure their similarity, accounting for synonyms and antonyms. The code uses two models: a Sentence-BERT (SBERT) model for paraphrase detection and a RoBERTa model for textual entailment. Additionally, it uses the `PyMultiDictionary` library to adjust similarity scores based on synonyms and antonyms.

## Features

- **SBERT Paraphrase Model**: Measures similarity based on paraphrased text.
- **RoBERTa MNLI Model**: Classifies text pairs as entailment, contradiction, or neutral.
- **Synonym & Antonym Adjustment**: Adjusts similarity scores for matching synonyms and penalizes for antonyms.
- **Negation Handling**: Identifies negation in text and adjusts scores accordingly.

## Installation

1. **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2. **Install the required packages**:
    ```bash
    pip install sentence-transformers transformers torch PyMultiDictionary
    ```

## Usage

To run the code, input two text strings, and it will output an adjusted similarity score.

```python
text1 = "The gentle breeze swept through the trees as the sun rose, casting a soft, golden glow over the landscape."
text2 = "Bathing the peaceful landscape in golden light."

compare_text_inputs(text1, text2)
