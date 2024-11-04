from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from PyMultiDictionary import MultiDictionary

# Initialize SBERT model and RoBERTa MNLI model
paraphrase_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
tokenizer = AutoTokenizer.from_pretrained('roberta-large-mnli')
roberta_model = AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli')

# Initialize PyMultiDictionary for synonyms and antonyms checking
dictionary = MultiDictionary()

def get_word_count(text):
    """Return the number of words in the text."""
    return len(text.split())

def compare_with_paraphrase_model(text1, text2, threshold=0.7):
    """Compare texts using the paraphrase model and apply synonym/antonym adjustments."""
    def get_text_embeddings(text, model):
        sentences = text.split('\n')
        sentences = [s.strip() for s in sentences if s.strip()]
        embeddings = model.encode(sentences, convert_to_tensor=True)
        return embeddings, sentences

    def compare_texts(text1, text2, model):
        text1_embeddings, text1_sentences = get_text_embeddings(text1, model)
        text2_embeddings, text2_sentences = get_text_embeddings(text2, model)
        cosine_scores = util.pytorch_cos_sim(text1_embeddings, text2_embeddings)
        avg_score = torch.mean(torch.max(cosine_scores, dim=1)[0]).item()
        return avg_score, cosine_scores, text1_sentences, text2_sentences

    def find_matching_sentences(cosine_matrix, text1_sentences, text2_sentences, threshold):
        matching_sentences_text1 = []
        matching_sentences_text2 = []
        for i in range(len(text1_sentences)):
            for j in range(len(text2_sentences)):
                if cosine_matrix[i][j] > threshold:
                    matching_sentences_text1.append(text1_sentences[i])
                    matching_sentences_text2.append(text2_sentences[j])
        return matching_sentences_text1, matching_sentences_text2

    similarity_score, cosine_matrix, text1_sentences, text2_sentences = compare_texts(text1, text2, paraphrase_model)
    matching_sentences_text1, matching_sentences_text2 = find_matching_sentences(cosine_matrix, text1_sentences, text2_sentences, threshold)
    
    # Synonym and antonym adjustment logic (same as in your paraphrase code)
    penalty, reward = 0, 0
    critical_antonym_penalty = 0.2
    synonym_reward = 0.1

    for sent1, sent2 in zip(matching_sentences_text1, matching_sentences_text2):
        words1 = sent1.split()
        words2 = sent2.split()
        for word1, word2 in zip(words1, words2):
            if check_for_antonyms(word1, word2):
                penalty += critical_antonym_penalty
            if check_for_synonyms(word1, word2):
                reward += synonym_reward

    adjusted_similarity = max(0, similarity_score - penalty + reward)
    return adjusted_similarity

def compare_with_roberta_model(text1, text2):
    """Compare texts using the RoBERTa model and handle negation, antonyms, and synonyms."""
    def classify_relationship(text1, text2):
        inputs = tokenizer(text1, text2, return_tensors='pt', truncation=True, padding=True)
        outputs = roberta_model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
        classes = ['contradiction', 'neutral', 'entailment']
        predicted_class = classes[torch.argmax(logits)]
        return predicted_class, probabilities

    predicted_class, probabilities = classify_relationship(text1, text2)

    penalty, reward = 0, 0
    critical_antonym_penalty = 0.3
    synonym_reward = 0.1

    words1 = text1.split()
    words2 = text2.split()
    negation_words = ["not", "no", "never", "none", "n't"]

    negation_found1 = any(word in negation_words for word in words1)
    negation_found2 = any(word in negation_words for word in words2)

    if negation_found1 != negation_found2:
        penalty += 0.5

    for word1, word2 in zip(words1, words2):
        if check_for_antonyms(word1, word2):
            penalty += critical_antonym_penalty
        if check_for_synonyms(word1, word2):
            reward += synonym_reward

    model_similarity_score = probabilities[2]
    adjusted_similarity = max(0, model_similarity_score - penalty + reward)
    return adjusted_similarity

def check_for_synonyms(word1, word2):
    synonyms_word1 = dictionary.synonym("en", word1) or []
    synonyms_word2 = dictionary.synonym("en", word2) or []
    return word1 in synonyms_word2 or word2 in synonyms_word1

def check_for_antonyms(word1, word2):
    antonyms_word1 = dictionary.antonym("en", word1) or []
    antonyms_word2 = dictionary.antonym("en", word2) or []
    return word1 in antonyms_word2 or word2 in antonyms_word1

def compare_text_inputs(text1, text2):
    """Main function to compare texts using different models based on word count."""
    word_count1 = get_word_count(text1)
    word_count2 = get_word_count(text2)
    
    if word_count1 <= 20 and word_count2 <= 20:
        print("Using RoBERTa MNLI model for comparison...")
        similarity_score = compare_with_roberta_model(text1, text2)
    else:
        print("Using Paraphrase SBERT model for comparison...")
        similarity_score = compare_with_paraphrase_model(text1, text2)
    
    print(f"Adjusted Similarity Score: {similarity_score}")
    return similarity_score

# Example usage
text1 = "The gentle breeze swept through the trees as the sun rose, casting a soft, golden glow over the landscape."
text2 = "bathing the peaceful landscape in golden light."

compare_text_inputs(text1, text2)

