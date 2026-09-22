import os
import re
import pysbd
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer

input_path = "input.txt"
model_name = "google/embeddinggemma-300m"
language = input("Language (en): ") or "en"
TRUST_CODE = False
NEIGHBOR_RADIUS = 4
LOCAL_K = 3
LEXICAL_WINDOW = 6
MIN_WORD_LENGTH = 5
EMBEDDING_WEIGHT = 0.6
LEXICAL_WEIGHT = 0.4
TARGET_SENTENCES_PER_BREAK = 5
MIN_SENTENCES_BETWEEN_BREAKS = 3
REQUESTED_BREAK_COUNT = None

STOPWORDS_BY_LANGUAGE = {
    "en": {
        "about", "after", "again", "against", "almost", "along", "already",
        "although", "always", "among", "another", "anyone", "anything",
        "around", "because", "become", "becomes", "before", "behind",
        "being", "below", "between", "beyond", "cannot", "could", "during",
        "either", "enough", "every", "everyone", "everything", "further",
        "having", "hence", "herself", "himself", "however", "indeed",
        "instead", "itself", "little", "might", "more", "most", "much",
        "myself", "neither", "never", "nobody", "nothing", "often", "once",
        "only", "other", "others", "ourselves", "outside", "over", "perhaps",
        "rather", "really", "same", "several", "shall", "should", "since",
        "some", "someone", "something", "sometimes", "still", "such",
        "than", "that", "their", "theirs", "themselves", "then", "there",
        "therefore", "these", "they", "this", "those", "though", "through",
        "throughout", "thus", "today", "together", "toward", "towards",
        "under", "unless", "until", "upon", "very", "was", "were", "what",
        "whatever", "when", "whenever", "where", "whereas", "wherever",
        "whether", "which", "while", "who", "whoever", "whom", "whose",
        "will", "with", "within", "without", "would", "yourself",
        "yourselves",
    }
}

def load_text():
    with open(input_path, "r", encoding="utf-8") as f:
        return f.read()

def split_sentences(text):
    segmenter = pysbd.Segmenter(language=language, clean=True)
    sentences = [s.strip() for s in segmenter.segment(text) if s.strip()]
    return sentences

def embed_sentences(sentences):
    model = SentenceTransformer(model_name, device="cpu", trust_remote_code=TRUST_CODE)
    embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=True, batch_size=16,)
    return embeddings

def pairwise_similarity(a, b):
    return a @ b.T

def local_cohesion(embeddings, index, side, radius, k):
    n = embeddings.shape[0]
    if side == "left":
        block_start = max(0, index - radius)
        block = embeddings[block_start:index]
        anchor = embeddings[index - 1]
    else:
        block_end = min(n, index + radius)
        block = embeddings[index:block_end]
        anchor = embeddings[index]
    if block.shape[0] == 0:
        return 1.0
    sims = block @ anchor
    sims = np.sort(sims)[::-1]
    top_k = sims[: min(k, sims.shape[0])]
    return float(top_k.mean())

def boundary_adjacent_drop(embeddings, index):
    left_vec = embeddings[index - 1]
    right_vec = embeddings[index]
    return float(left_vec @ right_vec)

def compute_embedding_scores(embeddings, radius, k):
    n = embeddings.shape[0]
    scores = np.zeros(n)
    for i in range(1, n):
        left_cohesion = local_cohesion(embeddings, i, "left", radius, k)
        right_cohesion = local_cohesion(embeddings, i, "right", radius, k)
        adjacent_similarity = boundary_adjacent_drop(embeddings, i)
        side_cohesion = (left_cohesion + right_cohesion) / 2.0
        scores[i] = side_cohesion - adjacent_similarity
    return scores

def tokenize_content_words(sentence):
    words = re.findall(r"[^\W\d_]+", sentence.lower(), flags=re.UNICODE)
    stopwords = STOPWORDS_BY_LANGUAGE.get(language, set())
    content_words = [w for w in words if len(w) >= MIN_WORD_LENGTH and w not in stopwords]
    return content_words

def build_sentence_word_lists(sentences):
    return [tokenize_content_words(s) for s in sentences]

def window_word_counts(word_lists, start, end):
    counts = Counter()
    for i in range(start, end):
        counts.update(word_lists[i])
    return counts

def lexical_overlap_score(left_counts, right_counts):
    if not left_counts or not right_counts:
        return 0.0
    left_words = set(left_counts)
    right_words = set(right_counts)
    shared = left_words & right_words
    union = left_words | right_words
    if not union:
        return 0.0
    jaccard = len(shared) / len(union)
    shared_weight = sum(min(left_counts[w], right_counts[w]) for w in shared)
    total_weight = sum(left_counts.values()) + sum(right_counts.values())
    weighted_overlap = (2.0 * shared_weight / total_weight) if total_weight > 0 else 0.0
    return (jaccard + weighted_overlap) / 2.0

def compute_lexical_scores(word_lists, window):
    n = len(word_lists)
    scores = np.zeros(n)
    for i in range(1, n):
        left_start = max(0, i - window)
        right_end = min(n, i + window)
        left_counts = window_word_counts(word_lists, left_start, i)
        right_counts = window_word_counts(word_lists, i, right_end)
        overlap = lexical_overlap_score(left_counts, right_counts)
        scores[i] = 1.0 - overlap
    return scores

def zscore(values):
    values = np.asarray(values, dtype=float)
    std = values.std()
    if std < 1e-8:
        return np.zeros_like(values)
    return (values - values.mean()) / std

def combine_scores(embedding_scores, lexical_scores, embedding_weight, lexical_weight):
    embedding_z = zscore(embedding_scores)
    lexical_z = zscore(lexical_scores)
    combined = embedding_weight * embedding_z + lexical_weight * lexical_z
    combined[0] = 0.0
    return combined

def find_local_maxima(scores):
    n = len(scores)
    peaks = []
    for i in range(1, n - 1):
        left_ok = scores[i] >= scores[i - 1]
        right_ok = scores[i] >= scores[i + 1]
        strictly_higher = scores[i] > scores[i - 1] or scores[i] > scores[i + 1]
        if left_ok and right_ok and strictly_higher:
            peaks.append(i)
    return peaks

def compute_depth(scores, peak_index):
    n = len(scores)
    left = peak_index
    while left > 0 and scores[left - 1] >= scores[left]:
        left -= 1
    right = peak_index
    while right < n - 1 and scores[right + 1] >= scores[right]:
        right += 1
    left_drop = scores[left] - scores[peak_index]
    right_drop = scores[right] - scores[peak_index]
    return (left_drop + right_drop) / 2.0

def rank_candidates(scores):
    peaks = find_local_maxima(scores)
    candidates = []
    for p in peaks:
        depth = compute_depth(scores, p)
        candidates.append((p, scores[p], depth))
    candidates.sort(key=lambda c: c[1], reverse=True)
    return candidates

def select_spaced_candidates(candidates, count, min_spacing):
    chosen = []
    chosen_positions = []
    for candidate in candidates:
        if len(chosen) >= count:
            break
        position = candidate[0]
        too_close = any(abs(position - p) < min_spacing for p in chosen_positions)
        if too_close:
            continue
        chosen.append(candidate)
        chosen_positions.append(position)
    return chosen

def snippet(sentence, max_chars=90):
    text = sentence.replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."

def print_all_candidates(sentences, candidates):
    print(f"\nALL CANDIDATE BOUNDARIES ({len(candidates)}), ranked by raw score")
    ranked_by_position = sorted(candidates, key=lambda c: c[0])
    score_rank = {c[0]: rank for rank, c in enumerate(candidates, start=1)}
    for index, raw_score, depth in ranked_by_position:
        print(f"boundary {index:>4}  score {raw_score:.4f}  depth {depth:.4f}  score_rank {score_rank[index]:>3}")
        print(f"    ...{snippet(sentences[index - 1])}")
        print(f"    {snippet(sentences[index])}...")

def select_target_count(sentences):
    if REQUESTED_BREAK_COUNT is not None:
        return REQUESTED_BREAK_COUNT
    estimate = max(1, round(len(sentences) / TARGET_SENTENCES_PER_BREAK) - 1)
    return estimate

def print_suggested_breaks(sentences, chosen):
    ordered = sorted(chosen, key=lambda c: c[0])
    print(f"\nSUGGESTED {len(ordered)} BREAKS (highest score, spacing-enforced, in reading order)\n")
    for index, raw_score, depth in ordered:
        print(f"boundary {index:>4}  score {raw_score:.4f}")
        print(f"    ...{snippet(sentences[index - 1])}")
        print(f"    {snippet(sentences[index])}...")

def write_segmented_output(sentences, chosen, output_path):
    boundary_set = set(c[0] for c in chosen)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, sentence in enumerate(sentences):
            if i in boundary_set:
                f.write("\n\n")
            elif i > 0:
                f.write(" ")
            f.write(sentence)
    print(f"\nWrote segmented text to '{output_path}'.")

def main():
    text = load_text()
    sentences = split_sentences(text)
    if len(sentences) < 3:
        print("At least 3 sentences are needed to detect topic shifts.")
        return
    print(f"{len(sentences)} sentences detected.")
    embeddings = embed_sentences(sentences)
    embedding_scores = compute_embedding_scores(embeddings, NEIGHBOR_RADIUS, LOCAL_K)
    word_lists = build_sentence_word_lists(sentences)
    lexical_scores = compute_lexical_scores(word_lists, LEXICAL_WINDOW)
    scores = combine_scores(embedding_scores, lexical_scores, EMBEDDING_WEIGHT, LEXICAL_WEIGHT)
    candidates = rank_candidates(scores)
    print_all_candidates(sentences, candidates)
    target_count = select_target_count(sentences)
    chosen = select_spaced_candidates(candidates, target_count, MIN_SENTENCES_BETWEEN_BREAKS)
    print_suggested_breaks(sentences, chosen)
    write_segmented_output(sentences, chosen, "input_segmented.txt")

if __name__ == "__main__":
    main()
