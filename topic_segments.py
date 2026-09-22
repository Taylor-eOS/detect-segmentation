import os
import pysbd
import numpy as np
from sentence_transformers import SentenceTransformer

input_path = "input.txt"
model_name = "google/embeddinggemma-300m"
language = input("Language (en): ") or "en"
TRUST_CODE = False
COHESION_WINDOW = 6
TARGET_SENTENCES_PER_BREAK = 5
MIN_SENTENCES_BETWEEN_BREAKS = 3
REQUESTED_BREAK_COUNT = None

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

def mean_cohesion(matrix):
    n = matrix.shape[0]
    if n < 2:
        return 1.0
    total = matrix.sum() - np.trace(matrix)
    count = n * n - n
    return total / count if count > 0 else 1.0

def mean_cross_similarity(matrix):
    if matrix.size == 0:
        return 1.0
    return matrix.mean()

def compute_boundary_scores(embeddings, window):
    n = embeddings.shape[0]
    scores = np.zeros(n)
    for i in range(1, n):
        left_low = max(0, i - window)
        right_high = min(n, i + window)
        left_block = embeddings[left_low:i]
        right_block = embeddings[i:right_high]
        if left_block.shape[0] == 0 or right_block.shape[0] == 0:
            continue
        left_left_sim = pairwise_similarity(left_block, left_block)
        right_right_sim = pairwise_similarity(right_block, right_block)
        left_right_sim = pairwise_similarity(left_block, right_block)
        left_cohesion = mean_cohesion(left_left_sim)
        right_cohesion = mean_cohesion(right_right_sim)
        cross_similarity = mean_cross_similarity(left_right_sim)
        side_cohesion = (left_cohesion + right_cohesion) / 2.0
        scores[i] = side_cohesion - cross_similarity
    return scores

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
    scores = compute_boundary_scores(embeddings, COHESION_WINDOW)
    candidates = rank_candidates(scores)
    print_all_candidates(sentences, candidates)
    target_count = select_target_count(sentences)
    chosen = select_spaced_candidates(candidates, target_count, MIN_SENTENCES_BETWEEN_BREAKS)
    print_suggested_breaks(sentences, chosen)
    write_segmented_output(sentences, chosen, "input_segmented.txt")

if __name__ == "__main__":
    main()
