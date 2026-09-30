import re
import pysbd
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer

input_path = "input.txt"
output_path = "output_segments.txt"
model_name = "google/embeddinggemma-300m"
language = input("Language (en): ") or "en"
TRUST_CODE = False
target_words = int(input("Target words (1000): ") or 1000)
min_words = int(target_words - (target_words / 3))
max_words = int(target_words + (target_words / 3))
print(f"Range: {min_words}-{max_words}")
TARGET_PULL = 10.0
UNDER_MIN_PENALTY = 100.0
NEIGHBOR_RADIUS = 4
LOCAL_K = 3
LEXICAL_WINDOW = 6
MIN_WORD_LENGTH = 5
EMBEDDING_WEIGHT = 0.6
LEXICAL_WEIGHT = 0.4
model = None

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

def read_source_text():
    with open(input_path, "r", encoding="utf-8") as f:
        return f.read()

def split_into_chapters(text):
    return [line for line in text.splitlines() if line.strip()]

def split_sentences(text):
    segmenter = pysbd.Segmenter(language=language, clean=False)
    sentences = []
    starts = []
    offset = 0
    for line in text.splitlines(keepends=True):
        cursor = 0
        if line.strip():
            for sentence in segmenter.segment(line):
                stripped = sentence.strip()
                if not stripped:
                    continue
                position = line.find(stripped, cursor)
                if position == -1:
                    position = cursor
                sentences.append(stripped)
                starts.append(offset + position)
                cursor = position + len(stripped)
        offset += len(line)
    return sentences, starts

def get_model():
    global model
    if model is None:
        model = SentenceTransformer(model_name, device="cpu", trust_remote_code=TRUST_CODE)
    return model

def embed_sentences(sentences):
    return get_model().encode(sentences, normalize_embeddings=True, show_progress_bar=False, batch_size=16)

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

def compute_boundary_scores(sentences):
    embeddings = embed_sentences(sentences)
    embedding_scores = compute_embedding_scores(embeddings, NEIGHBOR_RADIUS, LOCAL_K)
    word_lists = build_sentence_word_lists(sentences)
    lexical_scores = compute_lexical_scores(word_lists, LEXICAL_WINDOW)
    return combine_scores(embedding_scores, lexical_scores, EMBEDDING_WEIGHT, LEXICAL_WEIGHT)

def size_penalty(words):
    deviation = (words - target_words) / target_words
    penalty = TARGET_PULL * deviation ** 2
    if words < min_words:
        penalty += UNDER_MIN_PENALTY * (min_words - words) / target_words
    return penalty

def choose_cuts(scores, word_counts):
    n = len(word_counts)
    cumulative = [0]
    for count in word_counts:
        cumulative.append(cumulative[-1] + count)
    best = [float("inf")] * (n + 1)
    previous = [0] * (n + 1)
    best[0] = 0.0
    for j in range(1, n + 1):
        i = j - 1
        while i >= 0:
            words = cumulative[j] - cumulative[i]
            if i < j - 1 and words > max_words:
                break
            reward = scores[i] if i > 0 else 0.0
            cost = best[i] + size_penalty(words) - reward
            if cost < best[j]:
                best[j] = cost
                previous[j] = i
            i -= 1
    cuts = []
    position = n
    while position > 0:
        position = previous[position]
        if position > 0:
            cuts.append(position)
    return sorted(cuts)

def cut_text(chapter, starts, cuts):
    boundaries = [0] + [starts[c] for c in cuts] + [len(chapter)]
    return [chapter[boundaries[k]:boundaries[k + 1]] for k in range(len(boundaries) - 1)]

def split_chapter_into_parts(chapter):
    total_words = len(chapter.split())
    if total_words <= max_words:
        return [chapter]
    sentences, starts = split_sentences(chapter)
    if len(sentences) < 2:
        return [chapter]
    scores = compute_boundary_scores(sentences)
    word_counts = [len(s.split()) for s in sentences]
    cuts = choose_cuts(scores, word_counts)
    parts = cut_text(chapter, starts, cuts)
    print("Block of {} words, {} sentences, split into {} parts".format(total_words, len(sentences), len(parts)))
    return parts

def build_segments(chapters):
    segments = []
    for chapter in chapters:
        segments.extend(split_chapter_into_parts(chapter))
    return segments

def write_segments(segments):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(segment.strip() for segment in segments))

def print_size_report(segments):
    sizes = [len(segment.split()) for segment in segments]
    print("Wrote {} segments to {}".format(len(segments), output_path))
    print("Words per segment: min {}, mean {:.0f}, max {}".format(min(sizes), sum(sizes) / len(sizes), max(sizes)))
    print("Segments above {} words: {}".format(max_words, sum(1 for s in sizes if s > max_words)))

def main():
    text = read_source_text()
    chapters = split_into_chapters(text)
    segments = build_segments(chapters)
    if not segments:
        print("No text found in " + input_path)
        return
    write_segments(segments)
    print_size_report(segments)

if __name__ == "__main__":
    print("This segments on blank lines.")
    main()
