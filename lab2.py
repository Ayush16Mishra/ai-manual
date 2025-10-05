"""
lab2_astar_plagiarism.py

A* based sentence alignment for plagiarism detection.
- Implements sentence normalization + naive sentence splitter
- Levenshtein distance (edit distance)
- A* search over alignment states (i, j)
- Heuristic: for each remaining sentence in doc1, use its min Levenshtein
  distance to any remaining sentence in doc2 (admissible lower bound).
- Transitions: Align (i,j), Skip i (penalty = len(tokens_i)), Skip j (penalty = len(tokens_j))

Outputs an alignment list of (i, j, cost) where i or j may be -1 indicating a skip.
"""

from heapq import heappush, heappop
import re
import time


# -----------------------
# Utilities: text normalization and sentence splitting
# -----------------------
def normalize_sentence(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)   # remove punctuation -> space
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def split_into_sentences(text: str) -> list:
    # very simple sentence splitter: split on . ? ! and newlines
    raw = re.split(r'[.!?\n]+', text)
    # filter empty and normalize
    sentences = [normalize_sentence(r) for r in raw if r.strip() != ""]
    return sentences

# -----------------------
# Levenshtein distance (classic dynamic programming)
# -----------------------
def levenshtein(a: str, b: str) -> int:
    # operate over tokens (words), not characters — better for sentence alignment
    ta = a.split()
    tb = b.split()
    n, m = len(ta), len(tb)
    if n == 0: return m
    if m == 0: return n
    # DP row wise (space optimized)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = ta[i - 1]
        for j in range(1, m + 1):
            if ai == tb[j - 1]:
                cost = 0
            else:
                cost = 1
            cur[j] = min(prev[j] + 1,        # deletion
                         cur[j - 1] + 1,     # insertion
                         prev[j - 1] + cost) # substitution
        prev = cur
    return prev[m]

# -----------------------
# Heuristic function for A*
# admissible: for each remaining sentence in doc1 (i..end),
# compute its minimal Levenshtein to any remaining in doc2 (j..end).
# Sum those minima — this underestimates the true remaining cost.
# -----------------------
def heuristic_min_pairing(doc1, doc2, i, j, cache):
    # cache is dict for pairs (idx1, idx2) -> lev
    key = (i, j)
    # compute sum over doc1[i:]
    total = 0
    for p in range(i, len(doc1)):
        best = None
        for q in range(j, len(doc2)):
            k = (p, q)
            if k in cache:
                d = cache[k]
            else:
                d = levenshtein(doc1[p], doc2[q])
                cache[k] = d
            if best is None or d < best:
                best = d
                if best == 0:
                    break
        # if doc2 exhausted, skipping doc1 will cost at least len(tokens)
        if best is None:
            best = max(1, len(doc1[p].split()))
        total += best
    return total

# -----------------------
# A* search for sentence alignment
# State: (i, j) indices into doc1 and doc2
# Goal: i == len(doc1) and j == len(doc2)
# Actions:
#  - Align (i, j) -> cost = levenshtein(doc1[i], doc2[j])
#  - Skip i     -> cost = skip_cost(doc1[i])
#  - Skip j     -> cost = skip_cost(doc2[j])
# skip_cost chosen as number of tokens in the sentence (reasonable penalty)
# -----------------------
def astar_align(doc1, doc2, verbose=False):
    n1 = len(doc1)
    n2 = len(doc2)
    start = (0, 0)
    # priority queue entries: (f_score, g_score, (i,j), came_from_key, action_info)
    # came_from_key used to reconstruct path: map state -> (prev_state, action)
    open_heap = []
    came_from = {}
    gscore = {start: 0}
    cache_lev = {}  # cache lev distances
    h0 = heuristic_min_pairing(doc1, doc2, 0, 0, cache_lev)
    heappush(open_heap, (h0, 0, start))

    visited_expansions = 0
    while open_heap:
        f, g, (i, j) = heappop(open_heap)
        visited_expansions += 1
        if (i, j) == (n1, n2):
            # reconstruct path
            path = []
            cur = (n1, n2)
            while cur != start:
                prev, action = came_from[cur]
                path.append((cur, action))
                cur = prev
            path.reverse()
            if verbose:
                print(f"Expanded states: {visited_expansions}")
            return path, visited_expansions, gscore[(n1, n2)]
        # generate neighbors
        # 1) align if possible
        if i < n1 and j < n2:
            k = (i, j)
            if k in cache_lev:
                cost = cache_lev[k]
            else:
                cost = levenshtein(doc1[i], doc2[j])
                cache_lev[k] = cost
            neighbor = (i + 1, j + 1)
            tentative_g = g + cost
            if tentative_g < gscore.get(neighbor, float('inf')):
                gscore[neighbor] = tentative_g
                h = heuristic_min_pairing(doc1, doc2, neighbor[0], neighbor[1], cache_lev)
                heappush(open_heap, (tentative_g + h, tentative_g, neighbor))
                came_from[neighbor] = ((i, j), ('align', i, j, cost))
        # 2) skip i (advance in doc1)
        if i < n1:
            skip_cost = max(1, len(doc1[i].split()))
            neighbor = (i + 1, j)
            tentative_g = g + skip_cost
            if tentative_g < gscore.get(neighbor, float('inf')):
                gscore[neighbor] = tentative_g
                h = heuristic_min_pairing(doc1, doc2, neighbor[0], neighbor[1], cache_lev)
                heappush(open_heap, (tentative_g + h, tentative_g, neighbor))
                came_from[neighbor] = ((i, j), ('skip1', i, skip_cost))
        # 3) skip j (advance in doc2)
        if j < n2:
            skip_cost = max(1, len(doc2[j].split()))
            neighbor = (i, j + 1)
            tentative_g = g + skip_cost
            if tentative_g < gscore.get(neighbor, float('inf')):
                gscore[neighbor] = tentative_g
                h = heuristic_min_pairing(doc1, doc2, neighbor[0], neighbor[1], cache_lev)
                heappush(open_heap, (tentative_g + h, tentative_g, neighbor))
                came_from[neighbor] = ((i, j), ('skip2', j, skip_cost))

    # no alignment found (shouldn't usually happen)
    return None, visited_expansions, float('inf')

# -----------------------
# Helper to pretty print alignment
# -----------------------
def format_alignment(path, doc1, doc2):
    rows = []
    # path is list of (state, action) where state is post-action
    # reconstruct starting at (0,0)
    cur_i, cur_j = 0, 0
    for (state, action) in path:
        act_type = action[0]
        if act_type == 'align':
            _, i, j, cost = action
            rows.append((i, j, doc1[i], doc2[j], cost))
            cur_i, cur_j = i + 1, j + 1
        elif act_type == 'skip1':
            _, i, cost = action
            rows.append((i, -1, doc1[i], None, cost))
            cur_i = i + 1
        elif act_type == 'skip2':
            _, j, cost = action
            rows.append((-1, j, None, doc2[j], cost))
            cur_j = j + 1
    return rows

# -----------------------
# Test harness / example cases
# -----------------------
def run_test(docA_text, docB_text, label="Test"):
    docA = split_into_sentences(docA_text)
    docB = split_into_sentences(docB_text)
    print(f"\n=== {label} ===")
    print("DocA sentences:", len(docA))
    for idx,s in enumerate(docA):
        print(f"  A[{idx}]: {s}")
    print("DocB sentences:", len(docB))
    for idx,s in enumerate(docB):
        print(f"  B[{idx}]: {s}")

    path, expanded, total_cost = astar_align(docA, docB, verbose=False)
    if path is None:
        print("No alignment found.")
        return
    alignment = format_alignment(path, docA, docB)
    print(f"\nAlignment (i, j, A_sentence, B_sentence, cost):")
    for row in alignment:
        print(row)
    print(f"Total alignment cost: {total_cost}")
    print(f"States expanded during A*: {expanded}")
    # mark likely plagiarized pairs (low cost threshold)
    print("\nPossible plagiarized pairs (cost <= 2):")
    for (i,j,a,b,c) in alignment:
        if i>=0 and j>=0 and c <= 2:
            print(f"  A[{i}] <--> B[{j}] cost={c}")
    return alignment

if __name__ == "__main__":
    # Test case 1: identical documents
    doc1 = "This is a test document. It has several sentences. This is the third sentence."
    doc2 = "This is a test document. It has several sentences. This is the third sentence."
    run_test(doc1, doc2, "Identical Documents")

    # Test case 2: slight modifications (synonyms, reorder)
    doc1 = "Data science is an interdisciplinary field. It uses statistics and computer science. It extracts knowledge from data."
    doc2 = "Data-science is interdisciplinary. It leverages statistics and computing. From data, we extract knowledge."
    run_test(doc1, doc2, "Slightly Modified Document")

    # Test case 3: completely different documents
    doc1 = "Rocks and minerals are interesting. They form the Earth's crust."
    doc2 = "Cooking requires heat. Ingredients must be fresh for good taste."
    run_test(doc1, doc2, "Completely Different Documents")

    # Test case 4: partial overlap
    doc1 = "Machine learning models require data. Supervised learning uses labels. Unsupervised learning finds structure."
    doc2 = "Supervised learning uses labels. Reinforcement learning is about agents and rewards."
    run_test(doc1, doc2, "Partial Overlap")
