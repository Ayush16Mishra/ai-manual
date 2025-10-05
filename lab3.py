import random
import itertools
from copy import deepcopy

# -----------------------------
# K-SAT generator
# -----------------------------
def generate_k_sat(k, m, n):
    # n variables, variables are 1..n
    # each clause: pick k distinct variables, randomly negate
    clauses = []
    for _ in range(m):
        vars = random.sample(range(1, n + 1), k)
        clause = []
        for v in vars:
            if random.random() < 0.5:
                clause.append(v)
            else:
                clause.append(-v)  # negation
        clauses.append(tuple(clause))
    return clauses

# Evaluate clause satisfaction
def clause_satisfied(clause, assignment):
    for lit in clause:
        var = abs(lit)
        val = assignment[var]
        if lit > 0 and val == True:
            return True
        if lit < 0 and val == False:
            return True
    return False

def evaluate_formula(clauses, assignment):
    return sum(clause_satisfied(c, assignment) for c in clauses)

# two heuristics
def heuristic1(clauses, assignment):
    # number of satisfied clauses
    return evaluate_formula(clauses, assignment)

def heuristic2(clauses, assignment):
    # weighted sum: satisfied clauses + fraction of partially satisfied
    total = 0
    for c in clauses:
        if clause_satisfied(c, assignment):
            total += 2
        else:
            # check how many literals in c are true
            count_true = sum(((lit > 0 and assignment[abs(lit)]) or 
                             (lit < 0 and not assignment[abs(lit)])) for lit in c)
            total += count_true/len(c)
    return total

# Generate random assignment
def random_assignment(n):
    return {i: random.choice([True, False]) for i in range(1, n + 1)}

# Flip a variable
def flip_variable(assignment, var):
    new_assign = assignment.copy()
    new_assign[var] = not new_assign[var]
    return new_assign

# Neighborhood functions for VND
def neighborhood1(assignment, n):
    # flip one variable
    for v in range(1, n+1):
        yield flip_variable(assignment, v)

def neighborhood2(assignment, n):
    # flip two variables at once
    for v1 in range(1, n+1):
        for v2 in range(v1+1, n+1):
            new_a = assignment.copy()
            new_a[v1] = not new_a[v1]
            new_a[v2] = not new_a[v2]
            yield new_a

def neighborhood3(assignment, n):
    # flip three variables at once randomly
    vars = list(range(1, n+1))
    random.shuffle(vars)
    for triple in itertools.combinations(vars,3):
        new_a = assignment.copy()
        for v in triple:
            new_a[v] = not new_a[v]
        yield new_a

# Hill Climbing
def hill_climbing(clauses, n, heuristic):
    current = random_assignment(n)
    current_score = heuristic(clauses, current)
    improved = True
    steps = 0
    while improved and steps < 1000:
        improved = False
        for v in range(1, n+1):
            neighbor = flip_variable(current, v)
            score = heuristic(clauses, neighbor)
            if score > current_score:
                current, current_score = neighbor, score
                improved = True
                break
        steps += 1
    return current_score, current, steps

# Beam Search
def beam_search(clauses, n, heuristic, width=3):
    beam = [random_assignment(n) for _ in range(width)]
    scores = [heuristic(clauses,b) for b in beam]
    best_score = max(scores)
    best_assign = beam[scores.index(best_score)]
    steps = 0
    for _ in range(500):
        # generate all neighbors by flipping one variable
        candidates = []
        for b in beam:
            for v in range(1, n+1):
                neighbor = flip_variable(b, v)
                candidates.append(neighbor)
        # pick best width
        candidates.sort(key=lambda x: heuristic(clauses,x), reverse=True)
        beam = candidates[:width]
        scores = [heuristic(clauses,b) for b in beam]
        current_best_score = scores[0]
        if current_best_score > best_score:
            best_score = current_best_score
            best_assign = beam[0]
        steps += 1
    return best_score, best_assign, steps

# Variable-Neighborhood-Descent (VND)
def vnd(clauses, n, heuristic):
    neighborhoods = [neighborhood1, neighborhood2, neighborhood3]
    current = random_assignment(n)
    current_score = heuristic(clauses, current)
    improved = True
    steps = 0
    while improved and steps < 1000:
        improved = False
        for neigh_func in neighborhoods:
            for neighbor in neigh_func(current, n):
                score = heuristic(clauses, neighbor)
                if score > current_score:
                    current, current_score = neighbor, score
                    improved = True
                    break
            if improved:
                break
        steps += 1
    return current_score, current, steps

# Experiment driver
def run_experiment(k=3, m=20, n=10, runs=5):
    clauses = generate_k_sat(k, m, n)
    print(f"\nGenerated {k}-SAT with n={n}, m={m}")
    print("Sample clauses:", clauses[:5], "...")  # show first 5 for brevity
    
    for hname, hfunc in [("Heuristic1", heuristic1), ("Heuristic2", heuristic2)]:
        print(f"\n=== Using {hname} ===")
        
        # Store results across runs
        hc_scores, hc_steps = [], []
        bs3_scores, bs3_steps = [], []
        bs4_scores, bs4_steps = [], []
        vnd_scores, vnd_steps = [], []

        for _ in range(runs):
            # Hill-Climbing
            score, _, steps = hill_climbing(clauses, n, hfunc)
            hc_scores.append(score)
            hc_steps.append(steps)

            # Beam Search width 3
            score, _, steps = beam_search(clauses, n, hfunc, width=3)
            bs3_scores.append(score)
            bs3_steps.append(steps)

            # Beam Search width 4
            score, _, steps = beam_search(clauses, n, hfunc, width=4)
            bs4_scores.append(score)
            bs4_steps.append(steps)

            # VND
            score, _, steps = vnd(clauses, n, hfunc)
            vnd_scores.append(score)
            vnd_steps.append(steps)

        # Print averages and best
        print(f"Hill-Climbing: best={max(hc_scores)}, avg_score={sum(hc_scores)/runs:.2f}, avg_steps={sum(hc_steps)/runs:.1f}")
        print(f"BeamSearch w=3: best={max(bs3_scores)}, avg_score={sum(bs3_scores)/runs:.2f}, avg_steps={sum(bs3_steps)/runs:.1f}")
        print(f"BeamSearch w=4: best={max(bs4_scores)}, avg_score={sum(bs4_scores)/runs:.2f}, avg_steps={sum(bs4_steps)/runs:.1f}")
        print(f"VND: best={max(vnd_scores)}, avg_score={sum(vnd_scores)/runs:.2f}, avg_steps={sum(vnd_steps)/runs:.1f}")
 
if __name__=="__main__":
    # example experiment
    run_experiment(k=3,m=20,n=10,runs=3)
