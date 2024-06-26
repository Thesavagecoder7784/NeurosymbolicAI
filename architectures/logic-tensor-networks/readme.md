# Logical Tensor Networks (LTNs) for Prisoner's Dilemma Simulation
This repository contains code that simulates the Prisoner's Dilemma using Logical Tensor Networks (LTNs), implemented in TensorFlow. The Prisoner's Dilemma is a classic example in game theory where rational individuals making decisions in their self-interest may lead to a suboptimal outcome for all.

# Overview
The code uses LTNs to model how multiple prisoners interact and make decisions based on their unique characteristics (embeddings). Predicates such as Cooperate, Betray, and Payoff are defined within the LTN framework to simulate prisoner decisions and outcomes.

# Usage
## Constants and Grounding:
Embedding vectors are assigned to each prisoner ('a' to 'h'), representing their unique characteristics.
```python
embedding_size = 5
prisoners = {p: ltn.Constant(np.random.uniform(low=0.0, high=1.0, size=embedding_size), trainable=True) for p in 'abcdefgh'}
```

## Predicates:
Cooperate, Betray, and Payoff predicates are defined using MLPs to model decision-making and outcomes based on embeddings.
```python
Cooperate = ltn.Predicate.MLP([embedding_size], hidden_layer_sizes=(8, 8))
Betray = ltn.Predicate.MLP([embedding_size], hidden_layer_sizes=(8, 8))
Payoff = ltn.Predicate.MLP([embedding_size], hidden_layer_sizes=(8, 8))
```

## Data:
Lists cooperate, betray, payoff_high, and payoff_low define which prisoners cooperate, betray, receive high payoffs, and low payoffs, respectively.
```python
cooperate = ['a', 'b', 'e', 'h']
betray = ['c', 'd', 'f', 'g']
payoff_high = ['a', 'c']
payoff_low = ['b', 'd', 'e', 'f', 'g', 'h']
```

## Connectives and Quantifiers:
Logical connectives (Not, And, Or, Implies) and quantifiers (Forall, Exists) express relationships and constraints among prisoner actions and outcomes.
```python
Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=2), semantics="forall")
Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(p=6), semantics="exists")
formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError())
```

## Theory Axioms:
A set of axioms define logical constraints and rules governing prisoner behavior, decision-making, and payoff outcomes.
```python
@tf.function
def axioms(p_exists):
    p = ltn.Variable.from_constants("p", list(prisoners.values()))
    q = ltn.Variable.from_constants("q", list(prisoners.values()))
    axioms = []

    # Cooperation and Betrayal knowledge
    # Asserts that all prisoners in cooperate list are cooperating
    axioms.append(formula_aggregator([Cooperate(prisoners[p]) for p in cooperate]))
    # Asserts that all prisoners in betray list are betraying
    axioms.append(formula_aggregator([Betray(prisoners[p]) for p in betray]))
    # Asserts that prisoners not in cooperate list are not cooperating
    axioms.append(formula_aggregator([Not(Cooperate(prisoners[p])) for p in prisoners if p not in cooperate]))
    # Asserts that prisoners not in betry list are not betraying
    axioms.append(formula_aggregator([Not(Betray(prisoners[p])) for p in prisoners if p not in betray]))

    # Payoff knowledge
    # Asserts that prisoner receives high payoff
    axioms.append(formula_aggregator([Payoff(prisoners[p]) for p in payoff_high]))
    # Asserts that prisoner receives low payoff
    axioms.append(formula_aggregator([Not(Payoff(prisoners[p])) for p in payoff_low]))

    # Logic of the dilemma
    # Asserts that prisoner who cooperates does not betray
    axioms.append(Forall(p, Implies(Cooperate(p), Not(Betray(p)))))
    # Asserts that prisoner who betrays does not cooperate
    axioms.append(Forall(p, Implies(Betray(p), Not(Cooperate(p)))))
    # Asserts that prisoner who cooperates will get payoff
    axioms.append(Forall(p, Implies(Cooperate(p), Payoff(p))))
    # Asserts that prisoner who does not cooperate will not get payoff
    axioms.append(Forall(p, Implies(Not(Cooperate(p)), Not(Payoff(p)))))
    # Everyone chooses an action
    axioms.append(Forall(p, Or(Cooperate(p), Betray(p))))
```

## Training:
The model is trained using an optimizer to minimize loss based on the satisfaction level (sat_level) of defined axioms.

## Results:
Visualizations (plt_heatmap, scatter plots) show facts and predictions of prisoner behavior (cooperate/betray) and payoff outcomes.
