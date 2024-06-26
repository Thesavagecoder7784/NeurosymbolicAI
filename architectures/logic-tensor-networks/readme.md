# Logical Tensor Networks (LTNs) for Prisoner's Dilemma Simulation
This repository contains code that simulates the Prisoner's Dilemma using Logical Tensor Networks (LTNs), implemented in TensorFlow. The Prisoner's Dilemma is a classic example in game theory where rational individuals making decisions in their self-interest may lead to a suboptimal outcome for all.

# Overview
The code uses LTNs to model how multiple prisoners interact and make decisions based on their unique characteristics (embeddings). Predicates such as Cooperate, Betray, and Payoff are defined within the LTN framework to simulate prisoner decisions and outcomes.

# Usage
## Constants and Grounding:
Embedding vectors are assigned to each prisoner ('a' to 'h'), representing their unique characteristics.

## Predicates:
Cooperate, Betray, and Payoff predicates are defined using MLPs to model decision-making and outcomes based on embeddings.

## Data:
Lists cooperate, betray, payoff_high, and payoff_low define which prisoners cooperate, betray, receive high payoffs, and low payoffs, respectively.

## Connectives and Quantifiers:
Logical connectives (Not, And, Or, Implies) and quantifiers (Forall, Exists) express relationships and constraints among prisoner actions and outcomes.

## Theory Axioms:
A set of axioms define logical constraints and rules governing prisoner behavior, decision-making, and payoff outcomes.

## Training:
The model is trained using an optimizer to minimize loss based on the satisfaction level (sat_level) of defined axioms.

## Results:
Visualizations (plt_heatmap, scatter plots) show facts and predictions of prisoner behavior (cooperate/betray) and payoff outcomes.
