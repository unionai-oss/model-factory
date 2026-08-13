"""Inference team: serves policy checkpoints for rollouts and evals.

When the training team drops a `policy-checkpoint` artifact, this subsystem
loads the weights into a serving app (`mf-inference`) usable by the eval
team (candidate-vs-base generation) and the training loop (disaggregated
rollouts, dev profile+). Publishes `inference-endpoint` artifacts describing
what is currently being served.
"""
