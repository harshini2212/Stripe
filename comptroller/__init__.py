"""Comptroller — agentic AI + financial-correctness evaluation for Stripe spend.

Comptroller is an internal-style platform that (1) runs agentic AI workflows over
Stripe Issuing card and Stripe Treasury data — expense triage, dispute resolution, fraud
investigation, treasury analysis — and (2) rigorously evaluates every agent output
for *financial correctness* using deterministic checks, ML models, and LLM judges.

The package is intentionally backend-agnostic: it runs end-to-end with no network
access on a deterministic heuristic/ML backend, and lights up live Claude models
(Opus 4.8 / Sonnet 4.6 / Haiku 4.5) when ``ANTHROPIC_API_KEY`` is set — which turns
the eval harness into a true multi-model financial-correctness leaderboard.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
