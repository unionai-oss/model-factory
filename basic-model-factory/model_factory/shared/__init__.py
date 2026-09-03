"""Shared platform libraries — the only code teams import besides contracts.

sandbox / rewards: verified-reward machinery (pure python, unit-tested)
reporting:         HTML builders for flyte.report
assets:            latest-asset resolution (artifact API + run-scan fallback)
gates:             human-in-the-loop condition gates
images:            base container images + secret wiring
inference_client:  HTTP client for the inference team's serving app
"""
