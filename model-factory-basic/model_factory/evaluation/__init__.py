"""Model eval team: whenever training drops a `policy-checkpoint` artifact,
an eval run triggers — candidate vs base pass@1 on held-out verified tasks,
generated through the inference team's serving app. Publishes `eval-report`
and (after the auto gate + human promotion gate) `promoted-model`.
"""
