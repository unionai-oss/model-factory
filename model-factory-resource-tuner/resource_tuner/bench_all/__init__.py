"""RightSize Bench: the whole request at once -- memory, CPU, and a GPU or none.

RightSizeMem, RightSizeCPU and RightSizeGPU each score one axis of a
request. A scheduler makes all three at once, and the axes interact: on a
GPU node host memory and cores come with the instance; on a CPU node a GPU
is the most expensive mistake there is; and a GPU task with no GPU does not
run at all. This bench joins the three measured corpora, labels every run
on every axis it measured, and scores the joint request under one dollar
cost model (``scoring``) -- the same simulator the RL arm (``policy``)
trains against.

Modules: grid, dataset, features, prompting, scoring, arms, policy, envs,
stations, factory.
"""
