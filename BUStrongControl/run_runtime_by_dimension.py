import importlib


runtime_mod = importlib.import_module("benchmark_runtime_by_dimension")
importlib.reload(runtime_mod)


# Simulation settings.
runtime_mod.K_MIN = 2
runtime_mod.K_MAX = 10
runtime_mod.N_SIM = 10000
runtime_mod.RHO = 0.0
runtime_mod.SEED = 123
runtime_mod.N_JOBS = -1

# Data are generated from Pi_mix(0.7), with
# theta_K = Phi^{-1}(alpha / K) - Phi^{-1}(0.7).
runtime_mod.DATA_OBJECTIVE = "bayes"
runtime_mod.DATA_TARGET_POWER = 0.7
runtime_mod.ALTERNATIVE_PROBABILITY = 0.5


runtime_mod.main()
