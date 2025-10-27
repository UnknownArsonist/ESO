from GA_base import GA
from uniform_ga_mut import GeneticAlgorithm
import numpy as np
from gsemo import GSEMO, GSEMOConfig
from GA_base import Individual

def get_algorithm(name):
    if name == "SOEA":
        return SOEA_Submodular
    if name == "MOEA":
        return MOEA_Submodular
    if name == "EA":
        return EA
    if name == "RLS":
        return RLS
    if name == "GA":
        return GeneticAlgorithm
    if name == "GSEMO":
        def make(problem, budget, rng, k=10):
            cfg = GSEMOConfig(budget=budget, seed=int(rng.integers(1_000_000)))
            return _GSEMOAdapter(problem, k=k, cfg=cfg, rng=rng)
        return make
    if name == "SOEA":
        return SOEA_Submodular   
    if name == "MOEA":
        return MOEA_Submodular
    return None


class RLS(GA):
    def __init__(self, problem, budget=100_000, rng: np.random.Generator=np.random.default_rng(0)):
        super().__init__(problem, budget, rng, 1)

    def _mutate(self, x_bits):
        bits = x_bits.copy()
        randi = self.rng.integers(self.n)
        bits[randi] = 0 if bits[randi] == 1 else 1
        return bits


class EA(GA):
    def __init__(self, problem, budget=100_000, rng: np.random.Generator=np.random.default_rng(0)):
        super().__init__(problem, budget, rng, 1)

    def _mutate(self, x_bits):
        bits = np.copy(x_bits)
        # flip each bit with prob 1/n (compact form)
        flips = self.rng.random(self.n) < (1.0 / self.n)
        bits[flips] = 1 - bits[flips]
        return bits


class _GSEMOAdapter:
    def __init__(self, problem, k, cfg: GSEMOConfig, rng: np.random.Generator):
        self.inner = GSEMO(problem=problem, k=k, cfg=cfg)
        self.rng = rng

    def run(self):
        self.inner.run()
        return None

class SOEA_Submodular(GA):
    def __init__(self, problem, budget=10_000, rng=np.random.default_rng(0),
                 pop_size=20, target_k=None, diversity="maxmin", p_mut=None):
        super().__init__(problem, budget, rng, mu=pop_size)
        self.pop_size = pop_size
        self.target_k = target_k
        self.diversity = diversity
        self.p_mut = p_mut if p_mut is not None else (1.0 / max(1, self.n))

    def _random_bits_feasible(self):
        if self.target_k is None:
            return self.rng.integers(0, 2, size=self.n, dtype=np.int32)
        idx = self.rng.choice(self.n, size=self.target_k, replace=False)
        bits = np.zeros(self.n, dtype=np.int32)
        bits[idx] = 1
        return bits

    def _bitflip_mutation(self, bits):
        out = bits.copy()
        flip = self.rng.random(self.n) < self.p_mut
        out[flip] = 1 - out[flip]
        if self.target_k is not None:  
            c = out.sum()
            if c > self.target_k:
                one_idx = np.where(out == 1)[0]
                out[self.rng.choice(one_idx, size=c - self.target_k, replace=False)] = 0
            elif c < self.target_k:
                zero_idx = np.where(out == 0)[0]
                out[self.rng.choice(zero_idx, size=self.target_k - c, replace=False)] = 1
        return out

    def _initialize_pop(self):
        P = []
        for _ in range(self.pop_size):
            x = self._random_bits_feasible()
            fit = self._evaluate(x)
            P.append(Individual(bits=x, fitness=fit))
        self.best = max(P, key=lambda i: i.fitness)
        return P

    def run(self):
        P = self._initialize_pop()
        while self.eval_count < self.budget:
            offspring = []
            for _ in range(self.pop_size):
                p1, p2 = self.rng.choice(P, 2, replace=False)
                child_bits = np.where(self.rng.random(self.n) < 0.5, p1.bits, p2.bits)
                child_bits = self._bitflip_mutation(child_bits)
                fit = self._evaluate(child_bits)
                child = Individual(bits=child_bits, fitness=fit)
                offspring.append(child)
                if fit > self.best.fitness:
                    self.best = child
            P = sorted(P + offspring, key=lambda i: i.fitness, reverse=True)[:self.pop_size]
        return {"best_individual": self.best, "eval_count": self.eval_count}


# ==============================================================
#（MOEA_Submodular）
# ==============================================================
class MOEA_Submodular(GA):
    def __init__(self, problem, budget=10_000, rng=np.random.default_rng(0),
                 pop_size=40, target_k=None, p_mut=None):
        super().__init__(problem, budget, rng, mu=pop_size)
        self.N = pop_size
        self.target_k = target_k
        self.p_mut = p_mut if p_mut is not None else (1.0 / max(1, self.n))

    def _mutate(self, bits):
        out = bits.copy()
        flip = self.rng.random(self.n) < self.p_mut
        out[flip] = 1 - out[flip]
        return out

    def _crossover(self, a, b):
        mask = self.rng.random(self.n) < 0.5
        return np.where(mask, a, b).astype(np.int32)

    def _obj(self, bits):
        f1 = self._evaluate(bits)
        if self.target_k is not None:
            f2 = -abs(bits.sum() - self.target_k)
        else:
            f2 = bits.sum()
        return np.array([f1, f2], dtype=float)

    def run(self):
        X = [self.rng.integers(0, 2, size=self.n, dtype=np.int32) for _ in range(self.N)]
        F = [self._obj(x) for x in X]
        self.best = Individual(bits=X[np.argmax([f[0] for f in F])],
                               fitness=max([f[0] for f in F]))
        while self.eval_count < self.budget:
            offspring_X, offspring_F = [], []
            for _ in range(self.N):
                a, b = self.rng.choice(len(X), 2, replace=False)
                child = self._mutate(self._crossover(X[a], X[b]))
                val = self._obj(child)
                offspring_X.append(child)
                offspring_F.append(val)
                if val[0] > self.best.fitness:
                    self.best = Individual(bits=child, fitness=val[0])
            X += offspring_X
            F += offspring_F
            idx = np.argsort([-f[0] + 0.0001 * abs(f[1]) for f in F])[:self.N]
            X = [X[i] for i in idx]
            F = [F[i] for i in idx]
        return {"best_individual": self.best, "eval_count": self.eval_count}

