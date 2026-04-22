// Wren-style training harness for the Boom substrate.
//
// Runs a set of substrates (adepts) against K-fold examiner splits, collects
// out-of-fold predictions, then grid-searches per-target non-negative weights
// that sum to 1 ("promotion") before refitting every substrate on the full
// training set and combining them into the final ensemble.
//
// The harness also applies a per-target log1p transform to heavy-tailed
// distance targets (R95, R50_fines, R50_oversize); every substrate learns in
// the transformed space and predictions are inverted with expm1 before they
// reach the weight search, so NRMSE comparisons always happen in raw target
// space.

package main

import (
	"math"
	"math/rand"
)

// -----------------------------------------------------------------------------
// Substrate (adept) interface
// -----------------------------------------------------------------------------

type Substrate interface {
	Name() string
	Fit(X, Y [][]float64, rng *rand.Rand)
	Predict(x []float64) []float64
}

// ---- random forest adept ----

type rfSubstrate struct {
	m                                  *rfMultiOutput
	nTrees, minLeaf, featSub, maxDepth int
}

func newRFSubstrate(nTrees, minLeaf, featSub, maxDepth int) *rfSubstrate {
	return &rfSubstrate{nTrees: nTrees, minLeaf: minLeaf, featSub: featSub, maxDepth: maxDepth}
}

func (s *rfSubstrate) Name() string { return "rf" }

func (s *rfSubstrate) Fit(X, Y [][]float64, rng *rand.Rand) {
	s.m = fitRF(X, Y, s.nTrees, s.minLeaf, s.featSub, s.maxDepth, rng)
}

func (s *rfSubstrate) Predict(x []float64) []float64 { return s.m.predict(x) }

// ---- k-NN adept ----

type knnSubstrate struct {
	m *knnModel
	k int
}

func newKNNSubstrate(k int) *knnSubstrate { return &knnSubstrate{k: k} }

func (s *knnSubstrate) Name() string { return "knn" }

func (s *knnSubstrate) Fit(X, Y [][]float64, rng *rand.Rand) {
	s.m = fitKNN(X, Y, s.k)
}

func (s *knnSubstrate) Predict(x []float64) []float64 { return s.m.predict(x) }

// ---- linear ridge adept ----

type linearSubstrate struct {
	m     *linearModel
	sc    *scaler
	ridge float64
}

func newLinearSubstrate(ridge float64) *linearSubstrate { return &linearSubstrate{ridge: ridge} }

func (s *linearSubstrate) Name() string { return "linear" }

func (s *linearSubstrate) Fit(X, Y [][]float64, rng *rand.Rand) {
	s.sc = fitScaler(X)
	Xs := s.sc.transform(X)
	s.m = fitLinear(Xs, Y, s.ridge)
}

func (s *linearSubstrate) Predict(x []float64) []float64 {
	xs := make([]float64, len(x))
	for j, v := range x {
		xs[j] = (v - s.sc.mean[j]) / s.sc.std[j]
	}
	return s.m.predict(xs)
}

// ---- gradient-boosted trees adept ----

type gbmSubstrate struct {
	m                                  *gbmMultiOutput
	nRounds, maxDepth, minLeaf, featSub int
	shrinkage, subsample               float64
}

func newGBMSubstrate(nRounds, maxDepth, minLeaf, featSub int, shrinkage, subsample float64) *gbmSubstrate {
	return &gbmSubstrate{
		nRounds: nRounds, maxDepth: maxDepth, minLeaf: minLeaf, featSub: featSub,
		shrinkage: shrinkage, subsample: subsample,
	}
}

func (s *gbmSubstrate) Name() string { return "gbm" }

func (s *gbmSubstrate) Fit(X, Y [][]float64, rng *rand.Rand) {
	s.m = fitGBM(X, Y, s.nRounds, s.maxDepth, s.minLeaf, s.featSub, s.shrinkage, s.subsample, rng)
}

func (s *gbmSubstrate) Predict(x []float64) []float64 { return s.m.predict(x) }

// -----------------------------------------------------------------------------
// Gradient-boosting regressor (per-target) using the shared tree builder
// -----------------------------------------------------------------------------

type gbmModel struct {
	init   float64
	trees  []*rfTree
	shrink float64
}

func fitGBMOne(X [][]float64, y []float64, nRounds, maxDepth, minLeaf, featSub int, shrink, subsample float64, rng *rand.Rand) *gbmModel {
	n := len(X)
	var init float64
	for _, v := range y {
		init += v
	}
	init /= float64(n)
	F := make([]float64, n)
	for i := range F {
		F[i] = init
	}
	trees := make([]*rfTree, nRounds)
	for m := 0; m < nRounds; m++ {
		resid := make([]float64, n)
		for i := 0; i < n; i++ {
			resid[i] = y[i] - F[i]
		}
		var idx []int
		if subsample >= 1.0 {
			idx = make([]int, n)
			for i := range idx {
				idx[i] = i
			}
		} else {
			keep := int(float64(n) * subsample)
			if keep < 1 {
				keep = 1
			}
			perm := rng.Perm(n)
			idx = perm[:keep]
		}
		t := buildTree(X, resid, idx, minLeaf, featSub, maxDepth, rng)
		trees[m] = t
		for i := 0; i < n; i++ {
			F[i] += shrink * t.predict(X[i])
		}
	}
	return &gbmModel{init: init, trees: trees, shrink: shrink}
}

func (g *gbmModel) predict(x []float64) float64 {
	s := g.init
	for _, t := range g.trees {
		s += g.shrink * t.predict(x)
	}
	return s
}

type gbmMultiOutput struct {
	models []*gbmModel
}

func fitGBM(X, Y [][]float64, nRounds, maxDepth, minLeaf, featSub int, shrink, subsample float64, rng *rand.Rand) *gbmMultiOutput {
	nt := len(Y[0])
	n := len(X)
	out := &gbmMultiOutput{models: make([]*gbmModel, nt)}
	for t := 0; t < nt; t++ {
		y := make([]float64, n)
		for i := 0; i < n; i++ {
			y[i] = Y[i][t]
		}
		out.models[t] = fitGBMOne(X, y, nRounds, maxDepth, minLeaf, featSub, shrink, subsample, rng)
	}
	return out
}

func (g *gbmMultiOutput) predict(x []float64) []float64 {
	out := make([]float64, len(g.models))
	for t, m := range g.models {
		out[t] = m.predict(x)
	}
	return out
}

// -----------------------------------------------------------------------------
// Per-target log1p transform (heavy-tailed targets)
// -----------------------------------------------------------------------------

func applyLogTargets(Y [][]float64, mask []bool) [][]float64 {
	out := make([][]float64, len(Y))
	for i, row := range Y {
		nr := make([]float64, len(row))
		for j, v := range row {
			if mask[j] {
				if v < 0 {
					v = 0
				}
				nr[j] = math.Log1p(v)
			} else {
				nr[j] = v
			}
		}
		out[i] = nr
	}
	return out
}

func invertLogRow(pred []float64, mask []bool) []float64 {
	out := make([]float64, len(pred))
	for j, v := range pred {
		if mask[j] {
			w := math.Expm1(v)
			if w < 0 {
				w = 0
			}
			out[j] = w
		} else {
			out[j] = v
		}
	}
	return out
}

// -----------------------------------------------------------------------------
// K-fold examiner: out-of-fold predictions per substrate
// -----------------------------------------------------------------------------

// kfoldOOF returns (n x nTargets) out-of-fold predictions for sub over k folds.
// Fit is called k times on (k-1)/k of the data, Predict on the held-out fold.
func kfoldOOF(sub Substrate, X, Y [][]float64, k int, rng *rand.Rand) [][]float64 {
	n := len(X)
	oof := make([][]float64, n)
	order := shuffleIndices(n, rng)
	foldSize := n / k
	for f := 0; f < k; f++ {
		valStart := f * foldSize
		valEnd := valStart + foldSize
		if f == k-1 {
			valEnd = n
		}
		valIdx := order[valStart:valEnd]
		valSet := make(map[int]bool, len(valIdx))
		for _, id := range valIdx {
			valSet[id] = true
		}
		trIdx := make([]int, 0, n-len(valIdx))
		for _, id := range order {
			if !valSet[id] {
				trIdx = append(trIdx, id)
			}
		}
		Xtr := gather(X, trIdx)
		Ytr := gather(Y, trIdx)
		subRng := rand.New(rand.NewSource(rng.Int63()))
		sub.Fit(Xtr, Ytr, subRng)
		for _, id := range valIdx {
			oof[id] = sub.Predict(X[id])
		}
	}
	return oof
}

// -----------------------------------------------------------------------------
// Weight promotion: per-target simplex grid search over K substrates
// -----------------------------------------------------------------------------

// tuneWeightsK searches per-target non-negative weights (sum=1) over the K
// substrates, minimising MSE on decoded OOF predictions vs raw targets.
// preds[m] is the OOF matrix for substrate m in log-transformed space.
// steps = 1/grid. Compositions of steps into K parts are enumerated exactly.
func tuneWeightsK(Yraw [][]float64, preds [][][]float64, mask []bool, steps int) [][]float64 {
	K := len(preds)
	n := len(Yraw)
	nt := len(Yraw[0])

	decoded := make([][][]float64, K)
	for m := 0; m < K; m++ {
		decoded[m] = make([][]float64, n)
		for i := 0; i < n; i++ {
			decoded[m][i] = invertLogRow(preds[m][i], mask)
		}
	}

	compositions := enumerateCompositions(K, steps)

	out := make([][]float64, K)
	for m := 0; m < K; m++ {
		out[m] = make([]float64, nt)
	}

	for t := 0; t < nt; t++ {
		bestSE := math.Inf(1)
		var bestW []int
		for _, c := range compositions {
			var se float64
			for i := 0; i < n; i++ {
				var p float64
				for m := 0; m < K; m++ {
					p += float64(c[m]) * decoded[m][i][t]
				}
				p /= float64(steps)
				d := p - Yraw[i][t]
				se += d * d
			}
			if se < bestSE {
				bestSE = se
				bestW = c
			}
		}
		for m := 0; m < K; m++ {
			out[m][t] = float64(bestW[m]) / float64(steps)
		}
	}
	return out
}

// enumerateCompositions: all K-tuples of non-negative ints summing to steps.
func enumerateCompositions(K, steps int) [][]int {
	var out [][]int
	cur := make([]int, K)
	var rec func(idx, rem int)
	rec = func(idx, rem int) {
		if idx == K-1 {
			cur[idx] = rem
			cp := make([]int, K)
			copy(cp, cur)
			out = append(out, cp)
			return
		}
		for v := 0; v <= rem; v++ {
			cur[idx] = v
			rec(idx+1, rem-v)
		}
	}
	rec(0, steps)
	return out
}

// -----------------------------------------------------------------------------
// wrenEnsemble: the promoted, production ensemble used at predict time
// -----------------------------------------------------------------------------

type wrenEnsemble struct {
	subs       []Substrate
	W          [][]float64 // W[substrate][target]
	mask       []bool
	numTargets int
}

func (e *wrenEnsemble) predict(x []float64) []float64 {
	K := len(e.subs)
	decoded := make([][]float64, K)
	for m, s := range e.subs {
		decoded[m] = invertLogRow(s.Predict(x), e.mask)
	}
	out := make([]float64, e.numTargets)
	for t := 0; t < e.numTargets; t++ {
		for m := 0; m < K; m++ {
			out[t] += e.W[m][t] * decoded[m][t]
		}
	}
	return out
}
