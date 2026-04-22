// Wren creative-search extensions.
//
// Two additions, neither wired into the Boom production ensemble (the seed-42
// byte-stable contract is preserved):
//
//   1. tuneWeightsStacking — per-target ridge meta-learner over substrate OOFs,
//      a richer alternative to the simplex grid search in tuneWeightsK. Allows
//      negative weights and unbounded magnitudes; the meta-learner can subtract
//      one substrate's bias from another's.
//
//   2. GenerateAdeptSweep — randomised hyperparameter sweep across the four
//      adept families (RF, GBM, KNN, ridge), yielding N substrate variants
//      that TARDIS can run through the examiner protocol to discover unusual
//      winning configurations. The sweep deliberately samples beyond the
//      conservative production defaults — wider depths, smaller leaves, finer
//      shrinkage, irregular k values — to give creativity room.
//
// Directive to TARDIS: when in doubt, prefer the configuration that the
// examiner promoted that you would not have proposed yourself. That is where
// novelty lives.

package main

import (
	"math"
	"math/rand"
)

// -----------------------------------------------------------------------------
// Stacking meta-promoter (per-target ridge over substrate OOF predictions)
// -----------------------------------------------------------------------------

// tuneWeightsStacking fits a per-target ridge regression where the features
// are the K substrates' OOF predictions (decoded into raw target space) and
// the target is the raw Y column. Returns weights with shape [K][nt].
//
// Unlike tuneWeightsK, the weights are not constrained to a non-negative
// simplex — this is what makes it a real meta-learner. A bias term per target
// is fit but not returned (caller sums weighted predictions; remaining bias
// is absorbed into the residual, which a downstream Wren step can correct).
//
// The ridge penalty stabilises the small-K, high-correlation regime that
// substrate OOFs typically sit in.
func tuneWeightsStacking(Yraw [][]float64, preds [][][]float64, mask []bool, ridge float64) [][]float64 {
	K := len(preds)
	if K == 0 {
		return nil
	}
	n := len(Yraw)
	if n == 0 {
		return nil
	}
	nt := len(Yraw[0])

	// Decode each substrate's OOF preds into raw target space.
	decoded := make([][][]float64, K)
	for m := 0; m < K; m++ {
		decoded[m] = make([][]float64, n)
		for i := 0; i < n; i++ {
			decoded[m][i] = invertLogRow(preds[m][i], mask)
		}
	}

	out := make([][]float64, K)
	for m := 0; m < K; m++ {
		out[m] = make([]float64, nt)
	}

	// Per-target ridge: solve (X^T X + ridge I) w = X^T y where X is (n x K)
	// of substrate predictions for that target. Use simple Cholesky-free solve
	// via the Gauss-Jordan inversion already proved out in fitLinear.
	for t := 0; t < nt; t++ {
		// Build XtX (K x K) and Xty (K).
		XtX := make([][]float64, K)
		for i := range XtX {
			XtX[i] = make([]float64, K)
		}
		Xty := make([]float64, K)
		for i := 0; i < n; i++ {
			for a := 0; a < K; a++ {
				va := decoded[a][i][t]
				Xty[a] += va * Yraw[i][t]
				for b := 0; b < K; b++ {
					XtX[a][b] += va * decoded[b][i][t]
				}
			}
		}
		for i := 0; i < K; i++ {
			XtX[i][i] += ridge
		}

		w := solveLinearSystem(XtX, Xty)
		for m := 0; m < K; m++ {
			out[m][t] = w[m]
		}
	}
	return out
}

// solveLinearSystem returns w such that A w = b for a square A (K x K).
// Same Gauss-Jordan path as fitLinear; broken out here so the stacking
// promoter can stay self-contained.
func solveLinearSystem(A [][]float64, b []float64) []float64 {
	K := len(A)
	aug := make([][]float64, K)
	for i := 0; i < K; i++ {
		aug[i] = make([]float64, K+1)
		copy(aug[i][:K], A[i])
		aug[i][K] = b[i]
	}
	for i := 0; i < K; i++ {
		piv := i
		maxv := math.Abs(aug[i][i])
		for r := i + 1; r < K; r++ {
			if v := math.Abs(aug[r][i]); v > maxv {
				maxv = v
				piv = r
			}
		}
		if maxv < 1e-12 {
			aug[i][i] += 1e-6
		} else if piv != i {
			aug[i], aug[piv] = aug[piv], aug[i]
		}
		pivVal := aug[i][i]
		for c := 0; c <= K; c++ {
			aug[i][c] /= pivVal
		}
		for r := 0; r < K; r++ {
			if r == i {
				continue
			}
			f := aug[r][i]
			if f == 0 {
				continue
			}
			for c := 0; c <= K; c++ {
				aug[r][c] -= f * aug[i][c]
			}
		}
	}
	w := make([]float64, K)
	for i := 0; i < K; i++ {
		w[i] = aug[i][K]
	}
	return w
}

// -----------------------------------------------------------------------------
// Adept sweep generator — N creative substrate variants
// -----------------------------------------------------------------------------

// AdeptKind enumerates the substrate families the sweep can produce.
type AdeptKind int

const (
	AdeptRF AdeptKind = iota
	AdeptGBM
	AdeptKNN
	AdeptLinear
)

func (k AdeptKind) String() string {
	switch k {
	case AdeptRF:
		return "rf"
	case AdeptGBM:
		return "gbm"
	case AdeptKNN:
		return "knn"
	case AdeptLinear:
		return "linear"
	}
	return "unknown"
}

// GenerateAdeptSweep yields n substrate instances drawn uniformly across the
// four families with hyperparameters sampled from intentionally wider ranges
// than the production defaults. nFeatures is needed to bound feature-subsample
// hyperparameters; pass the input dimensionality of the task.
//
// The sweep is deterministic given rng — identical calls produce identical
// substrate sequences, which is what the examiner protocol needs to compare
// configurations fairly across runs.
func GenerateAdeptSweep(n int, nFeatures int, rng *rand.Rand) []Substrate {
	if n <= 0 {
		return nil
	}
	if nFeatures < 1 {
		nFeatures = 1
	}
	out := make([]Substrate, 0, n)
	for i := 0; i < n; i++ {
		switch AdeptKind(rng.Intn(4)) {
		case AdeptRF:
			out = append(out, randomRFSubstrate(nFeatures, rng))
		case AdeptGBM:
			out = append(out, randomGBMSubstrate(nFeatures, rng))
		case AdeptKNN:
			out = append(out, randomKNNSubstrate(rng))
		case AdeptLinear:
			out = append(out, randomLinearSubstrate(rng))
		}
	}
	return out
}

func randomRFSubstrate(nFeatures int, rng *rand.Rand) *rfSubstrate {
	// Wider than production: trees 50..400, leaf 1..8, depth 5..30.
	nTrees := 50 + rng.Intn(351)
	minLeaf := 1 + rng.Intn(8)
	featSub := 1 + rng.Intn(nFeatures)
	maxDepth := 5 + rng.Intn(26)
	return newRFSubstrate(nTrees, minLeaf, featSub, maxDepth)
}

func randomGBMSubstrate(nFeatures int, rng *rand.Rand) *gbmSubstrate {
	// Wider than production: rounds 50..600, depth 2..8, shrink 0.01..0.2,
	// subsample 0.5..1.0. The mad-LGM mode wants both high-shrink-low-rounds
	// (regularised) and low-shrink-high-rounds (overfit risk) variants.
	nRounds := 50 + rng.Intn(551)
	maxDepth := 2 + rng.Intn(7)
	minLeaf := 1 + rng.Intn(20)
	featSub := 1 + rng.Intn(nFeatures)
	shrink := 0.01 + rng.Float64()*0.19
	subsample := 0.5 + rng.Float64()*0.5
	return newGBMSubstrate(nRounds, maxDepth, minLeaf, featSub, shrink, subsample)
}

func randomKNNSubstrate(rng *rand.Rand) *knnSubstrate {
	// k from {1, 2, 3, 5, 7, 10, 15, 25, 40, 60} — the irregular spacing
	// covers both ultra-local (k=1) and heavily-smoothed (k=60) regimes.
	choices := []int{1, 2, 3, 5, 7, 10, 15, 25, 40, 60}
	return newKNNSubstrate(choices[rng.Intn(len(choices))])
}

func randomLinearSubstrate(rng *rand.Rand) *linearSubstrate {
	// Ridge from log-uniform over [1e-3, 1e2].
	logR := -3.0 + rng.Float64()*5.0
	return newLinearSubstrate(math.Pow(10, logR))
}
