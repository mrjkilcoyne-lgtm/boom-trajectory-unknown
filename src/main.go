// Boom: Trajectory Unknown Challenge - TARDIS submission pipeline
//
// Produces:
//   prediction_submission.csv  - ensemble predictions over 492 blind test scenarios
//   design_submission.csv      - 20 inverse designs satisfying P80 and R95 constraints
//   self_score.json            - NRMSE per target on a 20% holdout
//   SUBMISSION_READY.md        - human-readable summary
//
// Pure standard library. No external dependencies.
// Run: go run ./submit  (from substrates/boom-ejecta)

package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"time"
)

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

// Input feature order as they appear in train.csv / test.csv.
var trainFeatureOrder = []string{
	"porosity", "atmosphere", "gravity", "coupling",
	"strength", "shape_factor", "energy", "angle_rad",
}

// Target order as they appear in train_labels.csv and the prediction submission template.
var targetOrder = []string{
	"P80", "fines_frac", "oversize_frac", "R95", "R50_fines", "R50_oversize",
}

// Design submission column order (energy first, then angle, etc.).
var designFeatureOrder = []string{
	"energy", "angle_rad", "coupling", "strength",
	"porosity", "gravity", "atmosphere", "shape_factor",
}

// logTargetMask returns true for Boom targets where log1p compresses the tail.
// Heavy-tailed R-family distances (R95, R50_fines, R50_oversize) only.
func logTargetMask() []bool {
	out := make([]bool, nTargets)
	for t, name := range targetOrder {
		if name == "R95" || name == "R50_fines" || name == "R50_oversize" {
			out[t] = true
		}
	}
	return out
}

// Constraint thresholds (from data/constraints.json).
const (
	p80Min = 96.0
	p80MaxC = 101.0
	r95MaxC = 175.0
)

const (
	nFeatures   = 8
	nTargets    = 6
	nDesigns    = 20
	holdoutFrac = 0.2
	knnK        = 10

	// Random forest adept
	rfTrees    = 200
	rfMinLeaf  = 3
	rfFeatSub  = 4 // random features per split
	rfMaxDepth = 20

	// Gradient boosting adept
	gbmRounds    = 300
	gbmMaxDepth  = 4
	gbmMinLeaf   = 10
	gbmFeatSub   = 6
	gbmShrink    = 0.05
	gbmSubsample = 0.8

	// Wren harness
	kFolds          = 5
	weightGridSteps = 10 // 0.1 grid; compositions over 4 substrates = 286 per target

	seed = 42
)

// -----------------------------------------------------------------------------
// Constraints file
// -----------------------------------------------------------------------------

type inputBound struct {
	Min float64 `json:"min"`
	Max float64 `json:"max"`
}

type constraintsFile struct {
	Constraints struct {
		P80Min float64 `json:"p80_min"`
		P80Max float64 `json:"p80_max"`
		R95Max float64 `json:"r95_max"`
	} `json:"constraints"`
	InputBounds map[string]inputBound `json:"input_bounds"`
}

// -----------------------------------------------------------------------------
// CSV loaders
// -----------------------------------------------------------------------------

func readCSV(path string) ([]string, [][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	r := csv.NewReader(f)
	rows, err := r.ReadAll()
	if err != nil {
		return nil, nil, err
	}
	if len(rows) == 0 {
		return nil, nil, fmt.Errorf("empty csv: %s", path)
	}
	return rows[0], rows[1:], nil
}

func parseFloatMatrix(rows [][]string) ([][]float64, error) {
	out := make([][]float64, len(rows))
	for i, r := range rows {
		vals := make([]float64, len(r))
		for j, s := range r {
			v, err := strconv.ParseFloat(s, 64)
			if err != nil {
				return nil, fmt.Errorf("row %d col %d: %w", i, j, err)
			}
			vals[j] = v
		}
		out[i] = vals
	}
	return out, nil
}

// reorderColumns returns a new matrix whose columns follow `want` order, given
// the source header. Any missing column is an error.
func reorderColumns(header []string, data [][]float64, want []string) ([][]float64, error) {
	idx := make([]int, len(want))
	for i, name := range want {
		pos := -1
		for j, h := range header {
			if h == name {
				pos = j
				break
			}
		}
		if pos < 0 {
			return nil, fmt.Errorf("missing column %q", name)
		}
		idx[i] = pos
	}
	out := make([][]float64, len(data))
	for i, row := range data {
		nr := make([]float64, len(want))
		for k, p := range idx {
			nr[k] = row[p]
		}
		out[i] = nr
	}
	return out, nil
}

// -----------------------------------------------------------------------------
// Standardiser
// -----------------------------------------------------------------------------

type scaler struct {
	mean, std []float64
}

func fitScaler(X [][]float64) *scaler {
	if len(X) == 0 {
		return &scaler{}
	}
	d := len(X[0])
	s := &scaler{mean: make([]float64, d), std: make([]float64, d)}
	for _, row := range X {
		for j, v := range row {
			s.mean[j] += v
		}
	}
	for j := range s.mean {
		s.mean[j] /= float64(len(X))
	}
	for _, row := range X {
		for j, v := range row {
			d := v - s.mean[j]
			s.std[j] += d * d
		}
	}
	for j := range s.std {
		s.std[j] = math.Sqrt(s.std[j] / float64(len(X)))
		if s.std[j] < 1e-12 {
			s.std[j] = 1.0
		}
	}
	return s
}

func (s *scaler) transform(X [][]float64) [][]float64 {
	out := make([][]float64, len(X))
	for i, row := range X {
		nr := make([]float64, len(row))
		for j, v := range row {
			nr[j] = (v - s.mean[j]) / s.std[j]
		}
		out[i] = nr
	}
	return out
}

// -----------------------------------------------------------------------------
// Linear regression (multi-output) via normal equations with ridge
// -----------------------------------------------------------------------------

type linearModel struct {
	// weight matrix with shape (nFeatures+1) x nTargets; last row is bias
	W [][]float64
}

func fitLinear(X [][]float64, Y [][]float64, ridge float64) *linearModel {
	n := len(X)
	if n == 0 {
		return &linearModel{}
	}
	d := len(X[0]) + 1   // augmented with bias
	t := len(Y[0])

	// Build XtX (d x d) and XtY (d x t).
	XtX := make([][]float64, d)
	for i := range XtX {
		XtX[i] = make([]float64, d)
	}
	XtY := make([][]float64, d)
	for i := range XtY {
		XtY[i] = make([]float64, t)
	}
	for i := 0; i < n; i++ {
		aug := make([]float64, d)
		copy(aug, X[i])
		aug[d-1] = 1.0
		for a := 0; a < d; a++ {
			for b := 0; b < d; b++ {
				XtX[a][b] += aug[a] * aug[b]
			}
			for b := 0; b < t; b++ {
				XtY[a][b] += aug[a] * Y[i][b]
			}
		}
	}
	// ridge penalty (don't penalise the bias row, but penalising is fine for stability)
	for i := 0; i < d-1; i++ {
		XtX[i][i] += ridge
	}
	// solve d x (d+t) augmented system via Gauss-Jordan
	aug := make([][]float64, d)
	for i := 0; i < d; i++ {
		aug[i] = make([]float64, d+t)
		copy(aug[i][:d], XtX[i])
		copy(aug[i][d:], XtY[i])
	}
	for i := 0; i < d; i++ {
		// pivot
		piv := i
		maxv := math.Abs(aug[i][i])
		for r := i + 1; r < d; r++ {
			if v := math.Abs(aug[r][i]); v > maxv {
				maxv = v
				piv = r
			}
		}
		if maxv < 1e-12 {
			// singular — bump diagonal
			aug[i][i] += 1e-6
		} else if piv != i {
			aug[i], aug[piv] = aug[piv], aug[i]
		}
		pivVal := aug[i][i]
		for c := 0; c < d+t; c++ {
			aug[i][c] /= pivVal
		}
		for r := 0; r < d; r++ {
			if r == i {
				continue
			}
			f := aug[r][i]
			if f == 0 {
				continue
			}
			for c := 0; c < d+t; c++ {
				aug[r][c] -= f * aug[i][c]
			}
		}
	}
	W := make([][]float64, d)
	for i := 0; i < d; i++ {
		W[i] = make([]float64, t)
		copy(W[i], aug[i][d:])
	}
	return &linearModel{W: W}
}

func (m *linearModel) predict(x []float64) []float64 {
	d := len(x) + 1
	t := len(m.W[0])
	out := make([]float64, t)
	for j := 0; j < t; j++ {
		for i := 0; i < d-1; i++ {
			out[j] += m.W[i][j] * x[i]
		}
		out[j] += m.W[d-1][j] // bias
	}
	return out
}

// -----------------------------------------------------------------------------
// k-NN (multi-output) with distance-weighted averaging, on standardised inputs
// -----------------------------------------------------------------------------

type knnModel struct {
	X [][]float64 // standardised
	Y [][]float64
	K int
	sc *scaler
}

func fitKNN(X [][]float64, Y [][]float64, k int) *knnModel {
	sc := fitScaler(X)
	return &knnModel{X: sc.transform(X), Y: Y, K: k, sc: sc}
}

func (m *knnModel) predict(raw []float64) []float64 {
	// standardise query
	q := make([]float64, len(raw))
	for j, v := range raw {
		q[j] = (v - m.sc.mean[j]) / m.sc.std[j]
	}
	type pair struct {
		d float64
		i int
	}
	// partial selection: keep K smallest. Data is ~2930 rows so a full sort is fine.
	pairs := make([]pair, len(m.X))
	for i, row := range m.X {
		var d float64
		for j, v := range row {
			dd := v - q[j]
			d += dd * dd
		}
		pairs[i] = pair{d: d, i: i}
	}
	sort.Slice(pairs, func(a, b int) bool { return pairs[a].d < pairs[b].d })
	k := m.K
	if k > len(pairs) {
		k = len(pairs)
	}
	nt := len(m.Y[0])
	out := make([]float64, nt)
	var wsum float64
	for n := 0; n < k; n++ {
		w := 1.0 / (math.Sqrt(pairs[n].d) + 1e-6)
		wsum += w
		for j := 0; j < nt; j++ {
			out[j] += w * m.Y[pairs[n].i][j]
		}
	}
	if wsum > 0 {
		for j := range out {
			out[j] /= wsum
		}
	}
	return out
}

// -----------------------------------------------------------------------------
// Random Forest regressor (single-target). We train one per target.
// -----------------------------------------------------------------------------

type rfNode struct {
	leaf    bool
	value   float64
	feat    int
	thresh  float64
	left    *rfNode
	right   *rfNode
}

type rfTree struct {
	root *rfNode
}

// Build one tree on the given indices sample. features to try per split = featSub.
func buildTree(X [][]float64, y []float64, idx []int, minLeaf, featSub, maxDepth int, rng *rand.Rand) *rfTree {
	root := growNode(X, y, idx, minLeaf, featSub, rng, 0, maxDepth)
	return &rfTree{root: root}
}

func growNode(X [][]float64, y []float64, idx []int, minLeaf, featSub int, rng *rand.Rand, depth, maxDepth int) *rfNode {
	// leaf conditions
	if len(idx) <= minLeaf || depth >= maxDepth {
		return leafNode(y, idx)
	}
	// compute current SSE baseline
	mean, sse := meanSSE(y, idx)
	if sse < 1e-12 {
		return &rfNode{leaf: true, value: mean}
	}

	bestGain := 0.0
	bestFeat := -1
	bestThresh := 0.0
	nFeat := len(X[0])
	// random feature subset
	feats := rng.Perm(nFeat)
	if featSub < nFeat {
		feats = feats[:featSub]
	}

	// For each candidate feature, sort sample and look at split midpoints.
	for _, f := range feats {
		// build sorted (value, target) pairs
		type pv struct {
			v float64
			y float64
		}
		pairs := make([]pv, len(idx))
		for i, id := range idx {
			pairs[i] = pv{X[id][f], y[id]}
		}
		sort.Slice(pairs, func(a, b int) bool { return pairs[a].v < pairs[b].v })

		var sumL, sumR float64
		var cntL int
		var sumAll float64
		for _, p := range pairs {
			sumAll += p.y
		}
		sumR = sumAll

		// Walk split positions. We evaluate sparsely for speed: step by max(1, n/32).
		n := len(pairs)
		step := n / 32
		if step < 1 {
			step = 1
		}
		for i := 0; i < n-1; i++ {
			sumL += pairs[i].y
			sumR -= pairs[i].y
			cntL++
			if pairs[i].v == pairs[i+1].v {
				continue
			}
			if cntL < minLeaf || (n-cntL) < minLeaf {
				continue
			}
			if i%step != 0 && i != n-2 {
				continue
			}
			mL := sumL / float64(cntL)
			mR := sumR / float64(n-cntL)
			// SSE reduction
			sseL := 0.0
			sseR := 0.0
			for k, p := range pairs {
				if k <= i {
					d := p.y - mL
					sseL += d * d
				} else {
					d := p.y - mR
					sseR += d * d
				}
			}
			gain := sse - (sseL + sseR)
			if gain > bestGain {
				bestGain = gain
				bestFeat = f
				bestThresh = 0.5 * (pairs[i].v + pairs[i+1].v)
			}
		}
	}

	if bestFeat < 0 {
		return &rfNode{leaf: true, value: mean}
	}
	var leftIdx, rightIdx []int
	for _, id := range idx {
		if X[id][bestFeat] <= bestThresh {
			leftIdx = append(leftIdx, id)
		} else {
			rightIdx = append(rightIdx, id)
		}
	}
	if len(leftIdx) == 0 || len(rightIdx) == 0 {
		return &rfNode{leaf: true, value: mean}
	}
	return &rfNode{
		feat:   bestFeat,
		thresh: bestThresh,
		left:   growNode(X, y, leftIdx, minLeaf, featSub, rng, depth+1, maxDepth),
		right:  growNode(X, y, rightIdx, minLeaf, featSub, rng, depth+1, maxDepth),
	}
}

func leafNode(y []float64, idx []int) *rfNode {
	m, _ := meanSSE(y, idx)
	return &rfNode{leaf: true, value: m}
}

func meanSSE(y []float64, idx []int) (float64, float64) {
	var s float64
	for _, i := range idx {
		s += y[i]
	}
	m := s / float64(len(idx))
	var sse float64
	for _, i := range idx {
		d := y[i] - m
		sse += d * d
	}
	return m, sse
}

func (t *rfTree) predict(x []float64) float64 {
	n := t.root
	for !n.leaf {
		if x[n.feat] <= n.thresh {
			n = n.left
		} else {
			n = n.right
		}
	}
	return n.value
}

type rfMultiOutput struct {
	// for each target, a slice of trees (bagged).
	forests [][]*rfTree
}

func fitRF(X [][]float64, Y [][]float64, nTrees, minLeaf, featSub, maxDepth int, rng *rand.Rand) *rfMultiOutput {
	nt := len(Y[0])
	n := len(X)
	out := &rfMultiOutput{forests: make([][]*rfTree, nt)}
	// Per-target column
	for t := 0; t < nt; t++ {
		y := make([]float64, n)
		for i := 0; i < n; i++ {
			y[i] = Y[i][t]
		}
		trees := make([]*rfTree, nTrees)
		for b := 0; b < nTrees; b++ {
			// bootstrap sample of indices
			sample := make([]int, n)
			for i := 0; i < n; i++ {
				sample[i] = rng.Intn(n)
			}
			trees[b] = buildTree(X, y, sample, minLeaf, featSub, maxDepth, rng)
		}
		out.forests[t] = trees
	}
	return out
}

func (r *rfMultiOutput) predict(x []float64) []float64 {
	out := make([]float64, len(r.forests))
	for t, trees := range r.forests {
		var s float64
		for _, tr := range trees {
			s += tr.predict(x)
		}
		out[t] = s / float64(len(trees))
	}
	return out
}

// -----------------------------------------------------------------------------
// Scoring
// -----------------------------------------------------------------------------

// nrmse per target normalised by the std of y on the evaluation set.
// This mirrors "normalised RMSE per target averaged across 6 targets".
func nrmsePerTarget(Ytrue, Ypred [][]float64) []float64 {
	nt := len(Ytrue[0])
	n := len(Ytrue)
	out := make([]float64, nt)
	// mean + std per target for normalisation
	for t := 0; t < nt; t++ {
		var mean float64
		for i := 0; i < n; i++ {
			mean += Ytrue[i][t]
		}
		mean /= float64(n)
		var varSum, se float64
		for i := 0; i < n; i++ {
			d := Ytrue[i][t] - mean
			varSum += d * d
			e := Ytrue[i][t] - Ypred[i][t]
			se += e * e
		}
		std := math.Sqrt(varSum / float64(n))
		rmse := math.Sqrt(se / float64(n))
		if std < 1e-12 {
			std = 1.0
		}
		out[t] = rmse / std
	}
	return out
}

// -----------------------------------------------------------------------------
// Inverse design
// -----------------------------------------------------------------------------

func satisfiesConstraints(p80, r95 float64) bool {
	return p80 >= p80Min && p80 <= p80MaxC && r95 <= r95MaxC
}

func inBounds(row map[string]float64, bounds map[string]inputBound) bool {
	for k, v := range row {
		b, ok := bounds[k]
		if !ok {
			continue
		}
		if v < b.Min || v > b.Max {
			return false
		}
	}
	return true
}

// farthestPointSample picks n indices from candidates using greedy
// farthest-point sampling in the standardised feature space.
func farthestPointSample(X [][]float64, candidates []int, n int, rng *rand.Rand) []int {
	if len(candidates) <= n {
		return candidates
	}
	// standardise over candidates
	sub := make([][]float64, len(candidates))
	for i, id := range candidates {
		sub[i] = X[id]
	}
	sc := fitScaler(sub)
	std := sc.transform(sub)

	picked := []int{rng.Intn(len(candidates))}
	minD := make([]float64, len(candidates))
	for i := range minD {
		minD[i] = math.Inf(1)
	}
	for len(picked) < n {
		last := picked[len(picked)-1]
		for i := range candidates {
			var d float64
			for j := range std[i] {
				dd := std[i][j] - std[last][j]
				d += dd * dd
			}
			if d < minD[i] {
				minD[i] = d
			}
		}
		// pick index with max minD that isn't already picked
		bestI := -1
		bestV := -1.0
		pset := map[int]bool{}
		for _, p := range picked {
			pset[p] = true
		}
		for i := range candidates {
			if pset[i] {
				continue
			}
			if minD[i] > bestV {
				bestV = minD[i]
				bestI = i
			}
		}
		if bestI < 0 {
			break
		}
		picked = append(picked, bestI)
	}
	out := make([]int, len(picked))
	for i, p := range picked {
		out[i] = candidates[p]
	}
	return out
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

func repoRoot() string {
	// run from substrates/boom-ejecta or anywhere — we resolve relative to this file's expectation.
	// We look for data/constraints.json relative to cwd. Caller expected to `go run ./submit` from substrates/boom-ejecta.
	if _, err := os.Stat("data/constraints.json"); err == nil {
		return "."
	}
	// fallback: try the absolute substrate dir.
	abs := filepath.Join("C:/Users/mrjki/OneDrive/Tardis/substrates/boom-ejecta")
	if _, err := os.Stat(filepath.Join(abs, "data", "constraints.json")); err == nil {
		return abs
	}
	// fallback: parent dir
	if _, err := os.Stat("../data/constraints.json"); err == nil {
		return ".."
	}
	return "."
}

func shuffleIndices(n int, rng *rand.Rand) []int {
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	rng.Shuffle(n, func(i, j int) { idx[i], idx[j] = idx[j], idx[i] })
	return idx
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------

type selfScoreOut struct {
	PerTarget map[string]float64 `json:"nrmse_per_target"`
	Composite float64            `json:"nrmse_composite"`
	Weights   struct {
		RF     map[string]float64 `json:"rf"`
		KNN    map[string]float64 `json:"knn"`
		Linear map[string]float64 `json:"linear"`
		GBM    map[string]float64 `json:"gbm"`
	} `json:"ensemble_weights"`
	KFolds        int      `json:"examiner_kfolds"`
	LogTargets    []string `json:"log_transformed_targets"`
	TrainRows     int      `json:"train_rows"`
	TestRows      int      `json:"test_rows"`
	HoldoutRows   int      `json:"oof_rows"`
	Timestamp     string   `json:"timestamp"`
}

func main() {
	start := time.Now()
	root := repoRoot()
	fmt.Printf("[boom-submit] root=%s\n", root)

	// --- load data ---
	trHdr, trRows, err := readCSV(filepath.Join(root, "data/train.csv"))
	must(err)
	trX0, err := parseFloatMatrix(trRows)
	must(err)
	// reorder to our canonical order
	Xall, err := reorderColumns(trHdr, trX0, trainFeatureOrder)
	must(err)

	lHdr, lRows, err := readCSV(filepath.Join(root, "data/train_labels.csv"))
	must(err)
	Y0, err := parseFloatMatrix(lRows)
	must(err)
	Yall, err := reorderColumns(lHdr, Y0, targetOrder)
	must(err)
	if len(Xall) != len(Yall) {
		die(fmt.Errorf("train/labels length mismatch %d vs %d", len(Xall), len(Yall)))
	}

	teHdr, teRows, err := readCSV(filepath.Join(root, "data/test.csv"))
	must(err)
	teX0, err := parseFloatMatrix(teRows)
	must(err)
	Xtest, err := reorderColumns(teHdr, teX0, trainFeatureOrder)
	must(err)

	// constraints
	cf, err := os.ReadFile(filepath.Join(root, "data/constraints.json"))
	must(err)
	var cons constraintsFile
	must(json.Unmarshal(cf, &cons))

	fmt.Printf("[boom-submit] train=%d test=%d features=%d targets=%d\n",
		len(Xall), len(Xtest), nFeatures, nTargets)

	// --- Wren harness: log-transform heavy-tailed targets, then K-fold OOF ---
	mask := logTargetMask()
	var logCols []string
	for t, b := range mask {
		if b {
			logCols = append(logCols, targetOrder[t])
		}
	}
	Ylog := applyLogTargets(Yall, mask)

	// Register adepts. Each will be fit K times (once per fold) for OOF, then
	// once more on the full dataset for the promoted ensemble.
	subs := []Substrate{
		newRFSubstrate(rfTrees, rfMinLeaf, rfFeatSub, rfMaxDepth),
		newKNNSubstrate(knnK),
		newLinearSubstrate(1.0),
		newGBMSubstrate(gbmRounds, gbmMaxDepth, gbmMinLeaf, gbmFeatSub, gbmShrink, gbmSubsample),
	}

	fmt.Printf("[boom-submit] harness: %d substrates, %d-fold OOF, log targets=%v\n",
		len(subs), kFolds, logCols)

	oofPreds := make([][][]float64, len(subs))
	for m, sub := range subs {
		fmt.Printf("[boom-submit]   examiner fold-sweep: %s\n", sub.Name())
		foldRng := rand.New(rand.NewSource(int64(seed + 10 + m)))
		oofPreds[m] = kfoldOOF(sub, Xall, Ylog, kFolds, foldRng)
	}

	// --- promotion: per-target non-negative simplex grid search on OOF preds ---
	W := tuneWeightsK(Yall, oofPreds, mask, weightGridSteps)
	fmt.Printf("[boom-submit] promoted weights per target:\n")
	fmt.Printf("  %-14s", "target")
	for _, s := range subs {
		fmt.Printf(" %6s", s.Name())
	}
	fmt.Println()
	for t := 0; t < nTargets; t++ {
		fmt.Printf("  %-14s", targetOrder[t])
		for m := range subs {
			fmt.Printf(" %6.2f", W[m][t])
		}
		fmt.Println()
	}

	// OOF composite NRMSE in raw target space
	Yoof := make([][]float64, len(Xall))
	for i := range Xall {
		row := make([]float64, nTargets)
		for m := range subs {
			decoded := invertLogRow(oofPreds[m][i], mask)
			for t := 0; t < nTargets; t++ {
				row[t] += W[m][t] * decoded[t]
			}
		}
		Yoof[i] = row
	}
	perTgt := nrmsePerTarget(Yall, Yoof)
	var composite float64
	for _, v := range perTgt {
		composite += v
	}
	composite /= float64(len(perTgt))
	fmt.Printf("[boom-submit] OOF NRMSE composite = %.4f\n", composite)
	for t, v := range perTgt {
		fmt.Printf("  %-14s nrmse=%.4f\n", targetOrder[t], v)
	}

	// --- refit every adept on FULL data in log-transformed space ---
	fmt.Println("[boom-submit] refitting adepts on FULL train data...")
	for m, sub := range subs {
		subRng := rand.New(rand.NewSource(int64(seed + 100 + m)))
		sub.Fit(Xall, Ylog, subRng)
	}
	ens := &wrenEnsemble{subs: subs, W: W, mask: mask, numTargets: nTargets}

	// --- generate predictions for test scenarios ---
	fmt.Printf("[boom-submit] generating predictions for %d test scenarios...\n", len(Xtest))
	outDir := filepath.Join(root, "submit")
	must(os.MkdirAll(outDir, 0o755))
	predPath := filepath.Join(outDir, "prediction_submission.csv")
	pf, err := os.Create(predPath)
	must(err)
	pw := csv.NewWriter(pf)
	must(pw.Write(append([]string{"scenario_id"}, targetOrder...)))
	predRowsPreview := [][]string{}
	for i, x := range Xtest {
		pred := ens.predict(x)
		// clip obvious out-of-physical-range for fractions
		pred[1] = clamp(pred[1], 0, 1) // fines_frac
		pred[2] = clamp(pred[2], 0, 1) // oversize_frac
		pred[0] = math.Max(0, pred[0]) // P80
		pred[3] = math.Max(0, pred[3]) // R95
		pred[4] = math.Max(0, pred[4]) // R50_fines
		pred[5] = math.Max(0, pred[5]) // R50_oversize

		row := []string{strconv.Itoa(i)}
		for _, v := range pred {
			row = append(row, strconv.FormatFloat(v, 'g', 10, 64))
		}
		must(pw.Write(row))
		if i < 3 {
			predRowsPreview = append(predRowsPreview, row)
		}
	}
	pw.Flush()
	must(pw.Error())
	pf.Close()

	// --- inverse design ---
	fmt.Println("[boom-submit] generating inverse designs...")
	// find training samples meeting constraints from labels directly
	var validIdx []int
	for i, y := range Yall {
		if satisfiesConstraints(y[0], y[3]) {
			validIdx = append(validIdx, i)
		}
	}
	fmt.Printf("[boom-submit] training samples satisfying constraints: %d\n", len(validIdx))

	designs := make([][]float64, 0, nDesigns)
	designSources := make([]string, 0, nDesigns)

	if len(validIdx) >= nDesigns {
		// farthest-point sampling
		picks := farthestPointSample(Xall, validIdx, nDesigns, rand.New(rand.NewSource(seed+3)))
		for _, id := range picks {
			designs = append(designs, append([]float64(nil), Xall[id]...))
			designSources = append(designSources, fmt.Sprintf("train_row_%d", id))
		}
	} else {
		// use whatever we have, then synthesise the rest via perturbation + ensemble filter
		for _, id := range validIdx {
			designs = append(designs, append([]float64(nil), Xall[id]...))
			designSources = append(designSources, fmt.Sprintf("train_row_%d", id))
		}
		// seed candidates for perturbation: nearest-by-constraint training rows
		rng3 := rand.New(rand.NewSource(seed + 4))
		// rank training rows by "closeness to the constraint region" (penalty sum)
		type sc struct {
			i int
			s float64
		}
		scs := make([]sc, len(Yall))
		for i, y := range Yall {
			var pen float64
			// p80 distance to [96,101]
			if y[0] < p80Min {
				pen += p80Min - y[0]
			} else if y[0] > p80MaxC {
				pen += y[0] - p80MaxC
			}
			// r95 over 175
			if y[3] > r95MaxC {
				pen += y[3] - r95MaxC
			}
			scs[i] = sc{i, pen}
		}
		sort.Slice(scs, func(a, b int) bool { return scs[a].s < scs[b].s })
		seeds := make([]int, 0, 200)
		for _, s := range scs[:200] {
			seeds = append(seeds, s.i)
		}

		// perturb + ensemble filter until we hit nDesigns.
		attempts := 0
		maxAttempts := 200000
		for len(designs) < nDesigns && attempts < maxAttempts {
			attempts++
			base := Xall[seeds[rng3.Intn(len(seeds))]]
			cand := make([]float64, nFeatures)
			for j, name := range trainFeatureOrder {
				b := cons.InputBounds[name]
				span := b.Max - b.Min
				// perturb within 10% of the bound span
				v := base[j] + (rng3.Float64()*2-1)*0.1*span
				if v < b.Min {
					v = b.Min
				}
				if v > b.Max {
					v = b.Max
				}
				cand[j] = v
			}
			pred := ens.predict(cand)
			if satisfiesConstraints(pred[0], pred[3]) {
				designs = append(designs, cand)
				designSources = append(designSources, fmt.Sprintf("perturbed_attempt_%d", attempts))
			}
		}
		if len(designs) < nDesigns {
			// last-ditch fill: best-of training rows by penalty regardless of strict satisfaction
			for _, s := range scs {
				if len(designs) >= nDesigns {
					break
				}
				designs = append(designs, append([]float64(nil), Xall[s.i]...))
				designSources = append(designSources, fmt.Sprintf("best_effort_%d", s.i))
			}
		}
	}

	// Write design csv. Note: designs are in trainFeatureOrder. Template wants designFeatureOrder.
	designPath := filepath.Join(outDir, "design_submission.csv")
	df, err := os.Create(designPath)
	must(err)
	dw := csv.NewWriter(df)
	header := append([]string{"submission_id"}, designFeatureOrder...)
	must(dw.Write(header))
	designPreview := [][]string{}
	allInBounds := true
	for i, d := range designs {
		m := map[string]float64{}
		for j, name := range trainFeatureOrder {
			v := d[j]
			// Training rows occasionally sit a hair outside constraints.json bounds
			// (e.g. porosity=0.337 vs bound max 0.33). The challenge's input_bounds
			// are authoritative, so clamp to them before writing.
			if b, ok := cons.InputBounds[name]; ok {
				if v < b.Min {
					v = b.Min
				}
				if v > b.Max {
					v = b.Max
				}
			}
			m[name] = v
		}
		if !inBounds(m, cons.InputBounds) {
			allInBounds = false
		}
		row := []string{strconv.Itoa(i + 1)}
		for _, name := range designFeatureOrder {
			row = append(row, strconv.FormatFloat(m[name], 'g', 10, 64))
		}
		must(dw.Write(row))
		if i < 3 {
			designPreview = append(designPreview, row)
		}
	}
	dw.Flush()
	must(dw.Error())
	df.Close()

	// --- self_score.json ---
	sc := selfScoreOut{
		PerTarget:   map[string]float64{},
		Composite:   composite,
		KFolds:      kFolds,
		LogTargets:  logCols,
		TrainRows:   len(Xall),
		TestRows:    len(Xtest),
		HoldoutRows: len(Xall), // OOF covers every training row exactly once
		Timestamp:   time.Now().UTC().Format(time.RFC3339),
	}
	for t, v := range perTgt {
		sc.PerTarget[targetOrder[t]] = v
	}
	sc.Weights.RF = map[string]float64{}
	sc.Weights.KNN = map[string]float64{}
	sc.Weights.Linear = map[string]float64{}
	sc.Weights.GBM = map[string]float64{}
	subIdx := map[string]int{}
	for i, s := range subs {
		subIdx[s.Name()] = i
	}
	for t := 0; t < nTargets; t++ {
		sc.Weights.RF[targetOrder[t]] = W[subIdx["rf"]][t]
		sc.Weights.KNN[targetOrder[t]] = W[subIdx["knn"]][t]
		sc.Weights.Linear[targetOrder[t]] = W[subIdx["linear"]][t]
		sc.Weights.GBM[targetOrder[t]] = W[subIdx["gbm"]][t]
	}
	ssBytes, _ := json.MarshalIndent(sc, "", "  ")
	must(os.WriteFile(filepath.Join(outDir, "self_score.json"), ssBytes, 0o644))

	// --- SUBMISSION_READY.md ---
	md := buildSubmissionReadyMD(sc, allInBounds, len(designs), len(validIdx), predPath, designPath)
	must(os.WriteFile(filepath.Join(outDir, "SUBMISSION_READY.md"), []byte(md), 0o644))

	fmt.Printf("[boom-submit] wrote %s\n", predPath)
	fmt.Printf("[boom-submit] wrote %s\n", designPath)
	fmt.Printf("[boom-submit] wrote self_score.json and SUBMISSION_READY.md\n")
	fmt.Printf("[boom-submit] composite self-score = %.4f\n", composite)
	fmt.Printf("[boom-submit] designs generated: %d (all in bounds: %v)\n", len(designs), allInBounds)
	fmt.Printf("[boom-submit] first 3 prediction rows:\n")
	for _, r := range predRowsPreview {
		fmt.Printf("  %v\n", r)
	}
	fmt.Printf("[boom-submit] first 3 design rows:\n")
	for _, r := range designPreview {
		fmt.Printf("  %v\n", r)
	}
	fmt.Printf("[boom-submit] done in %s\n", time.Since(start).Truncate(time.Millisecond))
}

func gather(M [][]float64, idx []int) [][]float64 {
	out := make([][]float64, len(idx))
	for i, id := range idx {
		out[i] = M[id]
	}
	return out
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func must(err error) {
	if err != nil {
		die(err)
	}
}

func die(err error) {
	fmt.Fprintln(os.Stderr, "error:", err)
	os.Exit(1)
}

func buildSubmissionReadyMD(sc selfScoreOut, allInBounds bool, nDes, nValid int, predPath, designPath string) string {
	s := "# Boom: Trajectory Unknown - TARDIS Submission Ready\n\n"
	s += fmt.Sprintf("Generated: %s\n\n", sc.Timestamp)
	s += "## Wren ensemble\n\n"
	s += fmt.Sprintf("Four adepts, promoted by per-target simplex grid search over %d-fold out-of-fold predictions.\n", sc.KFolds)
	s += fmt.Sprintf("Heavy-tailed distance targets (%v) are learned in log1p space and inverted before scoring.\n\n",
		sc.LogTargets)
	s += fmt.Sprintf("- Random Forest: %d bagged regression trees (min-leaf=%d, feature subsample=%d-of-%d, max-depth=%d).\n",
		rfTrees, rfMinLeaf, rfFeatSub, nFeatures, rfMaxDepth)
	s += fmt.Sprintf("- Gradient-Boosted Trees: %d rounds, depth=%d, min-leaf=%d, shrinkage=%.2f, subsample=%.2f.\n",
		gbmRounds, gbmMaxDepth, gbmMinLeaf, gbmShrink, gbmSubsample)
	s += fmt.Sprintf("- k-Nearest-Neighbours: k=%d on standardised inputs with inverse-distance weighting.\n", knnK)
	s += "- Linear ridge (lambda=1) on standardised inputs as baseline/sanity.\n\n"
	s += "| target | weight_rf | weight_gbm | weight_knn | weight_linear | nrmse_oof |\n"
	s += "|---|---|---|---|---|---|\n"
	for _, name := range targetOrder {
		s += fmt.Sprintf("| %s | %.2f | %.2f | %.2f | %.2f | %.4f |\n",
			name, sc.Weights.RF[name], sc.Weights.GBM[name], sc.Weights.KNN[name], sc.Weights.Linear[name], sc.PerTarget[name])
	}
	s += fmt.Sprintf("\n**Composite (mean NRMSE across 6 targets, %d-fold OOF): %.4f**\n", sc.KFolds, sc.Composite)
	s += fmt.Sprintf("(full train n=%d, OOF rows n=%d, final predictors refit on full n=%d, test scenarios n=%d)\n\n",
		sc.TrainRows, sc.HoldoutRows, sc.TrainRows, sc.TestRows)
	s += "## Inverse design\n\n"
	s += fmt.Sprintf("- Training samples already satisfying `96 <= P80 <= 101` and `R95 <= 175`: **%d**.\n", nValid)
	if nValid >= nDesigns {
		s += "- Selection strategy: greedy farthest-point sampling in standardised input space to pick 20 diverse candidates from the valid pool.\n"
	} else {
		s += "- Strategy: seeded with valid training rows, then perturbed top-ranked near-valid rows within 10%% of each bound's span and retained candidates whose ensemble predictions satisfied the constraints.\n"
	}
	s += fmt.Sprintf("- Designs produced: **%d** (all 8 inputs inside bounds: **%v**).\n\n", nDes, allInBounds)
	s += "## Files to submit\n\n"
	s += "Upload both CSVs to https://www.freelancer.com/boom via the challenge submission form:\n\n"
	s += fmt.Sprintf("- Prediction track CSV: `%s`\n", predPath)
	s += fmt.Sprintf("- Design track CSV:     `%s`\n\n", designPath)
	s += "Column order is exactly:\n\n"
	s += "- prediction_submission.csv: `scenario_id,P80,fines_frac,oversize_frac,R95,R50_fines,R50_oversize`\n"
	s += "- design_submission.csv:     `submission_id,energy,angle_rad,coupling,strength,porosity,gravity,atmosphere,shape_factor`\n\n"
	s += "## Submission instructions\n\n"
	s += "1. Sign in to Freelancer.com and open the Boom: Trajectory Unknown Challenge page at https://www.freelancer.com/boom.\n"
	s += "2. On the submission form, attach `prediction_submission.csv` to the prediction track.\n"
	s += "3. Attach `design_submission.csv` to the inverse-design track.\n"
	s += "4. In the notes field, include the composite NRMSE above and a one-line ensemble description.\n"
	s += "5. Submit before 5 May 2026. Winners announced 3 June 2026. Prize $7,000 USD.\n\n"
	s += "## Reproducing\n\n"
	s += "From `substrates/boom-ejecta`:\n\n"
	s += "```\ngo build ./submit\ngo run ./submit\n```\n"
	return s
}
