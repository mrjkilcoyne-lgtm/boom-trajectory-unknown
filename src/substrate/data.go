// Package main provides the Boom: Trajectory Unknown substrate for TARDIS
// training. This is a Tier 2 (chaotic-but-deterministic with hidden variables)
// asteroid impact ejecta prediction challenge.
package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strconv"
)

// ImpactScenario describes the 8 observable parameters of an impact event.
type ImpactScenario struct {
	ScenarioID  int     `json:"scenario_id"`
	Porosity    float64 `json:"porosity"`
	Atmosphere  float64 `json:"atmosphere"`
	Gravity     float64 `json:"gravity"`
	Coupling    float64 `json:"coupling"`
	Strength    float64 `json:"strength"`
	ShapeFactor float64 `json:"shape_factor"`
	Energy      float64 `json:"energy"`
	AngleRad    float64 `json:"angle_rad"`
}

// EjectaOutcome describes the 6 statistical measures of ejecta from an impact.
type EjectaOutcome struct {
	P80          float64 `json:"P80"`
	FinesFrac    float64 `json:"fines_frac"`
	OversizeFrac float64 `json:"oversize_frac"`
	R95          float64 `json:"R95"`
	R50Fines     float64 `json:"R50_fines"`
	R50Oversize  float64 `json:"R50_oversize"`
}

// TrainingSample pairs an impact scenario with its known ejecta outcome.
type TrainingSample struct {
	Scenario ImpactScenario `json:"scenario"`
	Outcome  EjectaOutcome  `json:"outcome"`
}

// Constraints defines the inverse design challenge bounds.
type Constraints struct {
	OutputConstraints OutputConstraints         `json:"constraints"`
	InputBounds       map[string]MinMax         `json:"input_bounds"`
}

type OutputConstraints struct {
	P80Min float64 `json:"p80_min"`
	P80Max float64 `json:"p80_max"`
	R95Max float64 `json:"r95_max"`
}

type MinMax struct {
	Min float64 `json:"min"`
	Max float64 `json:"max"`
}

// InverseDesign is a proposed impact scenario for the inverse challenge.
type InverseDesign struct {
	SubmissionID int     `json:"submission_id"`
	Energy       float64 `json:"energy"`
	AngleRad     float64 `json:"angle_rad"`
	Coupling     float64 `json:"coupling"`
	Strength     float64 `json:"strength"`
	Porosity     float64 `json:"porosity"`
	Gravity      float64 `json:"gravity"`
	Atmosphere   float64 `json:"atmosphere"`
	ShapeFactor  float64 `json:"shape_factor"`
}

// Dataset holds all loaded challenge data.
type Dataset struct {
	Train       []TrainingSample
	Test        []ImpactScenario
	Constraints Constraints
}

// LoadDataset loads training, test, and constraint data from the data directory.
func LoadDataset(dataDir string) (*Dataset, error) {
	train, err := loadTrainData(dataDir)
	if err != nil {
		return nil, fmt.Errorf("loading train data: %w", err)
	}

	test, err := loadTestData(dataDir)
	if err != nil {
		return nil, fmt.Errorf("loading test data: %w", err)
	}

	constraints, err := loadConstraints(dataDir)
	if err != nil {
		return nil, fmt.Errorf("loading constraints: %w", err)
	}

	return &Dataset{
		Train:       train,
		Test:        test,
		Constraints: constraints,
	}, nil
}

func loadTrainData(dataDir string) ([]TrainingSample, error) {
	scenarios, err := loadScenarios(dataDir + "/train.csv")
	if err != nil {
		return nil, err
	}

	labels, err := loadLabels(dataDir + "/train_labels.csv")
	if err != nil {
		return nil, err
	}

	if len(scenarios) != len(labels) {
		return nil, fmt.Errorf("scenario count %d != label count %d", len(scenarios), len(labels))
	}

	samples := make([]TrainingSample, len(scenarios))
	for i := range scenarios {
		samples[i] = TrainingSample{
			Scenario: scenarios[i],
			Outcome:  labels[i],
		}
	}
	return samples, nil
}

func loadScenarios(path string) ([]ImpactScenario, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	records, err := r.ReadAll()
	if err != nil {
		return nil, err
	}

	// Header: porosity,atmosphere,gravity,coupling,strength,shape_factor,energy,angle_rad
	scenarios := make([]ImpactScenario, 0, len(records)-1)
	for i, row := range records[1:] {
		s := ImpactScenario{ScenarioID: i}
		s.Porosity, _ = strconv.ParseFloat(row[0], 64)
		s.Atmosphere, _ = strconv.ParseFloat(row[1], 64)
		s.Gravity, _ = strconv.ParseFloat(row[2], 64)
		s.Coupling, _ = strconv.ParseFloat(row[3], 64)
		s.Strength, _ = strconv.ParseFloat(row[4], 64)
		s.ShapeFactor, _ = strconv.ParseFloat(row[5], 64)
		s.Energy, _ = strconv.ParseFloat(row[6], 64)
		s.AngleRad, _ = strconv.ParseFloat(row[7], 64)
		scenarios = append(scenarios, s)
	}
	return scenarios, nil
}

func loadLabels(path string) ([]EjectaOutcome, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	records, err := r.ReadAll()
	if err != nil {
		return nil, err
	}

	// Header: P80,fines_frac,oversize_frac,R95,R50_fines,R50_oversize
	labels := make([]EjectaOutcome, 0, len(records)-1)
	for _, row := range records[1:] {
		o := EjectaOutcome{}
		o.P80, _ = strconv.ParseFloat(row[0], 64)
		o.FinesFrac, _ = strconv.ParseFloat(row[1], 64)
		o.OversizeFrac, _ = strconv.ParseFloat(row[2], 64)
		o.R95, _ = strconv.ParseFloat(row[3], 64)
		o.R50Fines, _ = strconv.ParseFloat(row[4], 64)
		o.R50Oversize, _ = strconv.ParseFloat(row[5], 64)
		labels = append(labels, o)
	}
	return labels, nil
}

func loadTestData(dataDir string) ([]ImpactScenario, error) {
	return loadScenarios(dataDir + "/test.csv")
}

func loadConstraints(dataDir string) (Constraints, error) {
	f, err := os.Open(dataDir + "/constraints.json")
	if err != nil {
		return Constraints{}, err
	}
	defer f.Close()

	var c Constraints
	if err := json.NewDecoder(f).Decode(&c); err != nil {
		return Constraints{}, err
	}
	return c, nil
}

// ScoreForwardPrediction scores a set of ejecta predictions against ground truth.
// Uses normalised RMSE per output column, then averages across all 6 targets.
// Returns per-target scores and composite. Lower NRMSE = better; score is 1-NRMSE clamped to [0,1].
type ForwardScore struct {
	PerTarget  map[string]float64 `json:"per_target"`
	Composite  float64            `json:"composite"`
	NRMSE      map[string]float64 `json:"nrmse"`
	SampleSize int                `json:"sample_size"`
}

func ScoreForwardPredictions(predictions []EjectaOutcome, truth []EjectaOutcome) (ForwardScore, error) {
	if len(predictions) != len(truth) {
		return ForwardScore{}, fmt.Errorf("prediction count %d != truth count %d", len(predictions), len(truth))
	}

	n := float64(len(predictions))
	targets := []string{"P80", "fines_frac", "oversize_frac", "R95", "R50_fines", "R50_oversize"}

	getVal := func(o EjectaOutcome, target string) float64 {
		switch target {
		case "P80":
			return o.P80
		case "fines_frac":
			return o.FinesFrac
		case "oversize_frac":
			return o.OversizeFrac
		case "R95":
			return o.R95
		case "R50_fines":
			return o.R50Fines
		case "R50_oversize":
			return o.R50Oversize
		}
		return 0
	}

	nrmse := make(map[string]float64)
	perTarget := make(map[string]float64)

	for _, t := range targets {
		var sumSqErr, minVal, maxVal float64
		minVal = math.Inf(1)
		maxVal = math.Inf(-1)

		for i := range predictions {
			pred := getVal(predictions[i], t)
			actual := getVal(truth[i], t)
			diff := pred - actual
			sumSqErr += diff * diff
			if actual < minVal {
				minVal = actual
			}
			if actual > maxVal {
				maxVal = actual
			}
		}

		rmse := math.Sqrt(sumSqErr / n)
		rang := maxVal - minVal
		if rang == 0 {
			rang = 1
		}
		nrmseVal := rmse / rang
		nrmse[t] = nrmseVal
		// Score: 1 - NRMSE, clamped to [0, 1]
		score := 1.0 - nrmseVal
		if score < 0 {
			score = 0
		}
		if score > 1 {
			score = 1
		}
		perTarget[t] = score
	}

	var compositeSum float64
	for _, s := range perTarget {
		compositeSum += s
	}
	composite := compositeSum / float64(len(targets))

	return ForwardScore{
		PerTarget:  perTarget,
		Composite:  composite,
		NRMSE:      nrmse,
		SampleSize: len(predictions),
	}, nil
}

// ValidateInverseDesign checks whether a set of proposed impact scenarios
// satisfies the inverse design constraints. Returns per-scenario pass/fail
// and an overall score (fraction of valid designs).
type InverseScore struct {
	Results     []InverseResult `json:"results"`
	ValidCount  int             `json:"valid_count"`
	TotalCount  int             `json:"total_count"`
	Score       float64         `json:"score"`
}

type InverseResult struct {
	SubmissionID   int     `json:"submission_id"`
	Valid          bool    `json:"valid"`
	InputsInBounds bool    `json:"inputs_in_bounds"`
	Reason         string  `json:"reason,omitempty"`
}

func ValidateInverseDesigns(designs []InverseDesign, constraints Constraints) InverseScore {
	results := make([]InverseResult, len(designs))
	validCount := 0

	for i, d := range designs {
		r := InverseResult{SubmissionID: d.SubmissionID, Valid: true, InputsInBounds: true}

		// Check input bounds
		checks := map[string]float64{
			"energy": d.Energy, "angle_rad": d.AngleRad, "coupling": d.Coupling,
			"strength": d.Strength, "porosity": d.Porosity, "gravity": d.Gravity,
			"atmosphere": d.Atmosphere, "shape_factor": d.ShapeFactor,
		}

		for name, val := range checks {
			bounds, ok := constraints.InputBounds[name]
			if ok && (val < bounds.Min || val > bounds.Max) {
				r.Valid = false
				r.InputsInBounds = false
				r.Reason = fmt.Sprintf("%s=%.4f outside bounds [%.4f, %.4f]", name, val, bounds.Min, bounds.Max)
				break
			}
		}

		// Note: we can only validate input bounds here. The output constraints
		// (P80 96-101mm, R95 <= 175m) require running the forward model, which
		// the TARDIS must do herself. The full scoring uses her forward predictions
		// on these designs to check output validity.

		if r.Valid {
			validCount++
		}
		results[i] = r
	}

	score := 0.0
	if len(designs) > 0 {
		score = float64(validCount) / float64(len(designs))
	}

	return InverseScore{
		Results:    results,
		ValidCount: validCount,
		TotalCount: len(designs),
		Score:      score,
	}
}
