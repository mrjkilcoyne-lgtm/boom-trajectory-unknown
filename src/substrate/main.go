package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"
)

// palaceURL is the MemPalace HTTP API endpoint.
var palaceURL = envOr("MEMPALACE_URL", "http://mempalace.substrates.svc.cluster.local:8095")

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// palaceStore fires a non-blocking POST to /store.
func palaceStore(room, id, content string, meta map[string]string) {
	go func() {
		body, _ := json.Marshal(map[string]any{
			"wing": "tardis", "room": room, "id": id,
			"content": content, "metadata": meta,
		})
		resp, err := http.Post(palaceURL+"/store", "application/json", bytes.NewReader(body))
		if err != nil {
			log.Printf("[palace] store error: %v", err)
			return
		}
		resp.Body.Close()
	}()
}

// palaceKG fires a non-blocking POST to /kg/add.
func palaceKG(subject, predicate, object string, confidence float64) {
	go func() {
		body, _ := json.Marshal(map[string]any{
			"subject": subject, "predicate": predicate, "object": object,
			"valid_from": time.Now().UTC().Format(time.RFC3339),
			"confidence": confidence,
		})
		resp, err := http.Post(palaceURL+"/kg/add", "application/json", bytes.NewReader(body))
		if err != nil {
			log.Printf("[palace] kg/add error: %v", err)
			return
		}
		resp.Body.Close()
	}()
}

var (
	dataset     *Dataset
	mu          sync.RWMutex
	predictions map[int]EjectaOutcome // scenario_id -> predicted outcome
)

func main() {
	dataDir := os.Getenv("DATA_DIR")
	if dataDir == "" {
		dataDir = "data"
	}

	var err error
	dataset, err = LoadDataset(dataDir)
	if err != nil {
		log.Fatalf("Failed to load dataset: %v", err)
	}

	predictions = make(map[int]EjectaOutcome)

	log.Printf("Boom Ejecta substrate loaded: %d training samples, %d test scenarios",
		len(dataset.Train), len(dataset.Test))

	port := os.Getenv("PORT")
	if port == "" {
		port = "8091"
	}

	http.HandleFunc("GET /healthz", handleHealth)
	http.HandleFunc("GET /info", handleInfo)
	http.HandleFunc("GET /train", handleGetTrain)
	http.HandleFunc("GET /train/{id}", handleGetTrainSample)
	http.HandleFunc("GET /test", handleGetTest)
	http.HandleFunc("GET /test/{id}", handleGetTestScenario)
	http.HandleFunc("GET /constraints", handleGetConstraints)
	http.HandleFunc("GET /stats", handleStats)
	http.HandleFunc("POST /predict", handlePredict)
	http.HandleFunc("POST /predict/batch", handlePredictBatch)
	http.HandleFunc("POST /score", handleScore)
	http.HandleFunc("POST /inverse/validate", handleInverseValidate)
	http.HandleFunc("POST /reset", handleReset)

	log.Printf("Boom Ejecta substrate listening on :%s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]string{"status": "ok", "substrate": "boom-ejecta", "tier": "2"})
}

func handleInfo(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]interface{}{
		"name":        "boom-ejecta",
		"tier":        2,
		"description": "Boom: Trajectory Unknown — asteroid impact ejecta prediction. Tier 2: real physics, partial observability, non-linear multi-output.",
		"challenge":   "Predict 6 ejecta outcomes from 8 partially-observed impact parameters. Plus inverse design: find inputs that produce constrained outputs.",
		"train_size":  len(dataset.Train),
		"test_size":   len(dataset.Test),
		"inputs":      []string{"porosity", "atmosphere", "gravity", "coupling", "strength", "shape_factor", "energy", "angle_rad"},
		"outputs":     []string{"P80", "fines_frac", "oversize_frac", "R95", "R50_fines", "R50_oversize"},
		"hidden_vars": "The 8 parameters only PARTIALLY describe each scenario. Hidden variables exist.",
		"scoring":     "Normalised RMSE per target, averaged across 6 targets. Score = 1 - NRMSE.",
	})
}

// GET /train — return training data. Supports ?limit=N&offset=M for pagination.
func handleGetTrain(w http.ResponseWriter, r *http.Request) {
	limit := queryInt(r, "limit", len(dataset.Train))
	offset := queryInt(r, "offset", 0)

	if offset >= len(dataset.Train) {
		writeJSON(w, map[string]interface{}{"samples": []TrainingSample{}, "total": len(dataset.Train)})
		return
	}

	end := offset + limit
	if end > len(dataset.Train) {
		end = len(dataset.Train)
	}

	writeJSON(w, map[string]interface{}{
		"samples": dataset.Train[offset:end],
		"total":   len(dataset.Train),
		"offset":  offset,
		"limit":   limit,
	})
}

// GET /train/{id} — return a single training sample with its outcome.
func handleGetTrainSample(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.Atoi(r.PathValue("id"))
	if err != nil || id < 0 || id >= len(dataset.Train) {
		http.Error(w, "invalid sample id", http.StatusBadRequest)
		return
	}
	writeJSON(w, dataset.Train[id])
}

// GET /test — return test scenarios (no labels). Supports ?limit=N&offset=M.
func handleGetTest(w http.ResponseWriter, r *http.Request) {
	limit := queryInt(r, "limit", len(dataset.Test))
	offset := queryInt(r, "offset", 0)

	if offset >= len(dataset.Test) {
		writeJSON(w, map[string]interface{}{"scenarios": []ImpactScenario{}, "total": len(dataset.Test)})
		return
	}

	end := offset + limit
	if end > len(dataset.Test) {
		end = len(dataset.Test)
	}

	writeJSON(w, map[string]interface{}{
		"scenarios": dataset.Test[offset:end],
		"total":     len(dataset.Test),
		"offset":    offset,
		"limit":     limit,
	})
}

// GET /test/{id} — return a single test scenario.
func handleGetTestScenario(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.Atoi(r.PathValue("id"))
	if err != nil || id < 0 || id >= len(dataset.Test) {
		http.Error(w, "invalid scenario id", http.StatusBadRequest)
		return
	}
	writeJSON(w, dataset.Test[id])
}

// GET /constraints — return inverse design constraints and input bounds.
func handleGetConstraints(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, dataset.Constraints)
}

// GET /stats — return dataset statistics and current prediction state.
func handleStats(w http.ResponseWriter, r *http.Request) {
	mu.RLock()
	predCount := len(predictions)
	mu.RUnlock()

	writeJSON(w, map[string]interface{}{
		"train_samples":       len(dataset.Train),
		"test_scenarios":      len(dataset.Test),
		"predictions_made":    predCount,
		"predictions_pending": len(dataset.Test) - predCount,
		"coverage":            fmt.Sprintf("%.1f%%", float64(predCount)/float64(len(dataset.Test))*100),
	})
}

// POST /predict — submit a single prediction for a test scenario.
// Body: {"scenario_id": 0, "prediction": {...}, "confidence": 0.7}
type PredictRequest struct {
	ScenarioID int           `json:"scenario_id"`
	Prediction EjectaOutcome `json:"prediction"`
	Confidence float64       `json:"confidence"`
}

func handlePredict(w http.ResponseWriter, r *http.Request) {
	var req PredictRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}

	if req.ScenarioID < 0 || req.ScenarioID >= len(dataset.Test) {
		http.Error(w, fmt.Sprintf("scenario_id %d out of range [0, %d)", req.ScenarioID, len(dataset.Test)), http.StatusBadRequest)
		return
	}

	mu.Lock()
	predictions[req.ScenarioID] = req.Prediction
	mu.Unlock()

	result := map[string]interface{}{
		"status":      "accepted",
		"scenario_id": req.ScenarioID,
		"confidence":  req.Confidence,
	}
	writeJSON(w, result)

	// Persist prediction to palace
	predJSON, _ := json.Marshal(result)
	palaceStore("boom-ejecta",
		fmt.Sprintf("be-pred-%d-%d", req.ScenarioID, time.Now().UnixMilli()),
		string(predJSON),
		map[string]string{"type": "prediction", "substrate": "boom-ejecta"},
	)
	palaceKG("boom-ejecta", "predicted_scenario", fmt.Sprintf("%d", req.ScenarioID), req.Confidence)
}

// POST /predict/batch — submit predictions for multiple test scenarios at once.
// Body: [{"scenario_id": 0, "prediction": {...}, "confidence": 0.7}, ...]
func handlePredictBatch(w http.ResponseWriter, r *http.Request) {
	var reqs []PredictRequest
	if err := json.NewDecoder(r.Body).Decode(&reqs); err != nil {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}

	mu.Lock()
	accepted := 0
	for _, req := range reqs {
		if req.ScenarioID >= 0 && req.ScenarioID < len(dataset.Test) {
			predictions[req.ScenarioID] = req.Prediction
			accepted++
		}
	}
	mu.Unlock()

	batchResult := map[string]interface{}{
		"status":   "accepted",
		"accepted": accepted,
		"total":    len(reqs),
	}
	writeJSON(w, batchResult)

	// Persist batch prediction to palace
	batchJSON, _ := json.Marshal(batchResult)
	palaceStore("boom-ejecta",
		fmt.Sprintf("be-batch-%d-%d", accepted, time.Now().UnixMilli()),
		string(batchJSON),
		map[string]string{"type": "prediction_batch", "substrate": "boom-ejecta"},
	)
	palaceKG("boom-ejecta", "batch_predicted", fmt.Sprintf("%d_scenarios", accepted), 1.0)
}

// POST /score — score the current predictions against a held-out validation set.
// For training substrate purposes, we score against a random 20% holdout from
// the training data. The real test labels are unknown.
// Body: {"mode": "holdout"} or {"mode": "cross_validate", "folds": 5}
type ScoreRequest struct {
	Mode  string `json:"mode"`  // "holdout" or "cross_validate"
	Folds int    `json:"folds"` // for cross_validate
}

func handleScore(w http.ResponseWriter, r *http.Request) {
	var req ScoreRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		req.Mode = "holdout"
	}

	// For the substrate, we use the training data to score.
	// Split: last 20% of training data as validation.
	splitIdx := int(float64(len(dataset.Train)) * 0.8)
	valData := dataset.Train[splitIdx:]

	mu.RLock()
	predCount := len(predictions)
	mu.RUnlock()

	if predCount == 0 {
		// If no test predictions, score using training validation set.
		// The TARDIS must have submitted predictions on the validation indices.
		writeJSON(w, map[string]interface{}{
			"error":   "no predictions submitted yet",
			"hint":    "Submit predictions via POST /predict or POST /predict/batch. For self-evaluation, predict on training scenarios [" + fmt.Sprintf("%d", splitIdx) + ", " + fmt.Sprintf("%d", len(dataset.Train)) + ") and POST /score/self.",
			"val_size": len(valData),
		})
		return
	}

	// Score whatever predictions have been submitted against test set.
	// Since we don't have test labels, return coverage stats.
	writeJSON(w, map[string]interface{}{
		"predictions_submitted": predCount,
		"test_total":            len(dataset.Test),
		"coverage":              fmt.Sprintf("%.1f%%", float64(predCount)/float64(len(dataset.Test))*100),
		"note":                  "Test labels are hidden. Use POST /score/self with training holdout predictions for self-evaluation.",
	})
}

// POST /score/self — the TARDIS submits predictions on training data indices
// to measure her own accuracy before attempting the blind test set.
// Body: [{"scenario_id": 2344, "prediction": {...}, "confidence": 0.7}, ...]
// scenario_id here refers to the training sample index.
func init() {
	http.HandleFunc("POST /score/self", handleScoreSelf)
}

func handleScoreSelf(w http.ResponseWriter, r *http.Request) {
	var reqs []PredictRequest
	if err := json.NewDecoder(r.Body).Decode(&reqs); err != nil {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}

	if len(reqs) == 0 {
		http.Error(w, "empty predictions", http.StatusBadRequest)
		return
	}

	preds := make([]EjectaOutcome, 0, len(reqs))
	truth := make([]EjectaOutcome, 0, len(reqs))
	confidences := make([]float64, 0, len(reqs))

	for _, req := range reqs {
		if req.ScenarioID < 0 || req.ScenarioID >= len(dataset.Train) {
			continue
		}
		preds = append(preds, req.Prediction)
		truth = append(truth, dataset.Train[req.ScenarioID].Outcome)
		confidences = append(confidences, req.Confidence)
	}

	if len(preds) == 0 {
		http.Error(w, "no valid predictions matched training indices", http.StatusBadRequest)
		return
	}

	score, err := ScoreForwardPredictions(preds, truth)
	if err != nil {
		http.Error(w, "scoring error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Compute calibration: mean absolute difference between confidence and accuracy.
	var calibSum float64
	for i, conf := range confidences {
		// Per-prediction accuracy: 1 - mean absolute error across targets (rough proxy)
		predAcc := perPredictionAccuracy(preds[i], truth[i])
		calibSum += abs(conf - predAcc)
	}
	calibration := calibSum / float64(len(confidences))

	scoreResult := map[string]interface{}{
		"forward_score":     score,
		"calibration_error": calibration,
		"calibration_note":  "Lower is better. 0 = perfectly calibrated confidence.",
		"samples_scored":    len(preds),
		"samples_submitted": len(reqs),
	}
	writeJSON(w, scoreResult)

	// Persist self-evaluation to palace
	scoreJSON, _ := json.Marshal(scoreResult)
	palaceStore("scoring",
		fmt.Sprintf("be-score-self-%d", time.Now().UnixMilli()),
		string(scoreJSON),
		map[string]string{"type": "score", "substrate": "boom-ejecta"},
	)
	palaceKG("boom-ejecta", "self_score", fmt.Sprintf("%.4f", score.Composite), score.Composite)
}

// POST /inverse/validate — validate proposed inverse designs against input bounds.
// Body: [{"submission_id": 0, "energy": 2.5, ...}, ...]
func handleInverseValidate(w http.ResponseWriter, r *http.Request) {
	var designs []InverseDesign
	if err := json.NewDecoder(r.Body).Decode(&designs); err != nil {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}

	score := ValidateInverseDesigns(designs, dataset.Constraints)
	writeJSON(w, score)
}

// POST /reset — clear all predictions (start fresh).
func handleReset(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	predictions = make(map[int]EjectaOutcome)
	mu.Unlock()
	writeJSON(w, map[string]string{"status": "reset", "message": "All predictions cleared."})
}

// perPredictionAccuracy computes a rough per-prediction accuracy score.
func perPredictionAccuracy(pred, truth EjectaOutcome) float64 {
	targets := []struct{ p, t float64 }{
		{pred.P80, truth.P80},
		{pred.FinesFrac, truth.FinesFrac},
		{pred.OversizeFrac, truth.OversizeFrac},
		{pred.R95, truth.R95},
		{pred.R50Fines, truth.R50Fines},
		{pred.R50Oversize, truth.R50Oversize},
	}

	var totalErr float64
	for _, tt := range targets {
		denom := abs(tt.t)
		if denom < 1e-6 {
			denom = 1.0
		}
		relErr := abs(tt.p-tt.t) / denom
		if relErr > 1.0 {
			relErr = 1.0
		}
		totalErr += relErr
	}
	acc := 1.0 - (totalErr / float64(len(targets)))
	if acc < 0 {
		acc = 0
	}
	return acc
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func queryInt(r *http.Request, key string, defaultVal int) int {
	v := r.URL.Query().Get(key)
	if v == "" {
		return defaultVal
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return defaultVal
	}
	return n
}

func writeJSON(w http.ResponseWriter, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}
