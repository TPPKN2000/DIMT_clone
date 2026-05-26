import os
import time
import threading
import mlflow
import psutil
import torch
import numpy as np
from collections import deque
from Levenshtein import distance as lev_distance
from dotenv import load_dotenv

load_dotenv()

# Evidently AI for data/model drift monitoring
try:
    import pandas as pd
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset
    from evidently.ui.workspace import CloudWorkspace
    EVIDENTLY_AVAILABLE = True
except ImportError as e:
    EVIDENTLY_AVAILABLE = False
    # Only print if not already printing from another module
    if os.environ.get("DEBUG_EVAL") == "1":
        print(f"[Eval] Evidently AI not available (ImportError: {e}). Drift monitoring disabled.")


class Evaluator:
    def __init__(self):
        mlflow.set_tracking_uri("sqlite:///mlruns.db")
        mlflow.set_experiment("agentic_pipeline_eval")
        self.runs = {}
        # Production metric accumulators
        self._latencies = deque(maxlen=1000)
        self._total_requests = 0
        self._failed_requests = 0
        self._total_tasks = 0
        self._successful_tasks = 0
        self._total_tokens = 0
        self._lock = threading.Lock()
        # Q12: Evidently drift reference data
        self._translation_history = deque(maxlen=500)
        self._evidently_min_samples = int(os.environ.get("EVIDENTLY_MIN_SAMPLES", "10"))

    def _get_evidently_api_key(self) -> str | None:
        """Read Evidently Cloud token, keeping the old project typo as a fallback."""
        return (
            os.environ.get("EVIDENTLY_API_KEY")
            or os.environ.get("EVIDENTLY_TRACE_COLLECTOR_API_KEY")
            or os.environ.get("EVIDENT_AI_API_KEY")
        )

    def _extract_dataset_drift(self, snapshot: object) -> bool:
        """Evidently 0.7 returns a Snapshot, not the old report dict format."""
        try:
            result = snapshot.dict()
            for metric in result.get("metrics", []):
                if metric.get("metric_name", "").startswith("DriftedColumnsCount"):
                    value = metric.get("value", {})
                    share = float(value.get("share", 0.0))
                    threshold = float(os.environ.get("EVIDENTLY_DRIFT_SHARE", "0.5"))
                    return share >= threshold
        except Exception:
            pass
        return False

    def start_inference(self, doc_id: str):
        self.runs[doc_id] = {"start_time": time.time()}
        with self._lock:
            self._total_requests += 1
            self._total_tasks += 1
        print(f"[Eval] Inference started for {doc_id}")

    def end_inference(self, doc_id: str):
        if doc_id in self.runs:
            latency = time.time() - self.runs[doc_id]["start_time"]
            self.runs[doc_id]["inference_time"] = latency
            with self._lock:
                self._latencies.append(latency)
                self._successful_tasks += 1
            print(f"[Eval] Inference ended for {doc_id} in {latency:.1f}s")

    def record_failure(self, doc_id: str):
        with self._lock:
            self._failed_requests += 1
        print(f"[Eval] ❌ Failure recorded for {doc_id}")

    def record_tokens(self, count: int):
        with self._lock:
            self._total_tokens += count

    def _get_system_metrics(self) -> dict:
        """Collect GPU/CPU utilization."""
        metrics = {"cpu_percent": psutil.cpu_percent(interval=0.1)}
        if torch.cuda.is_available():
            try:
                metrics["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
                metrics["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
                metrics["gpu_utilization_percent"] = (
                    torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_mem * 100
                )
            except Exception:
                pass
        return metrics

    def _get_production_metrics(self) -> dict:
        """Compute production metrics from accumulators."""
        with self._lock:
            p95_latency = float(np.percentile(list(self._latencies), 95)) if self._latencies else 0.0
            error_rate = (self._failed_requests / self._total_requests) if self._total_requests > 0 else 0.0
            success_rate = (self._successful_tasks / self._total_tasks) if self._total_tasks > 0 else 0.0
            tokens_per_request = (self._total_tokens / self._total_requests) if self._total_requests > 0 else 0.0
        return {
            "p95_latency": p95_latency,
            "error_rate": error_rate,
            "success_rate": success_rate,
            "tokens_per_request": tokens_per_request,
        }

    def log_metrics(self, doc_id: str, original_md: str, modified_md: str,
                    user_rating: int, download: bool, time_consumed: float):
        edit_dist = lev_distance(original_md, modified_md)
        inference_time = self.runs.get(doc_id, {}).get("inference_time", 0.0)
        prod_metrics = self._get_production_metrics()
        sys_metrics = self._get_system_metrics()

        # Q12: Track translation quality for drift detection
        self._translation_history.append({
            "doc_id": doc_id,
            "edit_distance": edit_dist,
            "user_rating": user_rating,
            "inference_time": inference_time,
            "text_length": len(original_md),
            "time_consumed": time_consumed,
        })

        with mlflow.start_run(run_name=f"eval_{doc_id}"):
            # Core metrics
            mlflow.log_metric("inference_time", inference_time)
            mlflow.log_metric("user_rating", user_rating)
            mlflow.log_metric("edit_distance", edit_dist)
            mlflow.log_metric("time_consumed_to_modify", time_consumed)
            mlflow.log_param("downloaded", download)
            # Production metrics
            mlflow.log_metric("p95_latency", prod_metrics["p95_latency"])
            mlflow.log_metric("error_rate", prod_metrics["error_rate"])
            mlflow.log_metric("success_rate", prod_metrics["success_rate"])
            mlflow.log_metric("tokens_per_request", prod_metrics["tokens_per_request"])
            # System metrics
            mlflow.log_metric("cpu_percent", sys_metrics.get("cpu_percent", 0))
            if "gpu_utilization_percent" in sys_metrics:
                mlflow.log_metric("gpu_utilization_percent", sys_metrics["gpu_utilization_percent"])
                mlflow.log_metric("gpu_memory_allocated_mb", sys_metrics["gpu_memory_allocated_mb"])

        print(f"[Eval] ✅ Metrics logged for {doc_id}. Edit dist: {edit_dist}, "
              f"Rating: {user_rating}, P95: {prod_metrics['p95_latency']:.2f}s")

        # Q12: Run Evidently drift check if enough data
        drift_report = self._check_drift()

        return {
            "edit_distance": edit_dist,
            "inference_time": inference_time,
            "drift_detected": drift_report,
            **prod_metrics,
        }

    def _check_drift(self) -> dict | None:
        """Q12: Evidently AI data drift detection on translation quality metrics."""
        if not EVIDENTLY_AVAILABLE:
            return {"drift_detected": False, "sample_count": 0, "message": "Evidently not installed"}

        sample_count = len(self._translation_history)

        # Always try to push latest snapshot to Evidently Cloud
        api_key = self._get_evidently_api_key()
        project_id = os.environ.get("EVIDENTLY_PROJECT_ID")

        import warnings

        if sample_count < self._evidently_min_samples:
            # Push single-row snapshot for tracing even with few samples
            if api_key and project_id and sample_count > 0:
                try:
                    latest = self._translation_history[-1]
                    columns = ["edit_distance", "user_rating", "inference_time", "text_length"]
                    df = pd.DataFrame([latest])
                    data_definition = DataDefinition(numerical_columns=columns)
                    current_ds = Dataset.from_pandas(df[columns], data_definition=data_definition)
                    report = Report(metrics=[DataDriftPreset()])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        snapshot = report.run(current_ds, current_ds)

                    workspace = CloudWorkspace(
                        token=api_key,
                        url=os.environ.get("EVIDENTLY_CLOUD_URL", "https://app.evidently.cloud"),
                    )
                    workspace.add_run(
                        project_id,
                        snapshot,
                        include_data=False,
                        name=f"translation_trace_{int(time.time())}",
                    )
                    print(f"[Eval] Trace snapshot pushed to Evidently Cloud (samples={sample_count})")
                except Exception as cloud_err:
                    print(f"[Eval] Failed to push trace to Evidently Cloud: {cloud_err}")

            return {"drift_detected": False, "sample_count": sample_count,
                    "message": f"Need {self._evidently_min_samples} samples for drift detection"}

        try:
            df = pd.DataFrame(list(self._translation_history))
            midpoint = len(df) // 2
            reference = df.iloc[:midpoint]
            current = df.iloc[midpoint:]

            columns = ["edit_distance", "user_rating", "inference_time", "text_length"]
            data_definition = DataDefinition(numerical_columns=columns)
            reference_dataset = Dataset.from_pandas(
                reference[columns],
                data_definition=data_definition,
            )
            current_dataset = Dataset.from_pandas(
                current[columns],
                data_definition=data_definition,
            )

            report = Report(metrics=[DataDriftPreset()])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                snapshot = report.run(current_dataset, reference_dataset)
            drift_detected = self._extract_dataset_drift(snapshot)

            # Cloud Integration
            if api_key and project_id:
                try:
                    workspace = CloudWorkspace(
                        token=api_key,
                        url=os.environ.get("EVIDENTLY_CLOUD_URL", "https://app.evidently.cloud"),
                    )
                    workspace.add_run(
                        project_id,
                        snapshot,
                        include_data=False,
                        name=f"translation_drift_{int(time.time())}",
                    )
                    print(f"[Eval] Snapshot pushed to Evidently Cloud (Project: {project_id})")
                except Exception as cloud_err:
                    print(f"[Eval] Failed to push to Evidently Cloud: {cloud_err}")

            os.makedirs("output", exist_ok=True)
            snapshot.save_html("output/evidently_drift_report.html")

            if drift_detected:
                print("[Eval] DATA DRIFT DETECTED - translation quality may have changed!")

            return {"drift_detected": drift_detected, "sample_count": len(df)}
        except Exception as e:
            print(f"[Eval] Evidently error: {e}")
            return {"drift_detected": False, "sample_count": sample_count, "error": str(e)}

