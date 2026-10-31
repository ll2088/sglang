//! The serving benchmark client, ported from
//! `python/sglang/benchmark/serving.py`.
//!
//! Why a port: the Python client drives every request and parses every
//! response stream on one asyncio thread. Past a few thousand concurrent
//! streams that thread, not the server, sets the measured throughput, and the
//! benchmark reports the client's limit. Here each request is a Tokio task, so
//! stream parsing spreads across `--worker-threads` cores, and the one
//! remaining serial step (re-tokenizing the outputs) is batched.
//!
//! The report, the result JSON keys, and the flag names are kept identical, so
//! an existing command line and anything reading the `.jsonl` keep working.
//! See README.md for what this port does not cover.
//!
//! Python loads this as `sglang.srt.rust_extensions._bench`; see
//! `python/sglang/benchmark/rust_client.py`.

mod args;
use clap::Parser;
mod dataset;
mod hf;
mod metrics;
mod request;
mod run;
use pyo3::prelude::*;

pub use args::Args;
pub use run::run_blocking;

/// Run a benchmark described by a command line, as the CLI would.
///
/// `argv` starts with the program name, so clap's errors and `--help` name
/// the caller correctly. A usage error raises `SystemExit` with clap's own
/// exit code after printing clap's message, which keeps
/// `python -m sglang.benchmark.rust_client --help` behaving like any other
/// command line tool.
#[pyfunction]
fn run_argv(py: Python<'_>, argv: Vec<String>) -> PyResult<String> {
    let args = match Args::try_parse_from(&argv) {
        Ok(args) => args,
        Err(error) => {
            // clap renders both help and usage errors; print it the way it
            // asks to (help on stdout, errors on stderr).
            let _ = error.print();
            return Err(PyErr::new::<pyo3::exceptions::PySystemExit, _>(
                error.exit_code(),
            ));
        }
    };
    run(py, args)
}

/// Run a benchmark described by a JSON object whose keys are the long flag
/// names without their dashes. Missing keys take the flag's default, so a
/// caller sends only what it means to set.
///
/// This is the entry point for a Python caller that already parsed its own
/// arguments; `run_argv` is the one for a bare command line.
#[pyfunction]
fn run_config(py: Python<'_>, config: &str) -> PyResult<String> {
    let args: Args = serde_json::from_str(config).map_err(|error| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "invalid benchmark config: {error}"
        ))
    })?;
    run(py, args)
}

/// The GIL is released for the whole run: it owns a Tokio runtime with its
/// own threads and never touches Python, so holding the GIL would only
/// block the caller's other threads for the length of the benchmark.
fn run(py: Python<'_>, args: Args) -> PyResult<String> {
    let result = py
        .detach(|| run_blocking(args))
        .map_err(|error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{error:#}")))?;
    serde_json::to_string(&result).map_err(|error| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "cannot serialize the benchmark result: {error}"
        ))
    })
}

#[pymodule]
fn _bench(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run_argv, module)?)?;
    module.add_function(wrap_pyfunction!(run_config, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Python entry points keep their names, so an existing caller of
    /// `sglang.srt.rust_extensions._bench` keeps working after the token-source
    /// change.
    #[test]
    fn python_entry_points_keep_their_names() {
        // The functions are exported by name; this test pins the names the
        // Python wrapper imports.
        let _ = run_argv;
        let _ = run_config;
    }

    /// The result document keeps the Python script's key names, so existing
    /// readers of the `.jsonl` see the same fields after the token-source
    /// change.
    #[test]
    fn result_json_keeps_prompt_token_keys() {
        let outputs = vec![crate::request::RequestOutput {
            success: true,
            prompt_len: 128,
            dataset_prompt_len: 100,
            dataset_text_prompt_len: 100,
            output_len: 1,
            ..Default::default()
        }];
        let metrics = crate::metrics::calculate(&outputs, &[0], 1.0);
        let args = Args::default();
        let result = crate::metrics::result_json(
            &metrics,
            &args,
            f64::INFINITY,
            1.0,
            None,
            None,
            &outputs,
        );
        assert_eq!(result["total_input_tokens"], 128);
        assert_eq!(result["total_input_text_tokens"], 100);
        assert!(result.get("input_throughput").is_some());
        assert!(result.get("total_throughput").is_some());
    }

    /// The CLI surface and the report field names stay backward compatible:
    /// the long flags the Python wrapper passes still parse, and the result
    /// document keeps every token key it had before the token-source change.
    #[test]
    fn cli_and_report_fields_stay_compatible() {
        let args = Args::try_parse_from([
            "sglang-bench",
            "--backend",
            "vllm",
            "--num-prompts",
            "4",
            "--random-input-len",
            "16",
            "--random-output-len",
            "8",
        ])
        .expect("the documented flags still parse");
        let outputs = vec![crate::request::RequestOutput {
            success: true,
            prompt_len: 128,
            dataset_prompt_len: 100,
            dataset_text_prompt_len: 100,
            dataset_vision_prompt_len: 0,
            output_len: 1,
            ..Default::default()
        }];
        let metrics = crate::metrics::calculate(&outputs, &[0], 1.0);
        let result = crate::metrics::result_json(
            &metrics,
            &args,
            f64::INFINITY,
            1.0,
            None,
            None,
            &outputs,
        );
        for key in [
            "total_input_tokens",
            "total_input_text_tokens",
            "total_input_vision_tokens",
            "total_output_tokens",
            "total_output_tokens_retokenized",
            "input_throughput",
            "output_throughput",
            "total_throughput",
        ] {
            assert!(result.get(key).is_some(), "missing key {key}");
        }
        assert_eq!(result["total_input_tokens"], 128);
    }
}
