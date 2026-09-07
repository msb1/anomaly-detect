use std::path::PathBuf;

use anomaly_detect::config::{AppConfig, RunMode};
use anomaly_detect::dataset;
use anomaly_detect::kafka;
use anomaly_detect::pipeline::Pipeline;
use anyhow::{Context, Result};
use clap::Parser;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    /// YAML configuration file.
    #[arg(long, default_value = "config/anomaly-detect.yaml")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| "anomaly_detect=info".into()),
        )
        .init();
    let args = Args::parse();
    let config = AppConfig::load(&args.config)
        .with_context(|| format!("failed to initialize from {}", args.config.display()))?;
    let mut pipeline = Pipeline::new(&config).context("failed to initialize anomaly models")?;
    if config.mode == RunMode::Dataset {
        let summary = dataset::run(&config.dataset, &mut pipeline).await?;
        tracing::info!(
            rows_read = summary.rows_read,
            matched_rows = summary.matched_rows,
            detections = summary.detections,
            anomalies = summary.anomalies,
            result_key = %summary.result_key,
            "dataset analysis complete"
        );
        return Ok(());
    }
    let pipeline = std::sync::Arc::new(tokio::sync::Mutex::new(pipeline));
    let cancellation = CancellationToken::new();
    let signal = cancellation.clone();
    tokio::spawn(async move {
        match tokio::signal::ctrl_c().await {
            Ok(()) => signal.cancel(),
            Err(error) => tracing::error!(%error, "could not listen for Ctrl-C"),
        }
    });
    kafka::consume(&config.kafka, pipeline, cancellation).await
}
