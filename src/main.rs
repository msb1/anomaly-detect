use std::path::PathBuf;
use std::sync::Arc;

use anomaly_detect::config::AppConfig;
use anomaly_detect::kafka;
use anomaly_detect::pipeline::Pipeline;
use anyhow::{Context, Result};
use clap::Parser;
use tokio::sync::Mutex;
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
    let pipeline = Arc::new(Mutex::new(
        Pipeline::new(&config).context("failed to initialize anomaly models")?,
    ));
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
