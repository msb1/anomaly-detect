use std::collections::BTreeMap;
use std::io::Cursor;
use std::sync::Arc;

use anyhow::{Context, Result};
use aws_config::BehaviorVersion;
use aws_sdk_s3::{Client, primitives::ByteStream};
use parquet::basic::Compression;
use parquet::column::writer::ColumnWriter;
use parquet::data_type::ByteArray;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::writer::SerializedFileWriter;
use parquet::record::{Field, Row};
use parquet::schema::parser::parse_message_type;

use crate::config::DatasetConfig;
use crate::kafka::AnomalyMessage;
use crate::pipeline::Pipeline;
use crate::telemetry::{HeaderMap, TelemetryMessage};

#[derive(Debug, Default)]
pub struct DatasetRunSummary {
    pub rows_read: usize,
    pub matched_rows: usize,
    pub detections: usize,
    pub anomalies: usize,
    pub result_key: String,
}

/// Download, fully materialize, chronologically order, and replay a Parquet
/// dataset through the same Pipeline used by Kafka ingestion.
pub async fn run(config: &DatasetConfig, pipeline: &mut Pipeline) -> Result<DatasetRunSummary> {
    let shared = aws_config::defaults(BehaviorVersion::latest())
        .region(aws_config::Region::new("us-east-1"))
        .credentials_provider(aws_sdk_s3::config::Credentials::new(
            config.s3_access_key.clone(),
            config.s3_secret_key.clone(),
            None,
            None,
            "anomaly-detect-dataset-config",
        ))
        .endpoint_url(&config.s3_endpoint_url)
        .load()
        .await;
    let client = Client::from_conf(
        aws_sdk_s3::config::Builder::from(&shared)
            .force_path_style(true)
            .build(),
    );
    let object = client
        .get_object()
        .bucket(&config.s3_bucket_name)
        .key(&config.parquet_key)
        .send()
        .await
        .with_context(|| {
            format!(
                "could not download s3://{}/{}",
                config.s3_bucket_name, config.parquet_key
            )
        })?;
    let bytes = object
        .body
        .collect()
        .await
        .context("could not read downloaded Parquet object")?
        .into_bytes();
    let rows = decode_parquet(bytes)?;
    let (mut summary, results) = replay(rows, pipeline)?;
    let result_key = result_key(&config.parquet_key)?;
    upload_results(&client, config, &result_key, &results).await?;
    summary.result_key = result_key;
    Ok(summary)
}

fn decode_parquet(bytes: bytes::Bytes) -> Result<Vec<TelemetryMessage>> {
    let reader =
        SerializedFileReader::new(bytes).context("dataset object is not readable Parquet")?;
    let mut telemetry = Vec::with_capacity(reader.metadata().file_metadata().num_rows() as usize);
    for (index, row) in reader
        .get_row_iter(None)
        .context("could not iterate dataset Parquet rows")?
        .enumerate()
    {
        let row = row.with_context(|| format!("could not decode Parquet row {index}"))?;
        // Do not rely on the order in which the dataframe's columns were
        // written.  In particular, pandas commonly writes its datetime
        // column first and Parquet exposes it as a logical timestamp field,
        // rather than as a `Long` row field.
        telemetry.push(TelemetryMessage {
            entity_type: string_field(&row, &["entity_type"])?.to_owned(),
            entity_id: string_field(&row, &["entity_id"])?.to_owned(),
            sensor_id: string_field(&row, &["sensor_id"])?.to_owned(),
            sensor_type: string_field(&row, &["sensor_type"])?.to_owned(),
            timestamp_ms: timestamp_ms_field(&row, &["timestamp_ms", "timestamp", "event_time"])?,
            interval_ms: integer_field(&row, &["interval_ms"])?,
            metric: string_field(&row, &["metric"])?.to_owned(),
            value: double_field(&row, &["value"])?,
        });
    }
    // iot-sim writes metric rows in batches; make chronological replay explicit
    // so all input methods advance their windows identically.
    telemetry.sort_by(|left, right| {
        (
            left.timestamp_ms,
            &left.entity_id,
            &left.sensor_id,
            &left.metric,
        )
            .cmp(&(
                right.timestamp_ms,
                &right.entity_id,
                &right.sensor_id,
                &right.metric,
            ))
    });
    Ok(telemetry)
}

fn field<'a>(row: &'a Row, names: &[&str]) -> Result<&'a Field> {
    row.get_column_iter()
        .find(|(name, _)| names.iter().any(|candidate| name == candidate))
        .map(|(_, value)| value)
        .with_context(|| {
            format!(
                "dataset Parquet row is missing column(s): {}",
                names.join(", ")
            )
        })
}

fn string_field<'a>(row: &'a Row, names: &[&str]) -> Result<&'a str> {
    match field(row, names)? {
        Field::Str(value) => Ok(value),
        value => anyhow::bail!(
            "Parquet column '{}' is {}, expected a string",
            names[0],
            value
        ),
    }
}

fn integer_field(row: &Row, names: &[&str]) -> Result<i64> {
    match field(row, names)? {
        Field::Long(value) => Ok(*value),
        value => anyhow::bail!(
            "Parquet column '{}' is {}, expected an integer",
            names[0],
            value
        ),
    }
}

fn double_field(row: &Row, names: &[&str]) -> Result<f64> {
    match field(row, names)? {
        Field::Double(value) => Ok(*value),
        Field::Float(value) => Ok(*value as f64),
        value => anyhow::bail!(
            "Parquet column '{}' is {}, expected a number",
            names[0],
            value
        ),
    }
}

fn timestamp_ms_field(row: &Row, names: &[&str]) -> Result<i64> {
    match field(row, names)? {
        // pandas/pyarrow can write datetime64 columns with either of these
        // units.  The pipeline's internal timestamp unit is milliseconds.
        Field::TimestampMillis(value) => Ok(*value),
        Field::TimestampMicros(value) => Ok(*value / 1_000),
        // Keep accepting the original iot-sim schema, which stores epoch
        // milliseconds as a plain required INT64.
        Field::Long(value) => Ok(*value),
        value => anyhow::bail!(
            "Parquet column '{}' is {}, expected a timestamp",
            names[0],
            value
        ),
    }
}

fn replay(
    rows: Vec<TelemetryMessage>,
    pipeline: &mut Pipeline,
) -> Result<(DatasetRunSummary, Vec<AnomalyMessage>)> {
    let mut summary = DatasetRunSummary {
        rows_read: rows.len(),
        ..DatasetRunSummary::default()
    };
    let mut results = Vec::new();
    for (index, message) in rows.into_iter().enumerate() {
        let headers = headers_for(&message);
        let matched = pipeline.matching_streams(&headers);
        if matched.is_empty() {
            continue;
        }
        summary.matched_rows += 1;
        let outcome = pipeline
            .ingest(matched, &headers, message)
            .with_context(|| format!("could not ingest dataset row {index}"))?;
        summary.detections += outcome.detections.len();
        for detection in outcome.detections {
            let model = outcome
                .ready_models
                .iter()
                .find(|model| model.id == detection.model_id)
                .with_context(|| format!("dataset row {index} emitted an unknown model result"))?;
            if detection.anomalous {
                summary.anomalies += 1;
                tracing::warn!(
                    model_id = %detection.model_id,
                    timestamp_ms = detection.timestamp_ms,
                    score = ?detection.score,
                    details = ?detection.details,
                    "dataset anomaly detected"
                );
            }
            results.push(AnomalyMessage::from_detection(detection, model));
        }
    }
    Ok((summary, results))
}

fn result_key(parquet_key: &str) -> Result<String> {
    let filename = parquet_key.rsplit('/').next().unwrap_or(parquet_key);
    let dataset_name = filename
        .strip_suffix(".parquet")
        .filter(|name| !name.is_empty())
        .context("dataset.parquet_key must end with a dataset filename")?;
    Ok(format!("results/{dataset_name}-results.parquet"))
}

async fn upload_results(
    client: &Client,
    config: &DatasetConfig,
    key: &str,
    results: &[AnomalyMessage],
) -> Result<()> {
    let body = ByteStream::from(result_parquet(results)?);
    client
        .put_object()
        .bucket(&config.s3_bucket_name)
        .key(key)
        .content_type("application/vnd.apache.parquet")
        .body(body)
        .send()
        .await
        .with_context(|| format!("could not upload s3://{}/{}", config.s3_bucket_name, key))?;
    Ok(())
}

const RESULT_SCHEMA: &str = "message anomaly_results {\
    REQUIRED INT64 timestamp_ms;\
    REQUIRED BINARY model_id (UTF8);\
    REQUIRED BINARY model_kind (UTF8);\
    REQUIRED BINARY algorithm (UTF8);\
    REQUIRED BINARY score_algorithm (UTF8);\
    REQUIRED BINARY inputs_json (UTF8);\
    REQUIRED INT64 anomalous_points;\
    REQUIRED INT64 window_sample_count;\
    REQUIRED BOOLEAN is_anomalous;\
    REQUIRED BOOLEAN has_anomaly_score;\
    REQUIRED DOUBLE anomaly_score;\
    REQUIRED BINARY details_json (UTF8);\
}";

fn result_parquet(results: &[AnomalyMessage]) -> Result<Vec<u8>> {
    let schema = Arc::new(parse_message_type(RESULT_SCHEMA)?);
    let properties = Arc::new(
        WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build(),
    );
    let mut writer = SerializedFileWriter::new(Cursor::new(Vec::new()), schema, properties)?;
    for batch in results.chunks(1024) {
        let model_kinds = batch
            .iter()
            .map(|result| serde_json::to_string(&result.model_kind).expect("model kind serializes"))
            .collect::<Vec<_>>();
        let inputs = batch
            .iter()
            .map(|result| serde_json::to_string(&result.inputs).expect("inputs serialize"))
            .collect::<Vec<_>>();
        let details = batch
            .iter()
            .map(|result| serde_json::to_string(&result.details).expect("details serialize"))
            .collect::<Vec<_>>();
        let mut group = writer.next_row_group()?;
        write_i64(&mut group, batch.iter().map(|result| result.timestamp_ms))?;
        write_bytes(
            &mut group,
            batch.iter().map(|result| result.model_id.as_str()),
        )?;
        write_bytes(
            &mut group,
            model_kinds.iter().map(|value| value.trim_matches('"')),
        )?;
        write_bytes(
            &mut group,
            batch.iter().map(|result| result.algorithm.as_str()),
        )?;
        write_bytes(
            &mut group,
            batch.iter().map(|result| result.score_algorithm.as_str()),
        )?;
        write_bytes(&mut group, inputs.iter().map(String::as_str))?;
        write_i64(
            &mut group,
            batch
                .iter()
                .map(|result| result.anomalous_points.min(i64::MAX as usize) as i64),
        )?;
        write_i64(
            &mut group,
            batch
                .iter()
                .map(|result| result.window_sample_count.min(i64::MAX as usize) as i64),
        )?;
        write_bool(&mut group, batch.iter().map(|result| result.is_anomalous))?;
        write_bool(
            &mut group,
            batch.iter().map(|result| result.anomaly_score.is_some()),
        )?;
        write_f64(
            &mut group,
            batch
                .iter()
                .map(|result| result.anomaly_score.unwrap_or(f64::NAN)),
        )?;
        write_bytes(&mut group, details.iter().map(String::as_str))?;
        group.close()?;
    }
    Ok(writer.into_inner()?.into_inner())
}

fn write_bytes<'a>(
    group: &mut parquet::file::writer::SerializedRowGroupWriter<'_, Cursor<Vec<u8>>>,
    values: impl Iterator<Item = &'a str>,
) -> Result<()> {
    let values: Vec<ByteArray> = values.map(ByteArray::from).collect();
    let mut column = group
        .next_column()?
        .context("Parquet results schema has fewer columns than expected")?;
    match column.untyped() {
        ColumnWriter::ByteArrayColumnWriter(writer) => {
            writer.write_batch(&values, None, None)?;
        }
        _ => anyhow::bail!("unexpected Parquet results column type"),
    }
    column.close()?;
    Ok(())
}

fn write_i64(
    group: &mut parquet::file::writer::SerializedRowGroupWriter<'_, Cursor<Vec<u8>>>,
    values: impl Iterator<Item = i64>,
) -> Result<()> {
    let values: Vec<i64> = values.collect();
    let mut column = group
        .next_column()?
        .context("Parquet results schema has fewer columns than expected")?;
    match column.untyped() {
        ColumnWriter::Int64ColumnWriter(writer) => {
            writer.write_batch(&values, None, None)?;
        }
        _ => anyhow::bail!("unexpected Parquet results column type"),
    }
    column.close()?;
    Ok(())
}

fn write_bool(
    group: &mut parquet::file::writer::SerializedRowGroupWriter<'_, Cursor<Vec<u8>>>,
    values: impl Iterator<Item = bool>,
) -> Result<()> {
    let values: Vec<bool> = values.collect();
    let mut column = group
        .next_column()?
        .context("Parquet results schema has fewer columns than expected")?;
    match column.untyped() {
        ColumnWriter::BoolColumnWriter(writer) => {
            writer.write_batch(&values, None, None)?;
        }
        _ => anyhow::bail!("unexpected Parquet results column type"),
    }
    column.close()?;
    Ok(())
}

fn write_f64(
    group: &mut parquet::file::writer::SerializedRowGroupWriter<'_, Cursor<Vec<u8>>>,
    values: impl Iterator<Item = f64>,
) -> Result<()> {
    let values: Vec<f64> = values.collect();
    let mut column = group
        .next_column()?
        .context("Parquet results schema has fewer columns than expected")?;
    match column.untyped() {
        ColumnWriter::DoubleColumnWriter(writer) => {
            writer.write_batch(&values, None, None)?;
        }
        _ => anyhow::bail!("unexpected Parquet results column type"),
    }
    column.close()?;
    Ok(())
}

fn headers_for(message: &TelemetryMessage) -> HeaderMap {
    BTreeMap::from([
        ("entity_id".into(), message.entity_id.clone()),
        ("sensor_id".into(), message.sensor_id.clone()),
        ("sensor_type".into(), message.sensor_type.clone()),
        ("metric".into(), message.metric.clone()),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelKind;
    use parquet::record::RowAccessor;

    #[test]
    fn headers_match_telemetry_identity() {
        let message = TelemetryMessage {
            entity_type: "rack".into(),
            entity_id: "rack-01".into(),
            sensor_id: "sensor-01".into(),
            sensor_type: "data_center_rack".into(),
            timestamp_ms: 1,
            interval_ms: 1,
            metric: "temperature_f".into(),
            value: 70.0,
        };
        assert_eq!(
            headers_for(&message).get("metric"),
            Some(&"temperature_f".into())
        );
    }

    #[test]
    fn pandas_timestamp_fields_are_read_by_name_and_normalized_to_millis() {
        let row = Row::new(vec![
            (
                "timestamp".into(),
                Field::TimestampMillis(1_725_000_000_123),
            ),
            ("value".into(), Field::Double(70.0)),
            ("metric".into(), Field::Str("temperature_f".into())),
        ]);

        assert_eq!(
            timestamp_ms_field(&row, &["timestamp_ms", "timestamp"]).unwrap(),
            1_725_000_000_123
        );
        assert_eq!(double_field(&row, &["value"]).unwrap(), 70.0);
        assert_eq!(string_field(&row, &["metric"]).unwrap(), "temperature_f");
    }

    #[test]
    fn microsecond_timestamps_are_converted_to_millis() {
        let row = Row::new(vec![(
            "event_time".into(),
            Field::TimestampMicros(1_725_000_000_123_456),
        )]);
        assert_eq!(
            timestamp_ms_field(&row, &["timestamp", "event_time"]).unwrap(),
            1_725_000_000_123
        );
    }

    #[test]
    fn result_parquet_retains_detection_identity() {
        let results = vec![AnomalyMessage {
            schema_version: "iot.anomaly.v1".into(),
            timestamp_ms: 123,
            model_id: "rack-temperature".into(),
            model_kind: ModelKind::Univariate,
            algorithm: "mad".into(),
            score_algorithm: "mad".into(),
            inputs: vec!["rack_temperature".into()],
            anomalous_points: 1,
            window_sample_count: 10,
            is_anomalous: true,
            anomaly_score: Some(8.5),
            details: BTreeMap::new(),
        }];
        let bytes = result_parquet(&results).unwrap();
        let reader = SerializedFileReader::new(bytes::Bytes::from(bytes)).unwrap();
        let row = reader.get_row_iter(None).unwrap().next().unwrap().unwrap();
        assert_eq!(row.get_long(0).unwrap(), 123);
        assert_eq!(row.get_string(1).unwrap(), "rack-temperature");
        assert_eq!(row.get_bool(8).unwrap(), true);
        assert_eq!(row.get_double(10).unwrap(), 8.5);
        assert_eq!(
            result_key("dataset/iot-telemetry-123.parquet").unwrap(),
            "results/iot-telemetry-123-results.parquet"
        );
    }
}
