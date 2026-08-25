use std::collections::BTreeMap;

use rdkafka::message::Headers;
use serde::Deserialize;
use thiserror::Error;

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct TelemetryMessage {
    pub entity_type: String,
    pub entity_id: String,
    pub sensor_id: String,
    pub sensor_type: String,
    pub timestamp_ms: i64,
    pub metric: String,
    pub value: f64,
}

impl TelemetryMessage {
    pub fn validate_headers(&self, headers: &HeaderMap) -> Result<(), TelemetryError> {
        for (key, payload_value) in [
            ("entity_id", self.entity_id.as_str()),
            ("sensor_id", self.sensor_id.as_str()),
            ("sensor_type", self.sensor_type.as_str()),
            ("metric", self.metric.as_str()),
        ] {
            if let Some(header_value) = headers.get(key)
                && header_value != payload_value
            {
                return Err(TelemetryError::MetadataMismatch {
                    key: key.into(),
                    header: header_value.clone(),
                    payload: payload_value.into(),
                });
            }
        }
        if !self.value.is_finite() {
            return Err(TelemetryError::NonFiniteValue);
        }
        Ok(())
    }
}

pub type HeaderMap = BTreeMap<String, String>;

pub fn decode_headers<H: Headers>(headers: Option<&H>) -> Result<HeaderMap, TelemetryError> {
    let mut decoded = HeaderMap::new();
    let Some(headers) = headers else {
        return Ok(decoded);
    };
    for header in headers.iter() {
        let value = match header.value {
            Some(bytes) => {
                std::str::from_utf8(bytes).map_err(|source| TelemetryError::InvalidHeaderUtf8 {
                    key: header.key.into(),
                    source,
                })?
            }
            None => continue,
        };
        decoded.insert(header.key.into(), value.into());
    }
    Ok(decoded)
}

#[derive(Debug, Error)]
pub enum TelemetryError {
    #[error("Kafka header '{key}' is not UTF-8: {source}")]
    InvalidHeaderUtf8 {
        key: String,
        #[source]
        source: std::str::Utf8Error,
    },
    #[error("header '{key}' value '{header}' does not match payload value '{payload}'")]
    MetadataMismatch {
        key: String,
        header: String,
        payload: String,
    },
    #[error("telemetry value must be finite")]
    NonFiniteValue,
}
