type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

#[cfg(test)]
mod tests {
    use super::TestResult;

    #[test]
    fn serialize_deserialize_mask_settings() -> TestResult {
        let mask_settings = imanot::MaskSettings::default();
        let serialized = serde_json::to_string(&mask_settings)?;
        let deserialized: imanot::MaskSettings = serde_json::from_str(&serialized)?;
        assert_eq!(mask_settings, deserialized);
        Ok(())
    }
}
