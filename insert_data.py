from sensor.data_access.sensor_data import SensorData

if __name__ == "__main__":
    try:
        sensor_data = SensorData()

        # Step 1: Insert CSV into MongoDB
        count = sensor_data.save_csv_file(
            file_path="aps_failure_training_set1.csv",
            collection_name="sensor_collection"   # 👈 same name as in config.yaml
        )

        print(f"✅ {count} records inserted into MongoDB.")

        # Step 2: Confirm insert
        df = sensor_data.export_collection_as_dataframe("sensor_collection")
        print("📊 Loaded data shape:", df.shape)
        print(df.head())

    except Exception as e:
        print("❌ Error:", e)
