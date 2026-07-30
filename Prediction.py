features = scaler.transform([[air_temp,
                              process_temp,
                              rpm,
                              torque,
                              tool_wear]])

prediction = model.predict(features)
