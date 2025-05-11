/* Includes ---------------------------------------------------------------- */
#include <prueba_inferencing.h>
#include <DHT.h>

// DHT11 sensor configuration
#define DHTPIN 4
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

// Normalization ranges
const float min_temp = -1.13;
const float max_temp = 24.76;
const float min_hum  = 2.06;
const float max_hum  = 100.0;

// Season embeddings (Southern Hemisphere) 
const float embeddings[4][4] = {
    { 0.08841707, -0.01442201,  0.06266572, -0.04614694 }, // Summer (0)
    {-0.04601293,  0.10943006,  0.00833590,  0.05554458 }, // Autumn (1)
    {-0.07277188,  0.01109327, -0.09172057, -0.08308954 }, // Winter (2)
    {-0.05201762, -0.11797583,  0.06664423,  0.03551533 }  // Spring (3)
};

// Circular buffer and variables
float dynamic_features[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = {0};
int current_index   = 0;
int current_season = 1; // autumn

// Normalization function
float normalize(float value, float min_val, float max_val) {
    return (value - min_val) / (max_val - min_val);
}

// Determine season (Southern Hem.)
int get_season(int month) {
    const int season_map[12] = {0,0,1,1,1,2,2,2,3,3,3,0};
    return season_map[month - 1];
}

// Update buffer with dynamic embedding
void update_buffer(float temp, float hum, int season) {
    if (current_index >= 24) {
        // shift one sample (6 floats) to the beginning
        memmove(dynamic_features,
                dynamic_features + 6,
                (EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - 6) * sizeof(float));
        current_index = 23;
    }

    float norm_temp = normalize(temp, min_temp, max_temp);
    float norm_hum  = normalize(hum,   min_hum,  max_hum);
    const float *emb = embeddings[season];

    int pos = current_index * 6;
    dynamic_features[pos++] = norm_temp;
    dynamic_features[pos++] = norm_hum;
    dynamic_features[pos++] = emb[0];
    dynamic_features[pos++] = emb[1];
    dynamic_features[pos++] = emb[2];
    dynamic_features[pos++] = emb[3];

    current_index++;
}

int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
    memcpy(out_ptr, dynamic_features + offset, length * sizeof(float));
    return 0;
}

void setup() {
    Serial.begin(115200);
    dht.begin();
    while(!Serial);
    Serial.println("Seasonal System with Dynamic Embeddings");
}

void loop() {
    // 1. Read sensor
    float t = dht.readTemperature();
    float h = dht.readHumidity();
    if (isnan(t) || isnan(h)) {
        Serial.println("DHT11 sensor error!");
        delay(2000);
        return;
    }

    // 2. Simulate current month
    int month = 4; // april → autumn in Southern Hem.

    // 3. Update season and buffer
    current_season = get_season(month);
    update_buffer(t, h, current_season);

    // 4. Local debug
    Serial.print("Season: "); Serial.println(current_season);
    Serial.print("Temp: "); Serial.print(t);
    Serial.print("°C | Hum: "); Serial.print(h); Serial.println("%");

    // 5. When buffer is full, run inference
    if (current_index >= 24) {
        ei_impulse_result_t result = {0};
        signal_t signal;
        signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
        signal.get_data     = &raw_feature_get_data;

        if (run_classifier(&signal, &result, false) == EI_IMPULSE_OK) {
            print_inference_result(result);
        }
    }

    delay(5000);
}

void print_inference_result(ei_impulse_result_t result) {
    ei_printf("Timing: DSP %d ms, inference %d ms, anomaly %d ms\n",
              result.timing.dsp,
              result.timing.classification,
              result.timing.anomaly);

    ei_printf("Predictions (normalised + real):\n");
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        float norm = result.classification[i].value;
        ei_printf("  %s: %.5f", 
                  ei_classifier_inferencing_categories[i],
                  norm);

        float real = 0.0f;
        if (i % 2 == 0) {
            //  temperature (indices 0,2,4)
            real = norm * (max_temp - min_temp) + min_temp;
            ei_printf(" | Temp: %.2f°C\n", real);
        } else {
            //  humidity (indices 1,3,5)
            real = norm * (max_hum - min_hum) + min_hum;
            ei_printf(" | Hum: %.2f%%\n", real);
        }
    }

    #if EI_CLASSIFIER_HAS_ANOMALY
    ei_printf("Anomaly score: %.3f\n", result.anomaly);
    #endif
};
