// ============================================================================
// ESP32 Biofeedback Sensor Node
// ============================================================================
// Description:
//   Reads PPG (heart rate) and GSR (galvanic skin response) sensor data,
//   processes and filters the signals, displays BPM on an OLED screen,
//   and publishes results over MQTT via a self-hosted Wi-Fi Access Point.
//
// Hardware:
//   - ESP32 DevKit
//   - SH1106 128x64 OLED Display (I2C)
//   - PPG Pulse Sensor (analog, GPIO 34)
//   - GSR Sensor (analog, GPIO 35)
//   - Green LED  (GPIO 26) — system ready indicator
//   - Red LED    (GPIO 27) — alert/stress indicator
//   - Blue LED   (GPIO 25) — heartbeat flash
//   - Start Button (GPIO 14) — initiates sensor setup
//
// Dependencies:
//   - WiFi.h
//   - TinyMqtt
//   - Wire.h
//   - Adafruit_GFX
//   - Adafruit_SH110X
//   - PulseSensorPlayground (included but signal processing is manual)
//
// MQTT Topics:
//   - sensors/bpm    → current averaged BPM value
//   - sensors/gsr    → "STRESSED" or "RELAXED/NORMAL"
//   - sensors/status → system status messages
//   - patient/alert  → receives "ALERT" or "NORMAL" from Unity
//
// Author  : [Your Name]
// Version : 1.0
// ============================================================================

#include <WiFi.h>
#include <TinyMqtt.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>
#include <PulseSensorPlayground.h>


// ============================================================================
// Pin Definitions
// ============================================================================

const int START_BTN = 14;
const int GREEN_LED = 26;
const int RED_LED   = 27;
const int BLUE_LED  = 25;
const int PULSE_PIN = 34;
const int GSR_PIN   = 35;


// ============================================================================
// Wi-Fi Access Point Settings
// ============================================================================

const char* AP_SSID = "ESP32_AP";
const char* AP_PASS = "87654321";   // Minimum 8 characters

bool apStarted     = false;
bool brokerStarted = false;


// ============================================================================
// MQTT Topics, Broker, and Clients
// ============================================================================

const char* TOPIC_BPM    = "sensors/bpm";
const char* TOPIC_GSR    = "sensors/gsr";
const char* TOPIC_STATUS = "sensors/status";
const char* TOPIC_ALERT  = "patient/alert";

MqttBroker broker(1883);             // Hosts the broker on port 1883
MqttClient publisher(&broker);       // Publishes sensor data
MqttClient subscriber(&broker);      // Subscribes to alert commands from Unity


// ============================================================================
// OLED Display
// ============================================================================

Adafruit_SH1106G OledScreen(128, 64, &Wire, -1);

unsigned long lastI2CCheck    = 0;
unsigned long lastGraphUpdate = 0;
unsigned long lastHeaderUpdate = 0;
unsigned long lastOLEDRefresh = 0;

// Graph drawing state
int graphPrevX = 0;
int graphPrevY = 60;
int graphCurrX = 0;
int graphMappedValue = 0;


// ============================================================================
// Alert State
// ============================================================================

bool alertState = false;


// ============================================================================
// PPG Signal Processing Variables
// ============================================================================

int  processedPPG  = 0;
int  lastValidBPM  = 0;
int  lastPrintTime = 0;

// Trend confirmation — prevents false BPM jumps from noise
int trendCandidate       = 0;
int trendCount           = 0;
const int TREND_THRESHOLD = 8;    // Consecutive readings needed to confirm a shift
const int TREND_TOLERANCE = 10;   // Acceptable variance within a trend candidate

// DC Removal (high-pass filter)
float dcAlpha       = 0.97;
float dcValue       = 0;
bool  dcInitialized = false;

// Amplification
float gain = 8.0;

// Beat Detection
int  PPG_THRESHOLD        = 300;
bool beatDetected         = false;
unsigned long lastBeatTime = 0;
const unsigned long REFRACTORY_MS = 300;   // Minimum ms between valid beats
int BPM      = 0;
int lastBPM  = 0;
const int BPM_TOLERANCE = 35;              // Max allowed BPM jump before trend tracking

// BPM Rolling Average Buffer
const int BPM_BUFFER_SIZE = 5;
int bpmBuffer[BPM_BUFFER_SIZE] = {0};
int bpmBufferIndex = 0;
int bpmBufferCount = 0;

// Signal Smoothing Buffer
const int SMOOTH_SIZE = 4;
int smoothBuf[SMOOTH_SIZE] = {0};
int smoothIdx = 0;

// Warmup Period (allows DC filter to settle before reading beats)
const unsigned long WARMUP_MS = 10000;
bool warmupDone = false;


// ============================================================================
// GSR Calibration and Processing Variables
// ============================================================================

int GSR_threshold                   = 0;
int GSR_sensorValue                 = 0;
const int GSR_CALIBRATION_SAMPLES   = 500;
const int GSR_THRESHOLD_PERCENTAGE  = 15;
const int GSR_MIN_VALID_READING     = 100;

// -----------------------------------------------------------------------
// GSRCalibration (Struct)
// Holds the result of a GSR calibration run.
//   - sum          : sum of all valid analog readings
//   - validReadings: count of readings above the minimum threshold
// -----------------------------------------------------------------------
struct GSRCalibrationResult {
  long sum;
  int  validReadings;
};


// ============================================================================
// Alert Functions
// ============================================================================

// -----------------------------------------------------------------------
// RedBlink()
// Non-blocking red LED blink at ~300ms intervals.
// Called every loop iteration while alertState is true.
// -----------------------------------------------------------------------
void RedBlink() {
  static unsigned long lastToggle = 0;
  static bool state = false;

  if (millis() - lastToggle > 300) {
    state = !state;
    digitalWrite(RED_LED, state);
    lastToggle = millis();
  }
}

// -----------------------------------------------------------------------
// GreenBlink()
// Non-blocking green LED blink at ~50ms intervals.
// Used to indicate active heartbeat or system activity.
// -----------------------------------------------------------------------
void GreenBlink() {
  static unsigned long lastToggle = 0;
  static bool state = false;

  if (millis() - lastToggle > 50) {
    state = !state;
    digitalWrite(GREEN_LED, state);
    lastToggle = millis();
  }
}


// ============================================================================
// MQTT Callback
// ============================================================================

// -----------------------------------------------------------------------
// onMessageReceived()
// Fires when a message arrives on a subscribed topic.
// Handles "ALERT" and "NORMAL" commands from Unity on patient/alert topic.
//   - "ALERT"  : sets alertState = true, triggers red LED blinking.
//   - "NORMAL" : clears alertState, turns off red LED.
// -----------------------------------------------------------------------
void onMessageReceived(const MqttClient* client, const Topic& topic, const char* payload, size_t len) {
  String message = "";
  for (int i = 0; i < len; i++) {
    message += (char)payload[i];
  }

  if (message == "ALERT") {
    alertState = true;
  } else if (message == "NORMAL") {
    alertState = false;
    digitalWrite(RED_LED, LOW);
  }
}


// ============================================================================
// OLED Display Functions
// ============================================================================

// -----------------------------------------------------------------------
// resetI2C()
// Recovers from a frozen I2C bus by toggling SDA/SCL lines manually,
// reinitializing Wire, and sending a display-off command to the OLED.
// Called automatically every 5 seconds if the OLED stops responding.
// -----------------------------------------------------------------------
void resetI2C() {
  Wire.end();
  delay(100);

  // Manually toggle bus lines to unstick any held state
  pinMode(21, OUTPUT);
  pinMode(22, OUTPUT);
  for (int i = 0; i < 10; i++) {
    digitalWrite(21, LOW);
    digitalWrite(22, LOW);
    delay(5);
    digitalWrite(21, HIGH);
    digitalWrite(22, HIGH);
    delay(5);
  }

  // Reinitialize I2C at safe speed
  Wire.begin(21, 22);
  Wire.setClock(100000);
  delay(200);

  // Send display-off command to OLED to complete reset
  Wire.beginTransmission(0x3C);
  Wire.write(0x00);   // Command mode
  Wire.write(0xAE);   // Display OFF
  Wire.endTransmission();
  delay(100);
}

// -----------------------------------------------------------------------
// splashScreen()
// Displays a welcome screen on the OLED for 3 seconds at startup.
// -----------------------------------------------------------------------
void splashScreen() {
  OledScreen.clearDisplay();
  OledScreen.setTextColor(SH110X_WHITE);

  OledScreen.setTextSize(2);
  OledScreen.setCursor(10, 0);
  OledScreen.print("Welcome");

  OledScreen.setTextSize(1);
  OledScreen.setCursor(5, 20);
  OledScreen.print("Acrophobia Therapy");

  OledScreen.setCursor(15, 35);
  OledScreen.print("System Starting...");

  OledScreen.display();
  delay(3000);
  OledScreen.clearDisplay();
}

// -----------------------------------------------------------------------
// drawHeader()
// Renders the "BPM: XXX" header at the top of the OLED screen.
// Clears the old BPM value area before redrawing to prevent ghosting.
// -----------------------------------------------------------------------
void drawHeader() {
  OledScreen.setTextSize(2);
  OledScreen.setTextColor(SH110X_WHITE);
  OledScreen.setCursor(0, 0);
  OledScreen.print("BPM:");

  // Clear old BPM digits before writing new value
  OledScreen.fillRect(60, 0, 68, 16, SH110X_BLACK);
  OledScreen.setCursor(60, 0);
  OledScreen.print(lastValidBPM);
}


// ============================================================================
// PPG Signal Processing Functions
// ============================================================================

// -----------------------------------------------------------------------
// smoothRaw()
// Applies a simple moving average over the last SMOOTH_SIZE samples
// to reduce high-frequency noise from the raw PPG signal.
// Returns the averaged value.
// -----------------------------------------------------------------------
int smoothRaw(int rawPPG) {
  smoothBuf[smoothIdx] = rawPPG;
  smoothIdx = (smoothIdx + 1) % SMOOTH_SIZE;

  long sum = 0;
  for (int i = 0; i < SMOOTH_SIZE; i++) sum += smoothBuf[i];
  return (int)(sum / SMOOTH_SIZE);
}

// -----------------------------------------------------------------------
// processSignal()
// Removes the DC baseline from the PPG signal using an IIR high-pass
// filter, then amplifies the AC component to make peaks detectable.
//
// Steps:
//   1. Initialize dcValue on first call.
//   2. Update dcValue using exponential moving average (dcAlpha).
//   3. Subtract dcValue from raw to isolate the AC waveform.
//   4. Multiply by gain to amplify the signal.
//
// Returns the processed (centered + amplified) signal as an int.
// -----------------------------------------------------------------------
int processSignal(int rawValue) {
  if (!dcInitialized) {
    dcValue = rawValue;
    dcInitialized = true;
  }

  dcValue = dcAlpha * dcValue + (1.0 - dcAlpha) * rawValue;
  float centered  = rawValue - dcValue;
  float amplified = centered * gain;
  return (int)amplified;
}

// -----------------------------------------------------------------------
// computeAverageBPM()
// Returns the rolling average of the last BPM_BUFFER_SIZE valid BPM
// readings. Returns 0 if no readings have been collected yet.
// -----------------------------------------------------------------------
int computeAverageBPM() {
  int count = min(bpmBufferCount, BPM_BUFFER_SIZE);
  if (count == 0) return 0;

  long sum = 0;
  for (int i = 0; i < count; i++) sum += bpmBuffer[i];
  return (int)(sum / count);
}

// -----------------------------------------------------------------------
// detectBeat()
// Detects heartbeat peaks in the processed PPG signal and updates BPM.
//
// Logic:
//   - Triggers on rising edge above PPG_THRESHOLD.
//   - Enforces a refractory period (REFRACTORY_MS) to reject double-triggers.
//   - Accepts new BPM if within BPM_TOLERANCE of the last reading.
//   - If outside tolerance, begins trend tracking:
//       - Accumulates consistent out-of-range readings as a "trend candidate".
//       - After TREND_THRESHOLD confirmations, accepts the new BPM as valid.
//       - Inconsistent readings reset the trend tracker.
//   - Rejects BPM values outside the valid physiological range (70–150).
//   - Flashes BLUE_LED briefly on each confirmed beat.
// -----------------------------------------------------------------------
void detectBeat(int processedValue) {
  unsigned long now = millis();

  if (processedValue > PPG_THRESHOLD && !beatDetected) {

    // Ignore beats within the refractory period
    if (lastBeatTime != 0 && (now - lastBeatTime) < REFRACTORY_MS) {
      return;
    }

    beatDetected = true;

    if (lastBeatTime != 0) {
      unsigned long interval = now - lastBeatTime;
      int newBPM = 60000 / interval;

      if (newBPM > 70 && newBPM < 150) {

        if (lastBPM == 0 || abs(newBPM - lastBPM) <= BPM_TOLERANCE) {
          // Beat is within acceptable range — accept immediately
          BPM = newBPM;
          lastBPM = BPM;
          trendCandidate = 0;
          trendCount = 0;

          bpmBuffer[bpmBufferIndex] = BPM;
          bpmBufferIndex = (bpmBufferIndex + 1) % BPM_BUFFER_SIZE;
          if (bpmBufferCount < BPM_BUFFER_SIZE) bpmBufferCount++;

          lastValidBPM = computeAverageBPM();

          // Flash blue LED to visually confirm beat
          digitalWrite(BLUE_LED, HIGH);
          delay(50);
          digitalWrite(BLUE_LED, LOW);

        } else {
          // Beat is outside tolerance — begin or continue trend tracking
          if (trendCandidate == 0 || abs(newBPM - trendCandidate) <= TREND_TOLERANCE) {
            trendCandidate = newBPM;
            trendCount++;

            if (trendCount >= TREND_THRESHOLD) {
              // Trend confirmed — genuine BPM shift, accept new value
              BPM = trendCandidate;
              lastBPM = BPM;
              trendCandidate = 0;
              trendCount = 0;

              bpmBuffer[bpmBufferIndex] = BPM;
              bpmBufferIndex = (bpmBufferIndex + 1) % BPM_BUFFER_SIZE;
              if (bpmBufferCount < BPM_BUFFER_SIZE) bpmBufferCount++;

              lastValidBPM = computeAverageBPM();

              digitalWrite(BLUE_LED, HIGH);
              delay(50);
              digitalWrite(BLUE_LED, LOW);
            }

          } else {
            // Inconsistent with trend candidate — reset and restart with this reading
            trendCandidate = newBPM;
            trendCount = 1;
          }
        }

      } else {
        // BPM out of valid physiological range — discard and reset trend
        trendCandidate = 0;
        trendCount = 0;
      }
    }

    lastBeatTime = now;
  }

  // Reset beat flag on falling edge
  if (processedValue < PPG_THRESHOLD && beatDetected) {
    beatDetected = false;
  }
}


// ============================================================================
// GSR Functions
// ============================================================================

// -----------------------------------------------------------------------
// processGSR()
// Reads and evaluates the current GSR (skin conductance) level.
//
// Process:
//   1. Takes 10 analog readings and averages the valid ones (above minimum).
//   2. Compares average to the calibrated baseline threshold.
//   3. If conductance exceeds threshold by GSR_THRESHOLD_PERCENTAGE,
//      takes a second confirmation reading to reduce false positives.
//   4. Confirms stress state only if both readings agree.
//
// Returns:
//   0 = STRESSED     (conductance significantly above baseline)
//   1 = RELAXED/NORMAL
//   2 = INVALID      (sensor disconnected or too many bad readings)
// -----------------------------------------------------------------------
int processGSR() {
  long sum       = 0;
  int validCount = 0;

  // Take 10 readings and filter out invalid ones
  for (int i = 0; i < 10; i++) {
    int reading = analogRead(GSR_PIN);
    if (reading > GSR_MIN_VALID_READING) {
      sum += reading;
      validCount++;
    }
    delay(10);
  }

  // Require at least 5 valid readings to proceed
  if (validCount < 5) {
    delay(10);
    return 2;
  }

  GSR_sensorValue = sum / validCount;

  int difference      = GSR_sensorValue - GSR_threshold;
  int stressThreshold = (GSR_threshold * GSR_THRESHOLD_PERCENTAGE) / 100;

  // Only flag stress on positive conductance increases (not drops)
  if (difference > stressThreshold) {
    delay(2);

    // Confirmation pass — re-sample to reduce false positives
    int confirmReading = 0;
    for (int i = 0; i < 5; i++) {
      confirmReading += analogRead(GSR_PIN);
      delay(2);
    }
    confirmReading /= 5;

    if (confirmReading < GSR_MIN_VALID_READING) {
      return 2;
    }

    difference = confirmReading - GSR_threshold;

    if (difference > stressThreshold) {
      return 0;   // Stress confirmed by both passes
    } else {
      return 1;   // Second pass did not confirm stress
    }

  } else {
    return 1;   // Within normal range
  }
}

// -----------------------------------------------------------------------
// GSRCalibrate()
// Collects GSR_CALIBRATION_SAMPLES readings over ~2.5 seconds to
// establish the user's resting baseline conductance level.
//
// Skips readings below GSR_MIN_VALID_READING to filter disconnected
// or noisy sensor states.
//
// Returns a GSRCalibrationResult with the sum and count of valid readings.
// The caller should compute the average: threshold = sum / validReadings.
// -----------------------------------------------------------------------
GSRCalibrationResult GSRCalibrate() {
  long sum        = 0;
  int validReadings = 0;

  for (int i = 0; i < GSR_CALIBRATION_SAMPLES; i++) {
    GSR_sensorValue = analogRead(GSR_PIN);

    if (GSR_sensorValue > GSR_MIN_VALID_READING) {
      sum += GSR_sensorValue;
      validReadings++;
    }

    delay(5);
  }

  GSRCalibrationResult result;
  result.sum           = sum;
  result.validReadings = validReadings;
  return result;
}


// ============================================================================
// Setup
// ============================================================================

void setup() {
  Serial.begin(115200);
  analogSetAttenuation(ADC_11db);

  // Initialize I2C for OLED
  Wire.begin(21, 22);
  Wire.setClock(100000);
  delay(500);

  // ---------------------------------------------------------------------------
  // LED and Button Setup
  // ---------------------------------------------------------------------------
  pinMode(BLUE_LED,  OUTPUT);
  pinMode(RED_LED,   OUTPUT);
  pinMode(GREEN_LED, OUTPUT);
  digitalWrite(RED_LED,   LOW);
  digitalWrite(GREEN_LED, LOW);
  pinMode(START_BTN, INPUT_PULLUP);

  // ---------------------------------------------------------------------------
  // Wi-Fi Access Point
  // ---------------------------------------------------------------------------
  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASS);
  apStarted = true;

  // ---------------------------------------------------------------------------
  // MQTT Broker
  // ---------------------------------------------------------------------------
  broker.begin();
  brokerStarted = true;

  // Subscribe to alert topic and set callback
  subscriber.setCallback(onMessageReceived);
  subscriber.subscribe(TOPIC_ALERT);
  publisher.publish(TOPIC_STATUS, "ESP32 Broker Online");

  // ---------------------------------------------------------------------------
  // Wait for user to press start button before proceeding
  // ---------------------------------------------------------------------------
  while (digitalRead(START_BTN) == HIGH) {
    delay(1000);
  }

  // ---------------------------------------------------------------------------
  // OLED Initialization
  // ---------------------------------------------------------------------------
  if (!OledScreen.begin(0x3C, false)) {
    Serial.println("OLED not found!");
  }
  delay(200);
  splashScreen();

  // System ready indicator
  if (apStarted && brokerStarted) {
    digitalWrite(GREEN_LED, HIGH);
  }

  // ---------------------------------------------------------------------------
  // GSR Sensor Check — wait until sensor is properly connected
  // ---------------------------------------------------------------------------
  while (true) {
    GSR_sensorValue = analogRead(GSR_PIN);
    if (GSR_sensorValue > GSR_MIN_VALID_READING) break;
    delay(100);
  }

  // ---------------------------------------------------------------------------
  // GSR Calibration — retries until 80% of samples are valid
  // ---------------------------------------------------------------------------
  delay(200);
  GSRCalibrationResult calib = GSRCalibrate();

  while (calib.validReadings < GSR_CALIBRATION_SAMPLES * 0.8) {
    calib = GSRCalibrate();
  }

  GSR_threshold = calib.sum / calib.validReadings;

  delay(150);
}


// ============================================================================
// Main Loop
// ============================================================================

void loop() {
  // ---------------------------------------------------------------------------
  // MQTT Keep-Alive
  // ---------------------------------------------------------------------------
  broker.loop();
  publisher.loop();

  unsigned long now = millis();
  static unsigned long lastPPGSample = 0;

  // ---------------------------------------------------------------------------
  // Alert LED Handling
  // ---------------------------------------------------------------------------
  if (alertState) {
    RedBlink();
  } else {
    digitalWrite(RED_LED, LOW);
  }

  // ---------------------------------------------------------------------------
  // OLED I2C Health Check (every 5 seconds)
  // Resets the I2C bus if the OLED stops responding.
  // ---------------------------------------------------------------------------
  if (now - lastI2CCheck > 5000) {
    lastI2CCheck = now;

    Wire.beginTransmission(0x3C);
    if (Wire.endTransmission() != 0) {
      resetI2C();
    }
  }

  // ---------------------------------------------------------------------------
  // OLED Graph Drawing
  // Maps the processed PPG signal to the lower portion of the screen
  // and draws a scrolling waveform line.
  // ---------------------------------------------------------------------------
  int mappedY = 60 - map(processedPPG, 0, 4095, 0, 45);

  if (graphCurrX >= 128) {
    graphCurrX = 0;
    graphPrevX = 0;
    OledScreen.clearDisplay();
    drawHeader();
  }

  if (now - lastGraphUpdate >= 30) {
    lastGraphUpdate = now;
    OledScreen.drawLine(graphPrevX, graphPrevY, graphCurrX, mappedY, SH110X_WHITE);
    graphPrevX = graphCurrX;
    graphPrevY = mappedY;
    graphCurrX++;
  }

  // Refresh BPM header every 1.5 seconds
  if (now - lastHeaderUpdate >= 1500) {
    lastHeaderUpdate = now;
    drawHeader();
  }

  // Push framebuffer to screen at ~10 FPS
  if (now - lastOLEDRefresh >= 100) {
    lastOLEDRefresh = now;
    OledScreen.display();
  }

  // ---------------------------------------------------------------------------
  // PPG Sampling — runs every 10ms
  // ---------------------------------------------------------------------------
  if (now - lastPPGSample >= 10) {
    lastPPGSample = now;

    // --- Warmup Phase ---
    // Allow the DC filter to settle before accepting any BPM readings.
    if (!warmupDone) {
      if (now >= WARMUP_MS) {
        // Warmup complete — reset all BPM tracking state
        warmupDone     = true;
        lastBPM        = 0;
        lastBeatTime   = 0;
        bpmBufferIndex = 0;
        bpmBufferCount = 0;
      } else {
        // During warmup: process signal to settle dcValue but discard output
        int rawPPG     = analogRead(PULSE_PIN);
        int smoothed   = smoothRaw(rawPPG);
        processSignal(smoothed);
        return;
      }
    }

    // --- Active Sampling ---
    int rawPPG    = analogRead(PULSE_PIN);
    int smoothed  = smoothRaw(rawPPG);
    processedPPG  = processSignal(smoothed);
    detectBeat(processedPPG);
  }

  // ---------------------------------------------------------------------------
  // Sensor Data Publishing — only when a valid BPM is available
  // ---------------------------------------------------------------------------
  if (lastValidBPM > 0) {

    // If no beat detected in 5 seconds, sensor likely disconnected — skip
    if (now - lastBeatTime > 5000) {
      delay(50);
      return;
    }

    // --- GSR Reading with retry (up to 5 attempts on invalid reads) ---
    int gsrStatus = 2;
    String gsrOutput = "NA";

    for (int i = 0; i < 5 && gsrStatus == 2; i++) {
      gsrStatus = processGSR();
      if (gsrStatus == 2) delay(20);
    }

    if      (gsrStatus == 0) gsrOutput = "STRESSED";
    else if (gsrStatus == 1) gsrOutput = "RELAXED/NORMAL";

    // --- Publish BPM and GSR to MQTT every 2 seconds ---
    if (now - lastPrintTime > 2000) {
      lastPrintTime = now;

      publisher.publish(TOPIC_BPM, String(lastValidBPM));
      publisher.publish(TOPIC_GSR, gsrOutput);
    }
  }
}
