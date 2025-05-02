/*
Brain-Controlled Robot — Arduino Command Interpreter

This sketch receives EEG-based movement predictions as JSON over Serial,
parses them using ArduinoJson, and controls motors accordingly.

*/

#include <ArduinoJson.h>

// Motor control pins.
const int motorLeftFwd = 5;
const int motorLeftRev = 6;
const int motorRightFwd = 9;
const int motorRightRev = 10;

void setup() {
  Serial.begin(9600);

  pinMode(motorLeftFwd, OUTPUT);
  pinMode(motorLeftRev, OUTPUT);
  pinMode(motorRightFwd, OUTPUT);
  pinMode(motorRightRev, OUTPUT);

  stopMotors();
  Serial.println("[INFO] Robot ready for commands.");
}

void loop() {
  static String incoming = "";

  // Read full JSON line from Serial.
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      handleMessage(incoming);
      incoming = "";
    } else {
      incoming += c;
    }
  }
}

/**
 * Message parsing and dispatching.
 */
void handleMessage(const String& msg) {
  StaticJsonDocument<128> doc;
  DeserializationError error = deserializeJson(doc, msg);
  if (error) {
    Serial.print("[ERROR] JSON Parse Error: ");
    Serial.println(error.c_str());
    return;
  }

  const char* command = doc["class"];
  float confidence = doc["confidence"];

  Serial.print("[INFO] Command: ");
  Serial.print(command);
  Serial.print(" (conf: ");
  Serial.print(confidence, 2);
  Serial.println(")");

  if (confidence < 0.6) {
    Serial.println("[INFO] Low confidence — ignoring.");
    return;
  }

  if (strcmp(command, "FORWARD") == 0) {
    moveForward();
  } else if (strcmp(command, "BACKWARD") == 0) {
    moveBackward();
  } else if (strcmp(command, "LEFT") == 0) {
    turnLeft();
  } else if (strcmp(command, "RIGHT") == 0) {
    turnRight();
  } else if (strcmp(command, "STOP") == 0) {
    stopMotors();
  }
}

/**
 * Motor Control Functions.
 */
void moveForward() {
  digitalWrite(motorLeftFwd, HIGH);
  digitalWrite(motorLeftRev, LOW);
  digitalWrite(motorRightFwd, HIGH);
  digitalWrite(motorRightRev, LOW);
  Serial.println("[INFO] Moving FORWARD");
}

void moveBackward() {
  digitalWrite(motorLeftFwd, LOW);
  digitalWrite(motorLeftRev, HIGH);
  digitalWrite(motorRightFwd, LOW);
  digitalWrite(motorRightRev, HIGH);
  Serial.println("[INFO] Moving BACKWARD");
}

void turnLeft() {
  digitalWrite(motorLeftFwd, LOW);
  digitalWrite(motorLeftRev, HIGH);
  digitalWrite(motorRightFwd, HIGH);
  digitalWrite(motorRightRev, LOW);
  Serial.println("[INFO] Turning LEFT");
}

void turnRight() {
  digitalWrite(motorLeftFwd, HIGH);
  digitalWrite(motorLeftRev, LOW);
  digitalWrite(motorRightFwd, LOW);
  digitalWrite(motorRightRev, HIGH);
  Serial.println("[INFO] Turning RIGHT");
}

void stopMotors() {
  digitalWrite(motorLeftFwd, LOW);
  digitalWrite(motorLeftRev, LOW);
  digitalWrite(motorRightFwd, LOW);
  digitalWrite(motorRightRev, LOW);
  Serial.println("[INFO] STOP");
}
