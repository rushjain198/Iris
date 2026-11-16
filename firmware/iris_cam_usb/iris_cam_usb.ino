// Iris glasses camera over the USB-C cable (no WiFi): Seeed XIAO ESP32-S3 Sense camera board.
// Each frame is sent as:  0xA5 0x5A 'J' 'P' | uint32 little-endian length | JPEG bytes
// Host commands (text lines): "F<n>" framesize (6 = 320x240, 10 = 640x480), "Q<n>" JPEG quality (10-63)
// Board: XIAO_ESP32S3, Tools > PSRAM = OPI PSRAM, USB Mode = Hardware CDC and JTAG, USB CDC On Boot = Enabled.
// No light source on this board: nothing bright near anyone's eyes.

#include "esp_camera.h"

#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 10
#define SIOD_GPIO_NUM 40
#define SIOC_GPIO_NUM 39
#define Y9_GPIO_NUM 48
#define Y8_GPIO_NUM 11
#define Y7_GPIO_NUM 12
#define Y6_GPIO_NUM 14
#define Y5_GPIO_NUM 16
#define Y4_GPIO_NUM 18
#define Y3_GPIO_NUM 17
#define Y2_GPIO_NUM 15
#define VSYNC_GPIO_NUM 38
#define HREF_GPIO_NUM 47
#define PCLK_GPIO_NUM 13

static const uint8_t MAGIC[4] = {0xA5, 0x5A, 'J', 'P'};
static char cmd[16];
static uint8_t cmd_len = 0;

static void handle_command() {
  sensor_t *s = esp_camera_sensor_get();
  int v = atoi(cmd + 1);
  if (cmd[0] == 'F' && v >= 0 && v <= 13) s->set_framesize(s, (framesize_t)v);
  else if (cmd[0] == 'Q' && v >= 10 && v <= 63) s->set_quality(s, v);
}

static void read_commands() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (cmd_len) { cmd[cmd_len] = 0; handle_command(); }
      cmd_len = 0;
    } else if (cmd_len < sizeof(cmd) - 1) {
      cmd[cmd_len++] = c;
    }
  }
}

void setup() {
  Serial.setTxBufferSize(16384);
  Serial.begin(2000000);                   // baud is ignored by USB CDC; throughput is USB full-speed
  Serial.setTxTimeoutMs(200);              // wait for the host instead of dropping bytes (dropped bytes = broken JPEGs)

  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM; c.pin_d1 = Y3_GPIO_NUM; c.pin_d2 = Y4_GPIO_NUM; c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM; c.pin_d5 = Y7_GPIO_NUM; c.pin_d6 = Y8_GPIO_NUM; c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM; c.pin_pclk = PCLK_GPIO_NUM; c.pin_vsync = VSYNC_GPIO_NUM; c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM; c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM; c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size = FRAMESIZE_VGA;            // 640x480 default (doctor-worn: more pixels on the patient's eye)
  c.jpeg_quality = 14;
  c.fb_count = 2;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;
  if (esp_camera_init(&c) != ESP_OK) {
    while (true) { Serial.println("camera init FAILED"); delay(1000); }
  }
  sensor_t *s = esp_camera_sensor_get();
  s->set_whitebal(s, 1);
}

void loop() {
  read_commands();
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) { delay(5); return; }
  uint32_t n = fb->len;
  Serial.write(MAGIC, 4);
  Serial.write((const uint8_t *)&n, 4);
  Serial.write(fb->buf, fb->len);
  esp_camera_fb_return(fb);
}
