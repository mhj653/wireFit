# Wire Test Data Workflow

## 1. Generate synthetic samples

```bat
generate_wire_samples.cmd
```

Default output:

```text
SampleData\WireSamples\261011
```

Use this folder directly in Camera Capture -> virtual camera mode.

## 2. Capture data on the field PC

1. Open Camera Capture.
2. Connect the D405.
3. Keep `자동 저장` enabled.
4. Click `이미지 캡쳐`, or capture from Hole Detection.

Saved data goes to:

```text
Logs\<YYMMDD>\
  color\
  depth\
  depth_raw\
  depth_scale.json
```

`depth_raw` is the uint16 depth image used again by Wire Detection.

## 3. Replay data on the debug PC

1. Copy the whole date folder, for example `Logs\261011`.
2. Open Camera Capture.
3. Select virtual camera mode.
4. Choose the copied date folder.

The app will replay the saved color image and raw depth together.
