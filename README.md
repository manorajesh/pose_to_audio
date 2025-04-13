# pose to video

This small Python project translates hand gestures, including American Sign Language (ASL), into musical expressions through a digital modular synthesizer. By tracking your hand movements in real-time, the script converts spatial positioning, finger movements, and hand gestures into Open Sound Control (OSC) messages that can control various synthesizer parameters.

## Features

- Real-time hand tracking using MediaPipe and OpenCV
- Tracks multiple hand features simultaneously:
  - Finger curl detection
  - Hand velocity measurement
  - Distance between hands
  - Vertical positioning
- OSC protocol integration for seamless connection to digital audio workstations and modular synthesizers
- Works with both webcam input and video files
- Supports tracking of both hands simultaneously

## Requirements

- Python 3.6+
- OpenCV
- MediaPipe
- python-osc
- A compatible OSC receiver (VCV Rack, Max/MSP, Pure Data, etc.)

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/manorajesh/pose_vcv_plugin.git
   cd pose_to_audio
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use: env\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install opencv-python mediapipe python-osc
   ```

## Usage

### Basic Usage

Run the application with your webcam:

```bash
python src/main.py
```

### Using a Video File

You can also use a pre-recorded video:

```bash
python src/main.py path/to/your/video.mp4
```

### OSC Parameters

The application sends OSC messages to `127.0.0.1:7001` with the following channels:

- `/ch/1`: Finger curl intensity (average curl of all fingers)
- `/ch/2`: Hand movement velocity
- `/ch/3`: Distance between hands (when both hands are visible)
- `/ch/4`: Vertical position of hands

## How It Works

1. **Hand Detection and Tracking**: Using MediaPipe's hand tracking solution, the application identifies hand landmarks in each video frame.

2. **Feature Extraction**: The system extracts key features from the hand tracking data:

   - Finger curl calculation based on joint angles
   - Movement velocity through frame-to-frame position changes
   - Inter-hand distance measurement
   - Spatial positioning of hands

3. **Data Processing**: Raw tracking data is smoothed to remove jitter and improve responsiveness.

4. **OSC Transmission**: Processed data is converted to normalized values and transmitted as OSC messages.

5. **Audio Generation**: When connected to a compatible synthesizer (like VCV Rack), these OSC messages control various sound parameters to create dynamic, gesture-responsive audio.

## American Sign Language Integration

While PoseInfo tracks general hand movements, it can respond uniquely to ASL gestures:

- Different finger configurations in ASL signs create distinctive curl patterns
- The spatial arrangement of ASL signs (location, movement, orientation) generates varied OSC outputs
- Speed and intensity of signing directly affects the velocity parameters

This creates an expressive system where ASL can be "performed" as a musical instrument, with each sign and movement contributing to the sonic landscape.

## Connecting to Synthesizers

1. Ensure your synthesizer or DAW can receive OSC messages on port 7001 (configurable in the code)
2. Map the incoming OSC channels to desired parameters:
   - Map finger curl data to filter cutoffs for gesture-controlled filtering
   - Use velocity data to control note velocity or LFO rates
   - Map hand distance to effects like reverb depth or delay feedback
   - Use vertical positioning to control pitch or modulation intensity

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
