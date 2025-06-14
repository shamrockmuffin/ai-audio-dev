# Audio Analysis & Transcription - Streamlit App

A modern, feature-rich audio analysis and transcription application built with Streamlit. This application provides powerful audio processing capabilities including enhancement, transcription, and AI-powered analysis.

## Features

- **Audio Analysis**: Visualize waveforms and spectrograms
- **Audio Enhancement**: Noise reduction, normalization, and EQ enhancement
- **Transcription**: High-quality speech-to-text using Whisper
- **AI Enhancement**: Improve transcription quality with Claude
- **Interactive Chat**: AI assistant for audio-related queries
- **Multiple Export Formats**: TXT, JSON, SRT
- **Session Management**: Save and restore your work

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-audio
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp env.example .env
# Edit .env and add your Anthropic API key
```

## Usage

1. Start the application:
```bash
streamlit run main.py
```

2. Open your browser to http://localhost:8501

3. Upload an audio file using the sidebar

4. Explore the different tabs:
   - **Audio Analysis**: View waveform and spectrogram
   - **Enhancement**: Apply audio improvements
   - **Transcription**: Convert speech to text
   - **AI Assistant**: Chat about your audio

## Project Structure

```
audio_analyzer/
├── config/           # Configuration and settings
├── core/            # Core processing modules
├── services/        # External service integrations
├── ui/              # UI components and state management
├── utils/           # Utility functions
├── main.py          # Main application entry point
└── requirements.txt # Project dependencies
```

## Configuration

Key settings can be adjusted in `config/settings.py`:

- `MAX_AUDIO_LENGTH`: Maximum audio duration (seconds)
- `CHUNK_SIZE`: Processing chunk size (seconds)
- `DEFAULT_SAMPLE_RATE`: Default audio sample rate
- `MAX_FILE_SIZE_MB`: Maximum upload file size

## Advanced Features

### Custom Enhancement Pipeline

The application supports a customizable enhancement pipeline. You can add or remove processing stages:

```python
from core.enhancement_pipeline import EnhancementPipeline, PipelineStage

pipeline = EnhancementPipeline()
pipeline.add_stage(PipelineStage(
    name="Custom Filter",
    function=my_custom_filter,
    params={'param1': value}
))
```

### Session Management

Save your work for later:
- Use the session manager to save current state
- Load previous sessions from the sidebar
- Export session data for backup

## API Keys

This application requires an Anthropic API key for AI features. Get your key from:
https://console.anthropic.com/

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `MAX_AUDIO_LENGTH` in settings
2. **Slow Processing**: Enable GPU in settings if available
3. **API Errors**: Check your Anthropic API key is valid

### Debug Mode

Enable debug mode in `.env`:
```
DEBUG=True
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project is licensed under the MIT License. 