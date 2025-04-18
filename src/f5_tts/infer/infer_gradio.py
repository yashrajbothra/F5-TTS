# ruff: noqa: E402
# Above allows ruff to ignore E402: module level import not at top of file

import gc
import json
import re
import tempfile
from collections import OrderedDict
from importlib.resources import files

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio
from cached_path import cached_path
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import spaces
    USING_SPACES = True
except ImportError:
    USING_SPACES = False


def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func


from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)


DEFAULT_TTS_MODEL = "F5-TTS_v1"
tts_model_choice = DEFAULT_TTS_MODEL

DEFAULT_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]


# Load models
vocoder = load_vocoder()

# Set the number of threads PyTorch will use for CPU operations
num_cores = torch.get_num_threads()
print(f"Default number of PyTorch threads: {num_cores}")
desired_cores = torch.get_num_threads()  # Use all available logical cores
torch.set_num_threads(desired_cores)
print(f"Setting PyTorch to use {torch.get_num_threads()} threads for CPU.")


def load_f5tts():
    ckpt_path = str(cached_path(DEFAULT_TTS_MODEL_CFG[0]))
    F5TTS_model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    return load_model(DiT, F5TTS_model_cfg, ckpt_path)


F5TTS_ema_model = load_f5tts()

@gpu_decorator
def generate_response(messages, model, tokenizer):
    """Generate response using Qwen"""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


@gpu_decorator
def infer(
    ref_audio_orig,
    ref_text,
    gen_text,
    model,
    remove_silence,
    cross_fade_duration=0.15,
    nfe_step=32,
    speed=1,
    show_info=gr.Info,
):
    if not ref_audio_orig:
        gr.Warning("Please provide reference audio.")
        return gr.update(), gr.update(), ref_text

    if not gen_text.strip():
        gr.Warning("Please enter text to generate.")
        return gr.update(), gr.update(), ref_text

    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)

    # Use only the default model
    ema_model = F5TTS_ema_model

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
        show_info=show_info,
        progress=gr.Progress(),
    )

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Save the spectrogram
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave), spectrogram_path, ref_text


# Custom CSS styles for the app
custom_css = """
/* Main app styling */
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    color: #1a1a1a;
    background-color: #f7f7f7;
}

/* App title and logo styling */
h1.app-title {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.app-subtitle {
    font-size: 1.25rem;
    font-weight: 500;
    color: #4b5563;
    margin-bottom: 2rem;
}

/* Card styling */
.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 2rem;
    background-color: white;
    border-radius: 1rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
}

/* Upload area styling */
.upload-area {
    border: 2px dashed #d1d5db;
    border-radius: 0.75rem;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    transition: all 0.3s ease;
}

.upload-area:hover {
    border-color: #6b7280;
    background-color: #f9fafb;
}

.upload-icon {
    font-size: 2rem;
    color: #6b7280;
    margin-bottom: 1rem;
}

.upload-title {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.upload-subtitle {
    font-size: 0.875rem;
    color: #6b7280;
}

/* Record button styling */
.record-button {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border: 1px solid #d1d5db;
    border-radius: 0.5rem;
    background-color: white;
    color: #4b5563;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
}

.record-button:hover {
    background-color: #f9fafb;
    border-color: #9ca3af;
}

/* Text input styling */
.text-area-label {
    font-size: 1.125rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.text-area {
    width: 100%;
    min-height: 120px;
    padding: 0.75rem;
    border: 1px solid #d1d5db;
    border-radius: 0.5rem;
    font-size: 1rem;
    font-family: inherit;
    resize: vertical;
}

/* Generate button styling */
.generate-button {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    width: 100%;
    padding: 0.875rem 1.5rem;
    border: none;
    border-radius: 0.5rem;
    font-size: 1.125rem;
    font-weight: 600;
    color: white;
    background: linear-gradient(90deg, #4f46e5 0%, #9333ea 100%);
    cursor: pointer;
    transition: all 0.3s ease;
    margin-top: 1rem;
}

.generate-button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}

.generate-button:active {
    transform: translateY(1px);
}

/* Sound icon */
.sound-icon {
    font-size: 1.25rem;
}

/* Footer styling */
#custom-footer {
    text-align: center;
    font-weight: 500;
    margin-top: 2rem;
    color: #6b7280;
}

/* Advanced settings accordion */
.accordion {
    border: 1px solid #e5e7eb;
    border-radius: 0.5rem;
    margin-bottom: 1.5rem;
}

.accordion-header {
    padding: 1rem;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.accordion-content {
    padding: 1rem;
    border-top: 1px solid #e5e7eb;
}

/* Hide footer */
footer {
    display: none !important;
}

/* Slider styling */
.slider-container {
    margin-bottom: 1rem;
}

.slider-label {
    font-weight: 500;
    margin-bottom: 0.25rem;
}

.slider {
    width: 100%;
}

/* Output audio styling */
.audio-output {
    margin-top: 1.5rem;
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #f9fafb;
}

/* Make sure Gradio elements follow our design */
.gradio-container {
    max-width: 100% !important;
}

/* Remove top-right corner elements */
.icon-button-wrapper.top-panel.hide-top-corner {
    display: none !important;
}
"""

# Custom theme configuration
theme = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="gray",
    font=["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=["Roboto Mono", "ui-monospace", "SFMono-Regular", "monospace"],
).set(
    body_text_color="#1a1a1a",
    body_background_fill="#f7f7f7",
    block_background_fill="#ffffff",
    block_border_width="0px",
    block_shadow="0 4px 6px rgba(0, 0, 0, 0.05)",
    button_primary_background_fill="linear-gradient(90deg, #4f46e5 0%, #9333ea 100%)",
    button_primary_background_fill_hover="linear-gradient(90deg, #4338ca 0%, #7e22ce 100%)",
    button_primary_text_color="#ffffff",
    button_border_width="1px",
    checkbox_background_color="#f3f4f6",
    checkbox_background_color_selected="#4f46e5",
    checkbox_border_color="#d1d5db",
    checkbox_border_color_focus="#4f46e5",
    checkbox_border_color_hover="#9ca3af",
    checkbox_border_color_selected="#4f46e5",
    slider_color="#4f46e5",
    input_border_color="#d1d5db",
    input_border_width="1px",
    input_background_fill="#ffffff",
    input_shadow="none",
    input_shadow_focus="0 0 0 3px rgba(79, 70, 229, 0.2)",
    input_border_color_focus="#4f46e5",
)

with gr.Blocks(title="TalkClone", theme=theme, css=custom_css) as app:
    with gr.Row(elem_classes="container"):
        with gr.Column():
            # App title and logo
            gr.HTML("""
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right: 1rem;">
                        <rect width="48" height="48" rx="8" fill="#1a1a1a" fill-opacity="0.05"/>
                        <path d="M18 12C18 10.3431 19.3431 9 21 9H24C25.6569 9 27 10.3431 27 12V30C27 31.6569 25.6569 33 24 33H21C19.3431 33 18 31.6569 18 30V12Z" fill="#1a1a1a"/>
                        <path d="M30 18C30 16.3431 31.3431 15 33 15H36C37.6569 15 39 16.3431 39 18V24C39 25.6569 37.6569 27 36 27H33C31.3431 27 30 25.6569 30 24V18Z" fill="#1a1a1a"/>
                        <path d="M30 36C30 34.3431 28.6569 33 27 33H19C17.3431 33 16 34.3431 16 36V36C16 37.6569 17.3431 39 19 39H27C28.6569 39 30 37.6569 30 36V36Z" fill="#1a1a1a"/>
                    </svg>
                    <h1 class="app-title">TalkClone</h1>
                </div>
                <p class="app-subtitle">Turn text into speech using your reference audio</p>
            """)
            
            # Reference audio upload section
            with gr.Box(elem_classes="upload-area"):
                gr.HTML("""
                    <div class="upload-icon">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M12 4V16M12 4L8 8M12 4L16 8" stroke="#6b7280" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M20 16V18C20 19.1046 19.1046 20 18 20H6C4.89543 20 4 19.1046 4 18V16" stroke="#6b7280" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                    </div>
                    <div class="upload-title">Upload Reference Audio</div>
                    <div class="upload-subtitle">Drag & drop or click to upload</div>
                """)
                ref_audio_input = gr.Audio(
                    label="",
                    type="filepath",
                    max_length=15,
                    min_length=4,
                    elem_classes="hidden-audio-input",
                    show_label=False
                )
            
            # Record audio button
            with gr.Row(elem_classes="record-button-container"):
                gr.HTML("""
                    <button class="record-button">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M12 14C13.6569 14 15 12.6569 15 11V5C15 3.34315 13.6569 2 12 2C10.3431 2 9 3.34315 9 5V11C9 12.6569 10.3431 14 12 14Z" stroke="#4b5563" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M19 11V12C19 15.866 15.866 19 12 19C8.13401 19 5 15.866 5 12V11" stroke="#4b5563" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M12 19V22" stroke="#4b5563" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                        Record your voice
                    </button>
                """)
                
            # Text input section
            gr.Markdown("## Text to Generate", elem_classes="text-area-label")
            gen_text_input = gr.Textbox(
                placeholder="Type your script here...",
                lines=6,
                max_lines=10,
                elem_classes="text-area"
            )
            
            # Generate button
            generate_btn = gr.Button(
                "Generate Voice", 
                variant="primary",
                elem_classes="generate-button"
            )
            gr.HTML("""
                <script>
                    // Update the generate button with icon
                    document.addEventListener('DOMContentLoaded', function() {
                        const generateBtn = document.querySelector('.generate-button');
                        if (generateBtn) {
                            generateBtn.innerHTML = `
                                <svg class="sound-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M11 5L6 9H2V15H6L11 19V5Z" fill="white" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                    <path d="M15.54 8.46C16.4774 9.39764 17.004 10.6692 17.004 12C17.004 13.3308 16.4774 14.6024 15.54 15.54" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                    <path d="M19.07 5.93C20.9447 7.80528 21.9791 10.3447 21.9791 13C21.9791 15.6553 20.9447 18.1947 19.07 20.07" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                </svg>
                                Generate Voice
                            `;
                        }
                    });
                </script>
            """)
            
            # Advanced settings
            with gr.Accordion("Advanced Settings", open=False, elem_classes="accordion"):
                ref_text_input = gr.Textbox(
                    label="Reference Text",
                    info="Leave blank to automatically transcribe the reference audio. If you enter text it will override automatic transcription.",
                    lines=2
                )
                remove_silence = gr.Checkbox(
                    label="Remove Silences",
                    info="The model tends to produce silences, especially on longer audio. We can manually remove silences if needed. Note that this is an experimental feature and may produce strange results. This will also increase generation time.",
                    value=False
                )
                speed_slider = gr.Slider(
                    label="Speed",
                    minimum=0.3,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    info="Adjust the speed of the audio.",
                    elem_classes="slider-container"
                )
                nfe_slider = gr.Slider(
                    label="NFE Steps",
                    minimum=4,
                    maximum=64,
                    value=32,
                    step=2,
                    info="Set the number of denoising steps.",
                    elem_classes="slider-container"
                )
                cross_fade_duration_slider = gr.Slider(
                    label="Cross-Fade Duration (s)",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.15,
                    step=0.01,
                    info="Set the duration of the cross-fade between audio clips.",
                    elem_classes="slider-container"
                )
            
            # Output audio section
            audio_output = gr.Audio(
                label="Synthesized Audio", 
                elem_classes="audio-output"
            )

            # Footer
            gr.HTML(
                '<div id="custom-footer">Made With Love❤️ From Noman Elahi Dashti</div>'
            )
    
    # Function handling
    @gpu_decorator
    def basic_tts(
        ref_audio_input,
        ref_text_input,
        gen_text_input,
        remove_silence,
        cross_fade_duration_slider,
        nfe_slider,
        speed_slider,
    ):
        audio_out, spectrogram_path, ref_text_out = infer(
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            tts_model_choice,
            remove_silence,
            cross_fade_duration=cross_fade_duration_slider,
            nfe_step=nfe_slider,
            speed=speed_slider,
        )
        return audio_out, spectrogram_path, ref_text_out

    generate_btn.click(
        basic_tts,
        inputs=[
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            remove_silence,
            cross_fade_duration_slider,
            nfe_slider,
            speed_slider,
        ],
        outputs=[audio_output, ref_text_input],
    )


@click.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
@click.option("--host", "-H", default=None, help="Host to run the app on")
@click.option(
    "--share",
    "-s",
    default=False,
    is_flag=True,
    help="Share the app via Gradio share link",
)
@click.option("--api", "-a", default=False, is_flag=True, help="Allow API access")
@click.option(
    "--root_path",
    "-r",
    default=None,
    type=str,
    help='The root path (or "mount point") of the application, if it\'s not served from the root ("/") of the domain.',
)
@click.option(
    "--inbrowser",
    "-i",
    is_flag=True,
    default=False,
    help="Automatically launch the interface in the default web browser",
)
def main(port, host, share, api, root_path, inbrowser):
    app.queue(api_open=api).launch(
        server_name=host,
        server_port=port,
        share=share,
        show_api=api,
        root_path=root_path,
        inbrowser=inbrowser,
        pwa=True,
    )


if __name__ == "__main__":
    if not USING_SPACES:
        main()
    else:
        app.queue().launch()