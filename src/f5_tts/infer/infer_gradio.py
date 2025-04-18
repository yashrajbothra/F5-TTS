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
# Optionally, for inter-operator parallelism (less common for inference on CPU):
# torch.set_num_interop_threads(desired_cores)
# print(f"Setting PyTorch inter-op threads to {torch.get_num_interop_threads()}.")


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
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
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


remove_audio_css = """
.icon-button-wrapper.top-panel.hide-top-corner {
    display: none !important;
}
"""
custom_theme = gr.themes.Default(
    primary_hue="green",
    secondary_hue="gray",
    neutral_hue="slate"
).set(
    body_background_fill="#111827",
    body_text_color="#F9FAFB",
    block_background_fill="#1F2937",
    block_border_color="#10B981",
    input_background_fill="#374151",
    input_border_color="#10B981",
    button_primary_background_fill="#B4FD83",
    button_primary_background_fill_hover="#A5F070",
    button_primary_text_color="#111827"
)

# Add custom CSS to enforce button styles
custom_css = """
button.primary {
    background-color: #B4FD83 !important;
    color: #111827 !important;
}
button.primary:hover {
    background-color: #A5F070 !important;
}
footer {
    display: none !important;
}
#custom-footer {
    text-align: center;
    font-weight: 500;
    margin-top: 2rem;
}
"""

with gr.Blocks(theme=custom_theme, css=custom_css) as app_tts:
    gr.HTML("""
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
         <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" fill="#FFFFFF" version="1.1" id="Capa_1" viewBox="0 0 344.156 344.156" style="margin-right: 0.5rem; width: 2.5rem; height: 2.5rem;">
<g>
	<path d="M343.766,28.723c0-5.525-4.483-10.006-10.006-10.006H106.574c-5.531,0-10.006,4.48-10.006,10.006v194.18   c-10.25-8.871-23.568-14.279-38.156-14.279C26.207,208.623,0,234.824,0,267.029c0,32.209,26.207,58.41,58.412,58.41   c32.215,0,58.412-26.201,58.412-58.41c0-2.854-0.246-175.924-0.246-175.924h207.176v131.666   c-10.229-8.795-23.487-14.148-38.008-14.148c-32.217,0-58.412,26.201-58.412,58.406c0,32.209,26.195,58.41,58.412,58.41   c32.205,0,58.41-26.201,58.41-58.41C344.156,264.068,343.766,28.723,343.766,28.723z M58.412,305.43   c-21.174,0-38.4-17.227-38.4-38.4c0-21.17,17.227-38.396,38.4-38.396s38.4,17.228,38.4,38.396   C96.812,288.203,79.586,305.43,58.412,305.43z M116.578,71.094V38.728h207.176v32.365L116.578,71.094L116.578,71.094z    M285.746,305.43c-21.174,0-38.4-17.227-38.4-38.4c0-21.17,17.228-38.396,38.4-38.396s38.4,17.228,38.4,38.396   C324.146,288.203,306.92,305.43,285.746,305.43z"/>
</g>
</svg>
            <h1 class="app-title">TalkClone</h1>
        </div>
        <p class="app-subtitle">Turn text into speech using your reference audio</p>
    """)

    ref_audio_input = gr.Audio(label="Reference Audio", type="filepath", max_length=15, min_length=4)
    gen_text_input = gr.Textbox(label="Text to Generate", lines=10, placeholder="Type your script here...")
    generate_btn = gr.Button("Generate Voice", variant="primary")
    with gr.Accordion("Advanced Settings", open=False):
        ref_text_input = gr.Textbox(
            label="Reference Text",
            info="Leave blank to automatically transcribe the reference audio. If you enter text it will override automatic transcription.",
            lines=2,
        )
        remove_silence = gr.Checkbox(
            label="Remove Silences",
            info="The model tends to produce silences, especially on longer audio. We can manually remove silences if needed. Note that this is an experimental feature and may produce strange results. This will also increase generation time.",
            value=False,
        )
        speed_slider = gr.Slider(
            label="Speed",
            minimum=0.3,
            maximum=2.0,
            value=1.0,
            step=0.1,
            info="Adjust the speed of the audio.",
        )
        nfe_slider = gr.Slider(
            label="NFE Steps",
            minimum=4,
            maximum=64,
            value=32,
            step=2,
            info="Set the number of denoising steps.",
        )
        cross_fade_duration_slider = gr.Slider(
            label="Cross-Fade Duration (s)",
            minimum=0.0,
            maximum=1.0,
            value=0.15,
            step=0.01,
            info="Set the duration of the cross-fade between audio clips.",
        )

    audio_output = gr.Audio(label="Synthesized Audio")

    gr.HTML(
        '<div id="custom-footer">Made With Love❤️ From Noman Elahi Dashti</div>'
    )

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


def parse_speechtypes_text(gen_text):
    # Pattern to find {speechtype}
    pattern = r"\{(.*?)\}"

    # Split the text by the pattern
    tokens = re.split(pattern, gen_text)

    segments = []

    current_style = "Regular"

    for i in range(len(tokens)):
        if i % 2 == 0:
            # This is text
            text = tokens[i].strip()
            if text:
                segments.append({"style": current_style, "text": text})
        else:
            # This is style
            style = tokens[i].strip()
            current_style = style

    return segments


with gr.Blocks(
    title="Talkclone",  # 👈 this sets the browser tab title
    theme=custom_theme,
    css="""
        footer { visibility: hidden; }
        #custom-footer {
            text-align: center;
            font-weight: bold;
            margin-top: 2rem;
        }
    """
) as app:
    app = app_tts

    gr.Markdown(
        "Made With Love❤️ From Noman Elahi Dashti",
        elem_id="custom-footer"
    )

@click.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
@click.option("--host", "-H", default=None, help="Host to run the app on")
@click.option(
    "--share",
    "-s",
    default=False,
    is_flag=True,
    help="Share the app via Talkclone share link",
)
@click.option("--api", "-a", default=False, is_flag=True, help="Allow API access")
@click.option(
    "--root_path",
    "-r",
    default=None,
    type=str,
    help='The root path (or "mount point") of the application, if it\'s not served from the root ("/") of the domain. Often used when the application is behind a reverse proxy that forwards requests to the application, e.g. set "/myapp" or full URL for application served at "https://example.com/myapp".',
)
@click.option(
    "--inbrowser",
    "-i",
    is_flag=True,
    default=False,
    help="Automatically launch the interface in the default web browser",
)
def main(port, host, share, api, root_path, inbrowser):
    global app
    print("Starting app...")
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
