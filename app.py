import gradio as gr
import torch, librosa, scipy.io.wavfile, tempfile, os, time
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    AutoModelForSeq2SeqLM, AutoTokenizer, VitsModel,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32

# ---------- Load models ----------
print("Loading English ASR (whisper-medium)...")
whisper_en_proc = WhisperProcessor.from_pretrained("openai/whisper-medium", language="english", task="transcribe")
whisper_en = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-medium", torch_dtype=dtype
).to(device).eval()

print("Loading Yoruba ASR (your fine-tuned model)...")
YORUBA_ASR = os.environ.get("YORUBA_ASR_REPO", "your-hf-username/whisper-small-yoruba-finetuned")
whisper_yo_proc = WhisperProcessor.from_pretrained("openai/whisper-small", language="yoruba", task="transcribe")
whisper_yo = WhisperForConditionalGeneration.from_pretrained(
    YORUBA_ASR, torch_dtype=dtype
).to(device).eval()
whisper_yo.generation_config.forced_decoder_ids = [
    (1, whisper_yo_proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")),
    (2, whisper_yo_proc.tokenizer.convert_tokens_to_ids("<|yo|>")),
    (3, whisper_yo_proc.tokenizer.convert_tokens_to_ids("<|transcribe|>")),
    (4, whisper_yo_proc.tokenizer.convert_tokens_to_ids("<|notimestamps|>")),
]
whisper_yo.generation_config.suppress_tokens = None

print("Loading NMT (nllb-1.3B)...")
nllb_tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
nllb = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-distilled-1.3B", torch_dtype=dtype
).to(device).eval()

print("Loading TTS (mms-tts-yor)...")
mms = VitsModel.from_pretrained("facebook/mms-tts-yor").to(device).eval()
mms_tok = AutoTokenizer.from_pretrained("facebook/mms-tts-yor")

print("All models loaded on", device)


def run_pipeline(audio_path, source_lang):
    if audio_path is None:
        return "No audio provided.", "", None, ""

    t0 = time.time()
    timings = {}

    # ---- ASR ----
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    t = time.time()
    if source_lang == "Yoruba":
        proc, mdl = whisper_yo_proc, whisper_yo
    else:
        proc, mdl = whisper_en_proc, whisper_en
    inputs = proc(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device).to(dtype)
    with torch.no_grad():
        ids = mdl.generate(inputs, max_new_tokens=225, language=("en" if source_lang == "English" else "yo"))
    asr_text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    timings["ASR"] = round(time.time() - t, 2)

    # ---- NMT ----
    t = time.time()
    if source_lang == "English":
        src, tgt = "eng_Latn", "yor_Latn"
    else:
        src, tgt = "yor_Latn", "eng_Latn"
    nllb_tok.src_lang = src
    nllb_in = nllb_tok(asr_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = nllb.generate(
            **nllb_in,
            forced_bos_token_id=nllb_tok.convert_tokens_to_ids(tgt),
            max_length=200, num_beams=5,
            no_repeat_ngram_size=3, repetition_penalty=1.3,
        )
    translation = nllb_tok.batch_decode(out, skip_special_tokens=True)[0].strip()
    timings["NMT"] = round(time.time() - t, 2)

    # ---- TTS (Yoruba only) ----
    audio_out = None
    if tgt == "yor_Latn":
        t = time.time()
        mms_in = mms_tok(translation, return_tensors="pt").to(device)
        with torch.no_grad():
            waveform = mms(**mms_in).waveform.cpu().numpy().squeeze()
        sr = mms.config.sampling_rate
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        scipy.io.wavfile.write(out_path, sr, (waveform * 32767).astype("int16"))
        audio_out = out_path
        timings["TTS"] = round(time.time() - t, 2)

    timings["Total"] = round(time.time() - t0, 2)
    timing_str = " | ".join(f"{k}: {v}s" for k, v in timings.items())
    return asr_text, translation, audio_out, timing_str


with gr.Blocks(title="Yoruba–English Speech-to-Speech Translation") as demo:
    gr.Markdown(
        "# Bilingual Yoruba–English Speech-to-Speech Translation\n"
        "Fine-tuned Whisper (Yoruba) + Whisper-medium (English) → NLLB-1.3B → MMS-TTS-Yoruba\n\n"
        "*Yoruba → English outputs text only (MMS is Yoruba-output only).*"
    )
    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(type="filepath", label="Input audio (short clip ≤ 15s recommended)")
            lang     = gr.Radio(["English", "Yoruba"], value="English", label="Source language")
            btn      = gr.Button("Translate", variant="primary")
        with gr.Column():
            asr_out  = gr.Textbox(label="ASR transcript")
            nmt_out  = gr.Textbox(label="Translation")
            audio_out = gr.Audio(label="Synthesized Yoruba audio", type="filepath")
            timing   = gr.Textbox(label="Latency breakdown", interactive=False)

    btn.click(run_pipeline, [audio_in, lang], [asr_out, nmt_out, audio_out, timing])

    gr.Markdown(
        "**Model configuration:**\n"
        "- English ASR: `openai/whisper-medium`\n"
        "- Yoruba ASR: **fine-tuned checkpoint** (trained on 484 utterances, held-out WER 0.591)\n"
        "- NMT: `facebook/nllb-200-distilled-1.3B`\n"
        "- TTS: `facebook/mms-tts-yor`\n"
    )

demo.launch()