# embedding_model.py
# Lazy helper to get wav2vec embeddings; used when USE_W2V=true

def get_wav2vec_embedding_factory(model_name="facebook/wav2vec2-base"):
    try:
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
        import torch
    except Exception as e:
        raise RuntimeError("transformers/torch required for Wav2Vec embeddings: " + str(e))

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()

    def extract(y, sr):
        inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(inputs.input_values).last_hidden_state
        emb = outputs.mean(dim=1).squeeze(0).cpu().numpy()
        return emb

    return extract
