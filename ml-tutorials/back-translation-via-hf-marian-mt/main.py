import logging

from transformers import MarianMTModel, MarianTokenizer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def translate(model_name, src_text, max_length=64):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    translated_ids = model.generate(**tokenizer(src_text, padding='max_length', max_length=max_length, return_tensors='pt'))
    out_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_ids]
    return out_text


def main():
    src_text = [
        "this is a sentence in english that we want to translate to french",
        "We show that encoder-only models have strong transfer performance while encoder-decoder models perform better on textual similarity tasks.",
        "We also demonstrate the effectiveness of scaling up the model size, which greatly improves sentence embedding quality."
    ]

    out_text = translate("Helsinki-NLP/opus-mt-en-fr", src_text)
    back_translated_text = translate("Helsinki-NLP/opus-mt-fr-en", out_text)

    logger.debug(f"src_text : {src_text}")
    logger.debug(f"back_translated_text {back_translated_text}")


if __name__ == "__main__":
    main()
