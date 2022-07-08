from transformers import MarianMTModel, MarianTokenizer


def translate_en_to_fr():
    src_text = [
        "this is a sentence in english that we want to translate to french",
        "I am a boy",
        "I am hungry"
    ]

    model_name = "Helsinki-NLP/opus-mt-en-fr"

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    translated = model.generate(**tokenizer(src_text, padding='max_length', max_length=32, return_tensors='pt'))
    print([tokenizer.convert_ids_to_tokens(t, skip_special_tokens=True) for t in translated])
    print()


def translate_en_to_fr_pt_es():
    src_text = [
        ">>fra<< this is a sentence in english that we want to translate to french",
        ">>por<< This should go to portuguese",
        ">>esp<< And this to Spanish",
    ]

    model_name = "Helsinki-NLP/opus-mt-en-roa"

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    translated = model.generate(**tokenizer(src_text, padding='max_length', max_length=32, return_tensors='pt'))
    print([tokenizer.convert_ids_to_tokens(t, skip_special_tokens=True) for t in translated])
    print()


if __name__ == "__main__":
    translate_en_to_fr()
    translate_en_to_fr_pt_es()
