import re
import unicodedata
from typing import Optional
from common_utils.load_files import load_file_txt


def remove_html(text: str) -> Optional[str]:
    return re.sub(r'<[^>]*>', '', text)


def unicode_normalize(text: str) -> Optional[str]:
    return unicodedata.normalize("NFC", text)


def remove_punctuation(text: str) -> Optional[str]:
    cleaned_text = re.sub(r'[^\w\s]', ' ', text)
    return cleaned_text


def lowercase(text: str) -> Optional[str]:
    return text.lower()


def remove_emoji(text: str) -> Optional[str]:
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoji emoticons
                               u"\U0001F300-\U0001F5FF"  # biểu tượng hình ảnh & ký hiệu
                               u"\U0001F680-\U0001F6FF"  # emoji transport & symbols
                               u"\U0001F1E0-\U0001F1FF"  # cờ các quốc gia
                               u"\U00002702-\U000027B0"  # emoji miscellaneous
                               u"\U000024C2-\U0001F251"  # emoji enclosed characters
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def teencode_normalize(text):
    """Change teencode to text"""
    text = text.lower()
    key_value_pairs = load_file_txt("dataset/teencode/teencode.txt")
    for e in key_value_pairs:
        key, value = e
        pattern = fr'(\b{key}\b)'
        text = re.sub(pattern, f'{value}', text)
    return text


def change_emoji_to_text(text):
    """ Change emoji to text """
    dict_emj = {
        "\U0001f604": "mặt_cười_thân_thiện ",
        "\U0001f600": "mặt_cười_giả_tạo ",
        "\U00002764": "trái_tim ",
        "\U0001F602": "cười_ra_nước_mắt ",
        "\U0001f636": "mặt_không_miệng ",
        "\U0001f611": "mặt_không_cảm_xúc "
    }

    for key in dict_emj.keys():
        # print(key)
        text_from_emoji = re.sub(fr'{key}', dict_emj[key], text)
        text = text_from_emoji
    return text


def preprocessing(text: str) -> Optional[str]:
    t = text.lower()
    t = remove_punctuation(t)
    t = remove_emoji(t)
    t = teencode_normalize(t)
    return t
