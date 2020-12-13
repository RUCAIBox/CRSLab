from crslab.download import DownloadableFile

resources = {
    'nltk': {
        'version': '0.12',
        'file': DownloadableFile('1Jc1A7whvzMOJInMeHjGzqAhRceMOXdee', 'redial_nltk.zip',
                                 '9a0317a102675b748cb31fe0e77e152c2b2e6ece3268bcade2a0cd549e99bba1',
                                 from_google=True),
        'special_token_idx': {
            'pad': 0,
            'start': 1,
            'end': 2,
            'unk': 3,
            'pad_entity': 0,
            'pad_word': 0
        },
    },
    'bert': {
        'version': '0.11',
        'file': DownloadableFile('1MJFZozsBSwfxSP6xjVvwapu47oFEH03E', 'redial_bert.zip',
                                 'ee10b26eb1ba04d23e508ee0f98ef1aa131507eae84847c23c3c6857e9c25c7f',
                                 from_google=True),
        'special_token_idx': {
            'pad': 0,
            'start': 101,
            'end': 102,
            'unk': 100,
            'sent_split': 2,
            'word_split': 3,
            'pad_entity': 0,
            'pad_word': 0,
        },
    },
    'gpt2': {
        'version': '0.11',
        'file': DownloadableFile('11WM3jPD_7VCOaM8HpxRUtQgQuq8y_V-z', 'redial_gpt2.zip',
                                 '3c3661ea8319c8195d9d8505558832e6d34a653e19a78f9943566aacf7bdd564',
                                 from_google=True),
        'special_token_idx': {
            'pad': 0,
            'start': 1,
            'end': 2,
            'unk': 3,
            'sent_split': 4,
            'word_split': 5,
            'pad_entity': 0,
            'pad_word': 0
        },
    }
}
