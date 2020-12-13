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
    'BERT': {
        'version': '0.1',
        'file': DownloadableFile('1Jc1A7whvzMOJInMeHjGzqAhRceMOXdee', 'redial_BERT.zip',
                                 '9a0317a102675b748cb31fe0e77e152c2b2e6ece3268bcade2a0cd549e99bba1',
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
            'endoftext': 50256  # to delete
        },
    },
    'GPT2': {
        'version': '0.1',
        'file': DownloadableFile('1Jc1A7whvzMOJInMeHjGzqAhRceMOXdee', 'redial_GPT2.zip',
                                 '9a0317a102675b748cb31fe0e77e152c2b2e6ece3268bcade2a0cd549e99bba1',
                                 from_google=True),
        'special_token_idx': {
            'pad': 50256,
            'start': 50256,
            'end': 50256,
            'unk': 2954,
            'sent_split': 50256,
            'word_split': 50256,
            'pad_entity': 0,
            'pad_word': 0
        },
    }
}
