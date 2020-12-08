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
    }
}
