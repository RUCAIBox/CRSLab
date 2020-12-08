from crslab.download import DownloadableFile

resources = {
    'pkuseg': {
        'version': '0.14',
        'file': DownloadableFile('1YEiRrWLlBr1mJa8VsOSEA1-W8cNI5tu0', 'tgredial_pkuseg.zip',
                                 '8b7e23205778db4baa012eeb129cf8d26f4871ae98cdfe81fde6adc27a73a8d6',
                                 from_google=True),
        'special_token_idx': {
            'pad': 0,
            'start': 1,
            'end': 2,
            'unk': 3,
            'pad_entity': 0,
            'pad_word': 0,
            'pad_topic': 0
        },
    }
}
