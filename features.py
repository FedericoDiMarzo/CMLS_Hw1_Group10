def feature1(mfcc, windowed_audio, fs):
    """
    Example of feature using both mfcc and windowed_audio
    """
    return 0


def feature2(mfcc):
    """
    Example of feature using just mfcc
    """
    return 0


feature_functions = {
    """
    Used to store feature name and function reference
    """
    
    'feature1': feature1,
    'feature2': feature2,
}
