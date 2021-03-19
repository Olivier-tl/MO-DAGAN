
import logging

LOG_FILENAME = 'logs.txt'
loaded = False

def initializeLogger():
    # Get logger
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)

    # Define handlers
    file_handler = logging.FileHandler(LOG_FILENAME, 'w+')
    stream_handler = logging.StreamHandler()

    # Set levels
    stream_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)

    # Set formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handler
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

def getLogger():
    global loaded
    if not loaded:
        initializeLogger()
        loaded = True
    return logging.getLogger('logger')