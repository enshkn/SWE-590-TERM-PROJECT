import json
import cv2
import numpy as np
import base64
import os
from functions import converter, encode, decode
import json
import base64


def lambda_handler(event, context):

    img = decode(event["body"])
    processed_image = converter(img)
    img_encoded = cv2.imencode(".png", processed_image)[1]  # Extract the encoded image data from the tuple
    img_base64 = base64.b64encode(img_encoded)
    

    return {
        'statusCode': 200,
        'body': img_base64,
        'isBase64Encoded': True,
        'headers': {'content-type': 'image/png'}
    }


