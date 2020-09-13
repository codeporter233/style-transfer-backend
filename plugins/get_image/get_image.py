from flask import request
import base64

from . import api


@api.route("/getImage", methods=["GET"])
def get_image():
    name = request.args.get("username", "")
    image_path = "E:/style-transfer-backend/user/" + name + "/newImage.JPG"
    with open(image_path, "rb") as f:
        res = base64.b64encode(f.read())
        print(type(res))
        return res