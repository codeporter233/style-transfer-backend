from flask import request

from . import api
from utils import makedir, thread
from deeplearning import train


@api.route('/sendImage', methods=["POST"])
def send_image():
    style_file = request.files["style"]
    content_file = request.files["content"]
    name = request.form.get("username")
    file_path = "E:/style-transfer-backend/user/" + name
    makedir.mkdir(file_path)
    style_path = file_path + "/style.JPG"
    content_path = file_path + "/content.JPG"
    style_file.save(style_path)
    content_file.save(content_path)
    # new_image_path = train.get_image(content_path, style_path, file_path)
    thread.executor.submit(new_thread, (content_path, style_path, file_path))
    return {
        "code": 0,
        "msg": "正在处理中"
    }


def new_thread(info: tuple):
    content_path = info[0]
    style_path = info[1]
    file_path = info[2]
    train.get_image(content_path, style_path, file_path)


# @api.route('/send_image', methods=["GET", "POST"])
# def send_image():
#     upload_file = request.files["file"]
#     name = request.form.get("username")
#     print(name)
#
#     file_name = "newPhoto"
#     file_path = "‪E:\style-transfer-backend\8K9A6688.JPG"
#     upload_file.save("E:/style-transfer-backend/newPhoto.JPG")
#     return "send_image"
