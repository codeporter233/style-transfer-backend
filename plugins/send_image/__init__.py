from flask import Blueprint

api = Blueprint("send_image", __name__)

from .send_image import *
