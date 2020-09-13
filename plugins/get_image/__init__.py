from flask import Blueprint

api = Blueprint("get_image", __name__)

from .get_image import *
