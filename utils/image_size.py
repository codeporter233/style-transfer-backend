from PIL import Image


def get_size(path):
    img = Image.open(path)
    print(img.size)
    w = img.width
    h = img.height
    return (w, h)