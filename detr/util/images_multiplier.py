from PIL import Image, ImageEnhance


help(Image)
def multiply(image_path, xml_path):
    image = Image.open(image_path)