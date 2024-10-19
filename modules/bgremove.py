from rembg import remove

class BGRemove():
    def __init__(self, bg_output = "#ffffff", alpha=255):
        self.bg_output = self.hex_t_rgb(bg_output)
        self.bg_output.append(alpha)
    
    def hex_t_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        rgb = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
        return rgb
    
    def remove(self, image):
        image_out = remove(image,bgcolor=self.bg_output)
        return image_out