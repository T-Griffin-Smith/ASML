import os

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.uix.filechooser import FileChooserIconView
from kivy.properties import StringProperty
from kivy.uix.floatlayout import FloatLayout  # Import FloatLayout
from kivy.core.image import Image as CoreImage  # Import CoreImage for setting the icon

# Android imports
from kivy.utils import platform
if platform == 'android':
    from android.permissions import request_permissions, Permission
    from plyer import filechooser

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from torchvision import models  # Import torchvision models
from PIL import Image as PILImage

# Define the ResNetUNet architecture
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_rate=0.1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, dropout_rate=0.1):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout_rate=dropout_rate)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UpResNet(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, out_channels, bilinear=True, dropout_rate=0.1):
        super(UpResNet, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_1, in_channels_1, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels_1 + in_channels_2, out_channels, dropout_rate=dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ResNetUNet(nn.Module):
    def __init__(self, in_channels, out_channels, resnet_type="resnet50", bilinear=False, dropout_rate=0.1):
        super(ResNetUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resnet_type = resnet_type
        self.bilinear = bilinear
        self.dropout_rate = dropout_rate

        if self.resnet_type == "resnet50":
            self.backbone_model = models.resnet50(weights="DEFAULT")
            self.channel_distribution = [3, 64, 256, 512, 1024]
        elif self.resnet_type == "resnet34":
            self.backbone_model = models.resnet34(weights="DEFAULT")
            self.channel_distribution = [3, 64, 64, 128, 256]
        else:
            self.backbone_model = models.resnet18(weights="DEFAULT")
            self.channel_distribution = [3, 64, 64, 128, 256]

        self.backbone_layers = list(self.backbone_model.children())

        self.inc = DoubleConv(in_channels, 64)
        self.block1 = nn.Sequential(*self.backbone_layers[0:3])
        self.block2 = nn.Sequential(*self.backbone_layers[3:5])
        self.block3 = nn.Sequential(*self.backbone_layers[5])
        self.block4 = nn.Sequential(*self.backbone_layers[6])

        # Update the upsampling layers to match the training code
        self.up1 = Up(self.channel_distribution[-1], self.channel_distribution[-2], bilinear=self.bilinear, dropout_rate=dropout_rate)
        self.up2 = Up(self.channel_distribution[-2], self.channel_distribution[-3], bilinear=self.bilinear, dropout_rate=dropout_rate)
        self.up3 = UpResNet(self.channel_distribution[-3], 64, self.channel_distribution[-4], bilinear=self.bilinear, dropout_rate=dropout_rate)
        self.up4 = UpResNet(self.channel_distribution[-4], 64, self.channel_distribution[-4], bilinear=self.bilinear, dropout_rate=dropout_rate)
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        y1 = self.up1(x4, x3)
        y2 = self.up2(x3, x2)
        y3 = self.up3(x2, x1)
        y4 = self.up4(x1, x0)
        logits = self.outc(y4)
        return logits

# Instantiate and load the model
resnet_unet = ResNetUNet(in_channels=3, out_channels=2, resnet_type="resnet50").to("cpu")
resnet_unet.load_state_dict(torch.load("GoatEyesModel.pth", map_location="cpu"))
resnet_unet.eval()

# Update the preprocessing pipeline
preprocess = Compose([
    Resize((512, 512)),  # Match the model's expected input size
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def model_a_predict(image_path, threshold=0.5):
    """
    Predicts whether a goat is diseased based on a single image.
    
    Args:
        image_path (str): Path to the input image.
        threshold (float): Threshold for determining if the goat is diseased.
        
    Returns:
        str: Prediction result as a string.
    """
    try:
        # Use the globally defined resnet_unet model
        resnet_unet.eval()
        image = PILImage.open(image_path).convert("RGB")  # Use PILImage for consistency
        transformed_image = preprocess(image).unsqueeze(0).to("cpu")  # Add batch dimension and use preprocess

        with torch.no_grad():
            output = resnet_unet(transformed_image)  # Model output
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # Predicted mask

        is_diseased = any(1 in row for row in pred_mask)  # Check if any row contains a 1
        if is_diseased:
            return "Prediction: Goat is diseased."
        else:
            return "Prediction: Goat is healthy."
    except Exception as e:
        return f"Error processing image: {str(e)}"

def model_b_predict(image_path):
    return "Model B: Analyzed Image"

class MainLayout(BoxLayout):
    selected_image = StringProperty('')
    model_choice = StringProperty('GoatEyes')  # Default to GoatEyes

    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)

        # Create a horizontal BoxLayout for the buttons
        button_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.2))

        self.select_image_btn = Button(
            text = 'Select Image from Gallery', 
            halign = 'center', 
            valign = 'middle', 
            padding = (20, 20),
            background_color = (11/255, 35/255, 65/255, 1),  # Normal state color
            background_normal = '',  # Remove default background
            background_down = '',  # Remove default pressed background
        )
        self.select_image_btn.bind(size=self._update_text_size)  # Update text size dynamically
        self.select_image_btn.bind(on_release=self.select_image)

        self.model_toggle_btn = Button(
            text = 'Toggle Model Type\n(Goat Eye Disease Classification)',
            halign = 'center', 
            valign = 'middle', 
            padding = (20, 20),
            background_color = (11/255, 35/255, 65/255, 1),  # Normal state color
            background_normal = '',  # Remove default background
            background_down = '',  # Remove default pressed background
        )
        self.model_toggle_btn.bind(size=self._update_text_size)  # Update text size dynamically
        self.model_toggle_btn.bind(on_release=self.toggle_model)

        self.process_btn = Button(
            text = 'Process Image',
            halign = 'center', 
            valign = 'middle', 
            padding = (20, 20),
            background_color = (11/255, 35/255, 65/255, 1),  # Normal state color
            background_normal = '',  # Remove default background
            background_down = '',  # Remove default pressed background
        )
        self.process_btn.bind(size=self._update_text_size)  # Update text size dynamically
        self.process_btn.bind(on_release=self.process_image)

        # Dynamically set the pressed color
        for button in [self.select_image_btn, self.model_toggle_btn, self.process_btn]:
            button.bind(on_press=lambda instance: setattr(instance, 'background_color', (73/255, 90/255, 112/255, 1)))
            button.bind(on_release=lambda instance: setattr(instance, 'background_color', (11/255, 35/255, 65/255, 1)))

        # Add buttons to the horizontal layout
        button_layout.add_widget(self.select_image_btn)
        button_layout.add_widget(self.model_toggle_btn)
        button_layout.add_widget(self.process_btn)

        # Create a vertical BoxLayout for the result label and image layout
        result_and_image_layout = BoxLayout(orientation='vertical', size_hint=(1, 0.8))

        # Add the result label
        self.result_label = Label(text='Select an image ', size_hint=(1, 0.2), color=(255/255, 255/255, 255/255, 255/255), halign='center', valign='middle')  # Black text
        result_and_image_layout.add_widget(self.result_label)

        # Add a FloatLayout to position the Image widget
        image_layout = FloatLayout(size_hint=(1, 0.6))
        self.image_display = Image(size_hint=(0.8, 0.8))  # Adjust size
        image_layout.add_widget(self.image_display)

        # Center the image within the FloatLayout
        self.image_display.bind(size=lambda instance, value: instance.center.__setitem__(0, image_layout.center[0]))
        self.image_display.bind(size=lambda instance, value: instance.center.__setitem__(1, image_layout.center[1]))

        # Add the image layout to the result and image layout
        result_and_image_layout.add_widget(image_layout)

        # Add the horizontal layout and result/image layout to the main layout
        self.add_widget(button_layout)
        self.add_widget(result_and_image_layout)

    def _update_text_size(self, instance, value):
        instance.text_size = (instance.width, None)

    def select_image(self, instance):
        if platform == 'android':
            request_permissions([Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE])
            filechooser.open_file(on_selection=self.handle_selection, filters=[("Image Files", "*.png;*.jpg;*.jpeg")])
        else:
            # For desktop testing
            content = FileChooserIconView(filters=["*.png", "*.jpg", "*.jpeg"])
            popup = Popup(title="Select Image", content=content, size_hint=(1, 1))
            content.bind(on_submit=lambda chooser, selection, touch: (self.handle_selection(selection), popup.dismiss()))
            popup.open()

    def handle_selection(self, selection):
        if selection:
            self.selected_image = selection[0]
            self.result_label.text = f"Selected: {os.path.basename(self.selected_image)}"  # Extract file name
            self.result_label.text_size = (self.result_label.width, None)  # Allow text wrapping
            self.image_display.source = self.selected_image  # Update the image source

    def toggle_model(self, instance):
        if self.model_choice == 'GoatEyes':
            self.model_choice = 'TextOCR'
            self.model_toggle_btn.text = 'Toggle Model Type\n(Handwritten Text Transcription)'
        else:
            self.model_choice = 'GoatEyes'
            self.model_toggle_btn.text = 'Toggle Model Type\n(Goat Eye Disease Classification)'

    def process_image(self, instance):
        if not self.selected_image:
            self.result_label.text = "Please select an image first!"
            return
        
        if self.model_choice == 'GoatEyes':
            result = model_a_predict(self.selected_image)
        else:
            result = model_b_predict(self.selected_image)
        
        self.result_label.text = result

class MLApp(App):
    def build(self):

        self.aspect_ratio = 1440 / 2560
        window_width = 360
        window_height = int(window_width / self.aspect_ratio)
        Window.size = (window_width, window_height)  # Example: 800x600 pixels
        Window.resizable = True  # Disable resizing
        Window.clearcolor = (11/255, 35/255, 65/255, 255/255)  # Auburn Navy background

        Window.set_title("ASML 2025")
        icon_path = os.path.join(os.path.dirname(__file__), "goat.ico")
        if os.path.exists(icon_path):
            Window.set_icon(icon_path)
        else:
            print(f"Icon file not found: {icon_path}")

        Window.bind(on_resize=self._resize_with_aspect_ratio)
        return MainLayout()

    def _resize_with_aspect_ratio(self, window, width, height):
        target_width = int(width)
        target_height = int(width / self.aspect_ratio)

        if width / height > self.aspect_ratio:
            if Window.size != (target_width, height):  # Only update if size differs
                Window.size = (target_width, height)
        else:
            if Window.size != (width, target_height):  # Only update if size differs
                Window.size = (width, target_height)

if __name__ == '__main__':
    MLApp().run()