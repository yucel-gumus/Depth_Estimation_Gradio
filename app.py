import gradio as gr
import torch
from PIL import Image
from misc import colorize

class DepthEstimationModel:
    def __init__(self) -> None:
        self.device = self._get_device()
        self.model = self._initialize_model(
            model_repo="isl-org/ZoeDepth", model_name="ZoeD_N"
        ).to(self.device)

    def _get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _initialize_model(self, model_repo="isl-org/ZoeDepth", model_name="ZoeD_N"):
        torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
        model = torch.hub.load(
         model_repo, model_name, pretrained=True, skip_validation=True  # skip_validation'ı True yaparak yüklemeyi zorlayabilirsiniz
        ).to(self.device)
        model.eval()
        print("Model initialized.")
        return model

    def save_colored_depth(self, depth_numpy):
        colored = colorize(depth_numpy)
        return Image.fromarray(colored)

    def calculate_depthmap(self, image):
        image = image.convert("RGB")
        print("Image read.")
        depth_numpy = self.model.infer_pil(image)
        return self.save_colored_depth(depth_numpy)

# Gradio arayüzü oluşturma
depth_estimator = DepthEstimationModel()

def predict(image):
    output_image = depth_estimator.calculate_depthmap(image)
    return output_image

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="image",
    title="Depth Estimation",
    description="Upload an image and get its depth estimation."
).launch()
