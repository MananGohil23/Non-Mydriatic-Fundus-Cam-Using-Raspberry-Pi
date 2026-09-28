import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def overlay_heatmap(image_bgr, cam, alpha=0.45):
    if cam is None:
        return None
    cam = cam.astype(np.float32)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    height, width = image_bgr.shape[:2]
    cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_CUBIC)
    color = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, color, alpha, 0)


def _softmax(values):
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


class StubInference:
    name = "stub"
    available = True

    def __init__(self, class_names, input_size=224):
        self.class_names = tuple(class_names)
        self.input_size = int(input_size)

    def _features(self, image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        detail = float(np.mean(np.sqrt(gx * gx + gy * gy)) / 255.0)
        b, g, r = image_bgr[..., 0].astype(np.float32), image_bgr[..., 1].astype(np.float32), image_bgr[..., 2].astype(np.float32)
        redness = float(np.mean(r - (g + b) / 2.0) / 255.0)
        return detail, redness

    def _cam(self, image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        energy = np.sqrt(gx * gx + gy * gy)
        energy = cv2.GaussianBlur(energy, (0, 0), sigmaX=max(3, min(image_bgr.shape[:2]) / 40))
        return energy

    def analyze(self, image_bgr, with_heatmap=True):
        detail, redness = self._features(image_bgr)
        raw = float(np.clip(detail * 4.0 + (0.5 - redness) * 0.4, 0.03, 0.97))
        if len(self.class_names) == 2:
            probabilities = {self.class_names[0]: 1.0 - raw, self.class_names[1]: raw}
        else:
            weights = _softmax([raw * (i + 1) for i in range(len(self.class_names))])
            probabilities = dict(zip(self.class_names, weights.tolist()))
        label = max(probabilities, key=probabilities.get)
        confidence = round(100.0 * probabilities[label], 1)
        referable_key = self.class_names[-1]
        risk = "referable" if label == referable_key else "low"
        heatmap = overlay_heatmap(image_bgr, self._cam(image_bgr)) if with_heatmap else None
        return {
            "label": label,
            "diagnosis": "Referable diabetic retinopathy suspected" if risk == "referable" else "No referable disease detected",
            "confidence": confidence,
            "probabilities": {k: round(100.0 * v, 2) for k, v in probabilities.items()},
            "risk": risk,
            "model": self.name,
            "heatmap": heatmap,
            "detail_score": round(detail, 4),
            "redness_score": round(redness, 4),
        }


class RetfoundInference:
    name = "retfound"
    available = True

    def __init__(self, checkpoint_path, model_arch, num_classes, class_names, input_size=224, device=None):
        import timm
        import torch

        self.torch = torch
        self.class_names = tuple(class_names)
        self.num_classes = int(num_classes)
        self.input_size = int(input_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = timm.create_model(
            model_arch, pretrained=False, num_classes=self.num_classes, img_size=self.input_size
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        state = {key.replace("module.", ""): value for key, value in state.items()}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        self.load_report = {"missing": len(missing), "unexpected": len(unexpected)}
        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, image_bgr):
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        tensor = self.torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)

    def _gradcam(self, tensor, target_index):
        torch = self.torch
        activations = {}
        gradients = {}

        def forward_hook(_module, _inputs, output):
            activations["value"] = output

        def backward_hook(_module, _grad_in, grad_out):
            gradients["value"] = grad_out[0]

        block = self.model.blocks[-1]
        target = getattr(block, "norm1", block)
        handle_f = target.register_forward_hook(forward_hook)
        handle_b = target.register_full_backward_hook(backward_hook)
        try:
            self.model.zero_grad(set_to_none=True)
            logits = self.model(tensor)
            logits[0, target_index].backward()
            act = activations.get("value")
            grad = gradients.get("value")
            if act is None or grad is None:
                return None
            act = act[0]
            grad = grad[0]
            if act.shape[0] > 1:
                act = act[1:]
                grad = grad[1:]
            weights = grad.mean(dim=0)
            cam = torch.relu((act * weights).sum(dim=-1))
            grid = int(round(cam.shape[0] ** 0.5))
            if grid * grid != cam.shape[0]:
                return None
            cam = cam.reshape(grid, grid).detach().cpu().numpy()
            return cam
        finally:
            handle_f.remove()
            handle_b.remove()

    def analyze(self, image_bgr, with_heatmap=True):
        torch = self.torch
        tensor = self._preprocess(image_bgr)
        with torch.no_grad():
            logits = self.model(tensor)[0].detach().cpu().numpy()
        probabilities = _softmax(logits)
        label = self.class_names[int(np.argmax(probabilities))]
        confidence = round(100.0 * float(np.max(probabilities)), 1)
        risk = "referable" if label == self.class_names[-1] else "low"
        heatmap = None
        if with_heatmap:
            cam = self._gradcam(tensor, int(np.argmax(probabilities)))
            heatmap = overlay_heatmap(image_bgr, cam)
        return {
            "label": label,
            "diagnosis": "Referable diabetic retinopathy suspected" if risk == "referable" else "No referable disease detected",
            "confidence": confidence,
            "probabilities": {
                name: round(100.0 * float(probabilities[i]), 2) for i, name in enumerate(self.class_names)
            },
            "risk": risk,
            "model": self.name,
            "heatmap": heatmap,
        }


def build_engine(settings):
    if settings.infer_mode == "retfound" and settings.checkpoint_path:
        try:
            return RetfoundInference(
                settings.checkpoint_path,
                settings.model_arch,
                settings.num_classes,
                settings.class_names,
                settings.input_size,
            )
        except Exception as exc:
            return StubInference(settings.class_names, settings.input_size), "retfound load failed: %s" % exc
    reason = "checkpoint not configured" if settings.infer_mode == "retfound" else None
    return StubInference(settings.class_names, settings.input_size), reason
