import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# GRADIENT REVERSAL LAYER
# ============================================================

class GradientReversalFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lambda_value):

        ctx.lambda_value = lambda_value

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):

        return (
            -ctx.lambda_value * grad_output,
            None
        )


class GradientReversalLayer(nn.Module):

    def __init__(self, lambda_value=1.0):

        super().__init__()

        self.lambda_value = lambda_value

    def forward(self, x):

        return GradientReversalFunction.apply(
            x,
            self.lambda_value
        )


# ============================================================
# EEGNET FEATURE EXTRACTOR
# ============================================================

class EEGNetFeatureExtractor(nn.Module):

    def __init__(
        self,
        channels=3,
        samples=1000
    ):

        super().__init__()

        # ----------------------------------------------------
        # Temporal convolution
        # ----------------------------------------------------

        self.temporal = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(1, 64),
                padding=(0, 32),
                bias=False
            ),

            nn.BatchNorm2d(16)
        )

        # ----------------------------------------------------
        # Spatial convolution
        # ----------------------------------------------------

        self.spatial = nn.Sequential(

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(channels, 1),
                groups=16,
                bias=False
            ),

            nn.BatchNorm2d(32),

            nn.ELU(),

            nn.AvgPool2d(
                kernel_size=(1, 4)
            ),

            nn.Dropout(0.25)
        )

        # ----------------------------------------------------
        # Depthwise separable convolution
        # ----------------------------------------------------

        self.separable = nn.Sequential(

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(1, 16),
                groups=32,
                padding=(0, 8),
                bias=False
            ),

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(1, 1),
                bias=False
            ),

            nn.BatchNorm2d(32),

            nn.ELU(),

            nn.AvgPool2d(
                kernel_size=(1, 8)
            ),

            nn.Dropout(0.25)
        )

        # ----------------------------------------------------
        # Determine feature dimension automatically
        # ----------------------------------------------------

        with torch.no_grad():

            dummy = torch.zeros(
                1,
                1,
                channels,
                samples
            )

            dummy_out = self._forward_features(
                dummy
            )

            feature_dim = dummy_out.shape[1]

        self.feature_dim = feature_dim

        print(
            f"EEGNet feature dimension: "
            f"{feature_dim}"
        )

    def _forward_features(self, x):

        x = self.temporal(x)

        x = self.spatial(x)

        x = self.separable(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        return x

    def forward(self, x):

        return self._forward_features(x)


# ============================================================
# ADVERSARIAL TRANSFER LEARNING MODEL
# ============================================================

class AdversarialEEGNet(nn.Module):

    def __init__(
        self,
        channels=3,
        samples=1000,
        num_classes=2,
        num_domains=2,
        lambda_value=0.1
    ):

        super().__init__()

        # ----------------------------------------------------
        # Feature extractor
        # ----------------------------------------------------

        self.feature_extractor = EEGNetFeatureExtractor(
            channels=channels,
            samples=samples
        )

        feature_dim = (
            self.feature_extractor.feature_dim
        )

        # ----------------------------------------------------
        # Motor imagery classifier
        # ----------------------------------------------------

        self.class_classifier = nn.Sequential(

            nn.Linear(
                feature_dim,
                128
            ),

            nn.ELU(),

            nn.Dropout(0.5),

            nn.Linear(
                128,
                num_classes
            )
        )

        # ----------------------------------------------------
        # Gradient reversal
        # ----------------------------------------------------

        self.grl = GradientReversalLayer(
            lambda_value=lambda_value
        )

        # ----------------------------------------------------
        # Domain classifier
        # ----------------------------------------------------

        self.domain_classifier = nn.Sequential(

            nn.Linear(
                feature_dim,
                128
            ),

            nn.ELU(),

            nn.Dropout(0.5),

            nn.Linear(
                128,
                num_domains
            )
        )

    def forward(
        self,
        x,
        return_features=False
    ):

        # ----------------------------------------------------
        # Extract EEG features
        # ----------------------------------------------------

        features = self.feature_extractor(x)

        # ----------------------------------------------------
        # Motor imagery prediction
        # ----------------------------------------------------

        class_output = self.class_classifier(
            features
        )

        # ----------------------------------------------------
        # Domain prediction
        #
        # Gradient reversal is applied here.
        # ----------------------------------------------------

        reversed_features = self.grl(
            features
        )

        domain_output = self.domain_classifier(
            reversed_features
        )

        if return_features:

            return (
                class_output,
                domain_output,
                features
            )

        return (
            class_output,
            domain_output
        )

    def set_lambda(self, lambda_value):

        self.grl.lambda_value = (
            lambda_value
        )


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TESTING ADVERSARIAL EEGNET")
    print("=" * 60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nDevice:",
        device
    )

    model = AdversarialEEGNet(
        channels=3,
        samples=1000,
        num_classes=2,
        num_domains=2,
        lambda_value=0.1
    ).to(device)

    x = torch.randn(
        8,
        1,
        3,
        1000
    ).to(device)

    class_output, domain_output = (
        model(x)
    )

    print(
        "\nInput:",
        x.shape
    )

    print(
        "Class output:",
        class_output.shape
    )

    print(
        "Domain output:",
        domain_output.shape
    )

    print(
        "\nMODEL TEST SUCCESSFUL"
    )
