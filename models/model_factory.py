from .classification import EfficientNet

class ModelFactory:
    def create(model_name: str):
        if model_name == 'efficientnet':
            model = EfficientNet()
        elif model_name == 'wgan':
            model = None # TODO: Instantiate WGAN
        else:
            raise ValueError(f'model_name "{config["model_name"]}" '
                                f'in config "{config_path}" not recognized.')
        return model